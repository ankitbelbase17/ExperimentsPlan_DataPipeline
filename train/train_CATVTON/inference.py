"""
CATVTON Inference Script
Loads latest checkpoint and generates try-on results
"""
import torch
import os
import argparse
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import config
from model import get_catvton_model
from utils import load_checkpoint
import boto3
from io import BytesIO


def parse_args():
    parser = argparse.ArgumentParser(description="CATVTON Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (local or s3://)")
    parser.add_argument("--test_dir", type=str, default="dataset_test", help="Test dataset directory")
    parser.add_argument("--output_dir", type=str, default="inference_outputs_catvton", help="Output directory")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    return parser.parse_args()


def load_image_from_path(path, size=512):
    """Load image from local path or S3"""
    if path.startswith("s3://"):
        # Parse S3 path
        path_parts = path[5:].split('/', 1)
        bucket = path_parts[0]
        key = path_parts[1]
        
        s3_client = boto3.client('s3',
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION
        )
        
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_data = response['Body'].read()
        image = Image.open(BytesIO(image_data)).convert('RGB')
    else:
        image = Image.open(path).convert('RGB')
    
    # Resize
    image = image.resize((size, size), Image.LANCZOS)
    return image


def get_test_pairs(test_dir):
    """Scan test directory for person-garment pairs"""
    pairs = []
    
    # Expected structure: test_dir/initial_image/, test_dir/cloth_image/, test_dir/try_on_image/
    person_dir = os.path.join(test_dir, "initial_image")
    cloth_dir = os.path.join(test_dir, "cloth_image")
    
    if not os.path.exists(person_dir) or not os.path.exists(cloth_dir):
        print(f"Warning: {person_dir} or {cloth_dir} not found")
        return pairs
    
    # Get all person images
    person_files = [f for f in os.listdir(person_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for person_file in person_files:
        # Extract stem (e.g., "001_person.png" -> "001")
        stem = person_file.split('_')[0]
        
        # Find matching cloth image
        cloth_files = [f for f in os.listdir(cloth_dir) if f.startswith(stem)]
        
        if cloth_files:
            pairs.append({
                'person': os.path.join(person_dir, person_file),
                'cloth': os.path.join(cloth_dir, cloth_files[0]),
                'stem': stem
            })
    
    return pairs


@torch.no_grad()
def inference(model, person_img, garment_img, num_steps=50, guidance_scale=7.5, device='cuda'):
    """
    Run inference to generate try-on result
    
    Args:
        model: CATVTON model
        person_img: PIL Image
        garment_img: PIL Image
        num_steps: Number of denoising steps
        guidance_scale: CFG scale
        device: Device
    
    Returns:
        result_img: PIL Image
    """
    # Transform images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    person_tensor = transform(person_img).unsqueeze(0).to(device)
    garment_tensor = transform(garment_img).unsqueeze(0).to(device)
    
    # Encode to latent space
    person_latents = model.encode_images(person_tensor)
    garment_latents = model.encode_images(garment_tensor)
    
    # Extract garment features
    garment_features = model.garment_encoder(garment_latents)
    B, C, H, W = garment_features.shape
    garment_features_flat = garment_features.view(B, C, -1).permute(0, 2, 1)
    garment_features_proj = model.garment_proj(garment_features_flat)
    
    # Text embeddings
    caption = "a person wearing a garment"
    inputs = model.tokenizer(
        [caption],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    text_embeddings = model.text_encoder(inputs.input_ids.to(device))[0]
    
    # Combined conditioning
    combined_embeddings = torch.cat([text_embeddings, garment_features_proj], dim=1)
    
    # Initialize with random noise
    latents = torch.randn_like(person_latents)
    
    # Set timesteps
    model.noise_scheduler.set_timesteps(num_steps)
    
    # Denoising loop
    for t in tqdm(model.noise_scheduler.timesteps, desc="Denoising"):
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        
        # Predict noise
        noise_pred = model.unet(
            latent_model_input,
            t,
            encoder_hidden_states=torch.cat([combined_embeddings, combined_embeddings])
        ).sample
        
        # Perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Compute previous noisy sample
        latents = model.noise_scheduler.step(noise_pred, t, latents).prev_sample
    
    # Decode latents
    latents = latents / 0.18215
    images = model.vae.decode(latents).sample
    
    # Convert to PIL
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")
    result_img = Image.fromarray(images[0])
    
    return result_img


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = get_catvton_model()
    model.to(device)
    model.eval()
    
    # Load checkpoint
    load_checkpoint(args.checkpoint, model)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get test pairs
    print(f"Scanning test directory: {args.test_dir}")
    test_pairs = get_test_pairs(args.test_dir)
    print(f"Found {len(test_pairs)} test pairs")
    
    # Run inference
    for pair in tqdm(test_pairs, desc="Generating try-ons"):
        person_img = load_image_from_path(pair['person'], config.IMAGE_SIZE)
        garment_img = load_image_from_path(pair['cloth'], config.IMAGE_SIZE)
        
        result_img = inference(
            model, 
            person_img, 
            garment_img,
            num_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            device=device
        )
        
        # Save result
        output_path = os.path.join(args.output_dir, f"{pair['stem']}_result.png")
        result_img.save(output_path)
        print(f"Saved: {output_path}")
    
    print(f"Inference complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
