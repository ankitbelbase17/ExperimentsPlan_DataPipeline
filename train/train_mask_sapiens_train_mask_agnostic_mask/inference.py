import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import DDPMScheduler
import config
from model import SapiensModel

def run_inference(checkpoint_path, person_path, cloth_path, mask_path, device="cuda"):
    # Load Model
    model = SapiensModel()
    model.to(device)
    
    if checkpoint_path:
        print(f"Loading checkpoint {checkpoint_path}")
        ckpt = torch.load(checkpoint_path)
        # Load UNet weights
        model.unet.load_state_dict(ckpt['unet_state_dict'] if 'unet_state_dict' in ckpt else ckpt['model_state_dict']) # Handling potential keys
    
    model.unet.eval()
    
    # Load Inputs
    def load_img(p, gray=False):
        img = Image.open(p)
        if gray:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        img = img.resize((512, 512))
        return img
        
    person_img = load_img(person_path)
    cloth_img = load_img(cloth_path)
    mask_img = load_img(mask_path, gray=True) # Agnostic mask
    
    # Preprocess
    from torchvision import transforms
    tf_img = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    tf_mask = transforms.Compose([transforms.ToTensor()]) # 0-1
    tf_clip = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor()])
    
    person_tensor = tf_img(person_img).unsqueeze(0).to(device)
    mask_tensor = tf_mask(mask_img).unsqueeze(0).to(device)
    cloth_tensor = tf_clip(cloth_img).unsqueeze(0).to(device)
    
    # Encode Conditionings
    with torch.no_grad():
        # Latents (GT Person)
        latents = model.vae.encode(person_tensor).latent_dist.sample() * 0.18215
        
        # Mask Downsample
        mask_down = F.interpolate(mask_tensor, size=(64, 64), mode="nearest")
        
        # Masked Latents (Source content where mask is 0)
        # Assuming mask 1 = area to repaint (cloth area)
        masked_latents = latents * (1 - mask_down)
        
        # Cloth Embedding
        cloth_emb = model.clip_image_encoder(cloth_tensor).last_hidden_state
        
    # Standard Inference Loop
    scheduler = DDPMScheduler.from_pretrained(config.MODEL_NAME, subfolder="scheduler")
    scheduler.set_timesteps(50)
    
    # Initial Noise for the Inpainting area (or whole image)
    # Ideally, we start with random noise.
    # Standard Inpainting Pipeline inputs: noise + mask + masked_image
    
    latents_gen = torch.randn_like(latents)
    
    for t in scheduler.timesteps:
        # 1. Prepare UNet Input
        # In diffusers Inpaint pipeline, input is often: cat(latents_gen, mask, masked_latents)
        # We must follow training concatenation order: [noisy_latents, mask, masked_latents]
        
        # Note: If we use a scheduler that scales inputs, apply it. DDPM doesn't scaling much.
        
        unet_input = torch.cat([latents_gen, mask_down, masked_latents], dim=1)
        
        with torch.no_grad():
            noise_pred = model.unet(unet_input, t, encoder_hidden_states=cloth_emb).sample
            
        latents_gen = scheduler.step(noise_pred, t, latents_gen).prev_sample
        
    # Decode
    with torch.no_grad():
        image = model.vae.decode(latents_gen / 0.18215).sample
        
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    result = Image.fromarray((image[0] * 255).astype(np.uint8))
    
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--person", type=str, required=True)
    parser.add_argument("--cloth", type=str, required=True)
    parser.add_argument("--mask", type=str, required=True)
    args = parser.parse_args()
    
    img = run_inference(args.checkpoint, args.person, args.cloth, args.mask)
    img.save("inference_result.png")
    print("Saved inference_result.png")
