import torch
import config
import numpy as np
from PIL import Image

def run_inference(checkpoint_path, prompt, device="cuda"):
    from model import get_dit_model
    from diffusers import AutoencoderKL
    from transformers import CLIPTokenizer, CLIPTextModel
    
    # Load Model
    dit_model, vae = get_dit_model()
    dit_model.to(device)
    vae.to(device)
    
    if checkpoint_path:
        print(f"Loading checkpoint {checkpoint_path}")
        ckpt = torch.load(checkpoint_path)
        dit_model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt['unet_state_dict'])
    
    # Text Encoder
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder").to(device)
    
    # Encode Prompt
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        
    # Prepare Latents
    bsz = 1
    shape = (bsz, config.DIT_CONFIG["in_channels"], config.DIT_CONFIG["sample_size"], config.DIT_CONFIG["sample_size"])
    z0 = torch.randn(shape, device=device)
    
    dit_model.eval()
    
    # --- Single Step Inference (Mean Flow) ---
    # The model has been trained to predict the mean velocity v = z1 - z0
    # Therefore, the relationship is z1 = z0 + v
    # We query the model at t=0 (start of flow)
    
    t_input = torch.tensor([0.0], device=device) # t=0
    
    with torch.no_grad():
        # Predict Mean Velocity
        velocity = dit_model(z0, t_input, text_embeddings).sample
        
    if config.DIT_CONFIG["learn_sigma"]:
        velocity, _ = velocity.chunk(2, dim=1)
        
    # Apply Mean Velocity for total duration dt=1
    z1 = z0 + velocity * 1.0
            
    # Decode
    with torch.no_grad():
        latents = 1 / 0.18215 * z1
        image = vae.decode(latents).sample
        
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = Image.fromarray((image[0] * 255).astype(np.uint8))
    
    return image

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a photo of a synthetic object")
    args = parser.parse_args()
    
    img = run_inference(args.checkpoint, args.prompt)
    img.save("inference_fast_1step.png")
    print("Saved inference_fast_1step.png")
