import torch
import config
from diffusers import DDPMScheduler
import numpy as np

def run_inference(checkpoint_path, prompt, objective="diffusion", device="cuda"):
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
        dit_model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt['unet_state_dict']) # Handling potential key diff
    
    # Text Encoder
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder").to(device)
    
    # Encode Prompt
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        
    # Prepare Latents
    bsz = 1
    # 64x64 is default for 512px image with VAE f=8
    shape = (bsz, config.DIT_CONFIG["in_channels"], config.DIT_CONFIG["sample_size"], config.DIT_CONFIG["sample_size"])
    latents = torch.randn(shape, device=device)
    
    dit_model.eval()
    
    if objective == "diffusion":
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(50) 
        
        for t in scheduler.timesteps:
            # expand latents if classifier guidance etc? no just standard
            latent_model_input = latents
            t_input = t.view(1).to(device)
            
            with torch.no_grad():
                noise_pred = dit_model(latent_model_input, t_input, text_embeddings).sample
                
            if config.DIT_CONFIG["learn_sigma"]:
                noise_pred, _ = noise_pred.chunk(2, dim=1)
                
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            
    elif objective == "rectified_flow":
        # Euler sampling for ODE: dZ_t = v(Z_t, t) dt
        # Z_0 is noise, Z_1 is data. We go 0 -> 1.
        
        num_steps = 50
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = i / num_steps
            t_input = torch.tensor([t * 1000], device=device) # scaled to Model time
            
            with torch.no_grad():
                velocity = dit_model(latents, t_input, text_embeddings).sample
                
            if config.DIT_CONFIG["learn_sigma"]:
                velocity, _ = velocity.chunk(2, dim=1)
                
            # Euler step: Z_{t+dt} = Z_t + v * dt
            latents = latents + velocity * dt
            
    # Decode
    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample
        
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    from PIL import Image
    image = Image.fromarray((image[0] * 255).astype(np.uint8))
    
    return image

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a photo of a synthetic object")
    parser.add_argument("--objective", type=str, choices=["diffusion", "rectified_flow"], required=True)
    args = parser.parse_args()
    
    img = run_inference(args.checkpoint, args.prompt, args.objective)
    img.save(f"inference_{args.objective}.png")
    print(f"Saved inference_{args.objective}.png")
