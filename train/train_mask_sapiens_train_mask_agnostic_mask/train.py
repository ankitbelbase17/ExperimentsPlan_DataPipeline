import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import wandb
import os
from tqdm import tqdm

import config
from model import SapiensModel
from dataloader import get_dataloader
from utils import save_checkpoint

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Training: Masked Sapiens via Agnostic Mask")
    
    wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        config={
            "learning_rate": config.LEARNING_RATE,
            "architecture": "SD1.5-Inpainting + CLIP-Vision Condition",
            "epochs": config.NUM_EPOCHS,
        }
    )

    # Model
    model = SapiensModel()
    model.to(device)
    model.unet.train() # Train UNet
    
    dataloader = get_dataloader()
    
    optimizer = AdamW(model.unet.parameters(), lr=config.LEARNING_RATE)
    scaler = GradScaler()
    
    global_step = 0
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        progress_bar = tqdm(dataloader)
        
        for step, batch in enumerate(progress_bar):
            # Input
            person_pixels = batch["person_pixel_values"].to(device) # [-1, 1]
            cloth_pixels = batch["cloth_pixel_values"].to(device) # [0, 1] for CLIP?
            mask = batch["mask"].to(device) # [0, 1], 1=mask
            
            # 1. Encode Person -> Latents (GT)
            with torch.no_grad():
                latents = model.vae.encode(person_pixels).latent_dist.sample()
                latents = latents * 0.18215
            
            # 2. Encode Mask
            # Resize mask to latent shape (64x64)
            mask_down = F.interpolate(mask, size=(64, 64), mode="nearest")
            
            # 3. Create Masked Image Latents
            # masked_latents = latents * (1 - mask_down)
            masked_latents = latents * (1 - mask_down)
            
            # 4. Encode Cloth -> Conditioning
            with torch.no_grad():
                # CLIP Vision Model expects pixel_values
                # We assume batch['cloth_pixel_values'] is reasonable input
                # Ideally, normalize with mean/std of CLIP
                # For simplicity here, passing tensor.
                cloth_emb = model.clip_image_encoder(cloth_pixels).last_hidden_state
                # cloth_emb shape: [B, 257, 1024] usually (256 tokens + CLS)
                
            # 5. Noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, model.noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            
            noisy_latents = model.noise_scheduler.add_noise(latents, noise, timesteps)
            
            # 6. Prepare UNet Input
            # Concatenate noise, mask, masked_latents
            # Shape: [B, 9, 64, 64]
            latent_model_input = torch.cat([noisy_latents, mask_down, masked_latents], dim=1)
            
            # 7. Predict
            with autocast():
                # Pass cloth_emb as encoder_hidden_states
                noise_pred = model.unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=cloth_emb
                ).sample
                
                loss = F.mse_loss(noise_pred, noise)
                
            # Optimization
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Logging
            wandb.log({"train_loss": loss.item()}, step=global_step)
            progress_bar.set_postfix(loss=loss.item())
            
            global_step += 1
            
        save_checkpoint(model, optimizer, epoch, global_step, loss.item())

if __name__ == "__main__":
    main()
