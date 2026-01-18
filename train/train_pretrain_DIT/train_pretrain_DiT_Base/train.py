import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import wandb
import os
import argparse
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel

import config
from model import get_dit_model
from dataloader import get_dataloader
from utils import save_checkpoint, load_latest_checkpoint

# We need a Text Encoder since we added Cross Attention to DiT
def get_text_encoder(device):
    # SD1.5 Text Encoder
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    text_encoder.to(device).eval()
    text_encoder.requires_grad_(False)
    return text_encoder, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", type=str, choices=["diffusion", "rectified_flow"], required=True, 
                        help="Training objective: 'diffusion' or 'rectified_flow'")
    args = parser.parse_args()
    
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Objective: {args.objective}")
    
    wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        config={
            "learning_rate": config.LEARNING_RATE,
            "architecture": "DiT-Base (250M)",
            "objective": args.objective,
            "epochs": config.NUM_EPOCHS,
        }
    )

    # 2. Model
    dit_model, vae = get_dit_model()
    dit_model.to(device)
    vae.to(device)
    
    text_encoder, tokenizer = get_text_encoder(device)
    
    # Dataloader
    dataloader = get_dataloader(tokenizer)
    
    # Optimizer
    optimizer = AdamW(dit_model.parameters(), lr=config.LEARNING_RATE)
    
    # Mixed Precision
    scaler = GradScaler()
    
    # Resume
    start_epoch, global_step = load_latest_checkpoint(dit_model, optimizer, scaler)
    
    # 3. Training Loop
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        progress_bar = tqdm(dataloader)
        
        for step, batch in enumerate(progress_bar):
            dit_model.train()
            
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            
            with torch.no_grad():
                # Encode Image -> Latents
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215
                
                # Encode Text
                encoder_hidden_states = text_encoder(input_ids)[0]
                
            loss = 0.0
            
            # --- Objective Logic ---
            if args.objective == "diffusion":
                # Standard DDPM/IDDPM logic
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Random timestep
                timesteps = torch.randint(0, 1000, (bsz,), device=device).long()
                
                # Add noise (would need scheduler logic, simplified here for scratch impl)
                # Ideally use DDPMScheduler to add noise, but implementing calculation manually for clarity/dependency-free
                # For consistency with diffusers, let's assume we grabbed a scheduler.
                # Since we didn't pass scheduler to main, let's do a simple linear schedule approx or load one.
                # Better: Use diffusers DDPMScheduler
                from diffusers import DDPMScheduler
                noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Predict
                with autocast():
                    # DiT forward: sample, timestep class/emb, encoder_hidden_states
                    model_pred = dit_model(noisy_latents, timesteps, encoder_hidden_states).sample
                    
                    # Split prediction if learning sigma (channels*2)
                    if config.DIT_CONFIG["learn_sigma"]:
                        model_pred, _ = model_pred.chunk(2, dim=1)
                        
                    loss = F.mse_loss(model_pred, noise)
                    
            elif args.objective == "rectified_flow":
                # Rectified Flow Logic
                # t ~ Uniform[0, 1]
                bsz = latents.shape[0]
                t = torch.rand(bsz, device=device).view(bsz, 1, 1, 1)
                
                z0 = torch.randn_like(latents) # Noise
                z1 = latents # Data
                
                # Interpolation
                zt = t * z1 + (1 - t) * z0
                
                # Target v = z1 - z0
                target_v = z1 - z0
                
                # For DiT conditioning, t is usually expected as timestep [0, 1000] or [0, 1] embedding
                # diffusers Transformer2DModel expects `timestep` arg.
                # We can pass t * 1000 to map 0-1 to 0-1000 scope if model expects standard embedding
                timestep_input = t.view(bsz) * 1000 
                
                with autocast():
                    model_pred = dit_model(zt, timestep_input, encoder_hidden_states).sample
                    
                    if config.DIT_CONFIG["learn_sigma"]:
                        model_pred, _ = model_pred.chunk(2, dim=1)
                        
                    loss = F.mse_loss(model_pred, target_v)

            # Optimization
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Logging
            current_loss = loss.item()
            wandb.log({"train_loss": current_loss, "epoch": epoch}, step=global_step)
            progress_bar.set_postfix(loss=current_loss)
            
            # Periodic Inference handled in inference.sh calls or separate thread ideally.
            # We skip inline inference here to keep file complexity low, relying on inference.sh

            global_step += 1
            
        save_checkpoint(dit_model, optimizer, epoch, global_step, current_loss, scaler)

if __name__ == "__main__":
    main()
