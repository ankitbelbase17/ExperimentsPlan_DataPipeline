import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import wandb
import os
from tqdm import tqdm

import config
from model import StableDiffusionModel
from dataloader import get_dataloader
from utils import save_checkpoint
from inference import run_inference
import metrics  # Placeholders

def main():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # WandB
    wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        config={
            "learning_rate": config.LEARNING_RATE,
            "architecture": "Stable Diffusion v1.5",
            "dataset": "dataset_stage_1",
            "epochs": config.NUM_EPOCHS,
        }
    )

    # 2. Model & Data
    sd_model = StableDiffusionModel()
    sd_model.to(device)
    
    dataloader = get_dataloader(sd_model.tokenizer)
    
    # Optimizer
    optimizer = AdamW(sd_model.unet.parameters(), lr=config.LEARNING_RATE)
    
    # Mixed Precision
    scaler = GradScaler()
    
    # 3. Training Loop
    global_step = 0
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        progress_bar = tqdm(dataloader)
        
        for step, batch in enumerate(progress_bar):
            sd_model.unet.train()
            
            # Prepare inputs
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            
            # Convert images to latent space
            latents = sd_model.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215
            
            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, sd_model.noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
            timesteps = timesteps.long()
            
            # Add noise
            noisy_latents = sd_model.noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get text embeddings
            encoder_hidden_states = sd_model.text_encoder(input_ids)[0]
            
            # Train step with Mixed Precision
            optimizer.zero_grad()
            
            with autocast():
                # Predict the noise residual
                noise_pred = sd_model.unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred, noise, reduction="mean")
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Logging
            current_loss = loss.item()
            wandb.log({"train_loss": current_loss, "epoch": epoch}, step=global_step)
            progress_bar.set_postfix(loss=current_loss)
            
            # Periodic Inference & Metrics (Every 100 steps)
            if global_step % config.METRICS_INTERVAL == 0:
                print(f"Running validation/inference at step {global_step}...")
                
                # Inference
                # Save temp checkpoint for inference
                temp_ckpt_path = os.path.join(config.OUTPUT_DIR, "temp_inference.ckpt")
                torch.save({'unet_state_dict': sd_model.unet.state_dict()}, temp_ckpt_path)
                
                # Generate sample
                gen_image = run_inference(
                    checkpoint_path=temp_ckpt_path,
                    prompt="a photo of a synthetic object",
                    device=device
                )
                
                wandb.log({
                    "inference_image": wandb.Image(gen_image, caption=f"Step {global_step}")
                }, step=global_step)
                
                # Placeholder Metrics Logging
                # fid = metrics.calculate_fid(...)
                # wandb.log({"FID": fid, "SSIM": ...}, step=global_step)
                wandb.log({"FID": 0.0, "SSIM": 0.0}, step=global_step) # Placeholder

            global_step += 1
            
        # End of Epoch
        save_checkpoint(sd_model, optimizer, epoch, global_step, current_loss)

if __name__ == "__main__":
    main()
