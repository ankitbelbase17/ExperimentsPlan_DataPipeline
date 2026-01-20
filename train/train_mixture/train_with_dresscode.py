"""
Training script for train_mixture with support for multiple dataloaders
Usage:
  - Default (original): python train.py
  - DressCODE from S3: python train.py --datasource dresscode
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import wandb
import os
from tqdm import tqdm
import argparse

import config
from model import StableDiffusionModel
from dataloader import get_dataloader
from dataloader_dresscode_s3 import get_dresscode_dataloader
from utils import save_checkpoint, load_latest_checkpoint
from inference import run_inference
import metrics  # Placeholders


def main(args):
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    datasource = args.datasource or "mixture"
    print(f"Using datasource: {datasource}")
    
    # WandB
    wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        config={
            "learning_rate": config.LEARNING_RATE,
            "architecture": "Stable Diffusion v1.5",
            "dataset": f"dataset_{datasource}",
            "epochs": config.NUM_EPOCHS,
            "batch_size": config.BATCH_SIZE,
        }
    )

    # 2. Model & Data
    sd_model = StableDiffusionModel()
    sd_model.to(device)
    
    # Load appropriate dataloader
    if datasource == "dresscode":
        print("Loading DressCODE dataset from S3...")
        dataloader = get_dresscode_dataloader(
            batch_size=config.BATCH_SIZE,
            split='train',
            categories=['dresses', 'lower_body', 'upper_body']
        )
    else:
        print("Loading mixture dataset...")
        dataloader = get_dataloader(sd_model.tokenizer)
    
    print(f"Dataset size: {len(dataloader.dataset)}")
    
    # Optimizer
    optimizer = AdamW(sd_model.unet.parameters(), lr=config.LEARNING_RATE)
    
    # Mixed Precision
    scaler = GradScaler()
    
    # Resume from checkpoint if available
    start_epoch, global_step = load_latest_checkpoint(sd_model, optimizer, scaler)
    
    # 3. Training Loop
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        progress_bar = tqdm(dataloader)
        
        for step, batch in enumerate(progress_bar):
            sd_model.unet.train()
            
            # Prepare inputs based on datasource
            if datasource == "dresscode":
                # DressCODE dataloader returns: ground_truth, cloth, mask
                pixel_values = batch["ground_truth"].to(device)
                cloth = batch["cloth"].to(device)
                mask = batch["mask"].to(device)
                
                # For now, use ground_truth as the training target
                # You can modify this to use cloth or apply mask for different training strategies
                
                # Create dummy text embeddings
                with torch.no_grad():
                    dummy_input_ids = torch.zeros((pixel_values.shape[0], 77), dtype=torch.long, device=device)
                    encoder_hidden_states = sd_model.text_encoder(dummy_input_ids)[0]
            else:
                # Original mixture dataloader
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                
                # Get text embeddings
                with torch.no_grad():
                    encoder_hidden_states = sd_model.text_encoder(input_ids)[0]
            
            # Convert images to latent space
            with torch.no_grad():
                latents = sd_model.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215
            
            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, 
                sd_model.noise_scheduler.config.num_train_timesteps, 
                (bsz,), 
                device=device
            ).long()
            
            # Add noise
            noisy_latents = sd_model.noise_scheduler.add_noise(latents, noise, timesteps)
            
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
                print(f"\nRunning validation/inference at step {global_step}...")
                
                # Inference
                # Save temp checkpoint for inference
                temp_ckpt_path = os.path.join(config.OUTPUT_DIR, "temp_inference.ckpt")
                torch.save({'unet_state_dict': sd_model.unet.state_dict()}, temp_ckpt_path)
                
                # Generate sample
                gen_image = run_inference(
                    checkpoint_path=temp_ckpt_path,
                    prompt="a person wearing cloth",
                    device=device
                )
                
                wandb.log({
                    "inference_image": wandb.Image(gen_image, caption=f"Step {global_step}")
                }, step=global_step)
                
                # Placeholder Metrics Logging
                wandb.log({"FID": 0.0, "SSIM": 0.0}, step=global_step)

            global_step += 1
            
        # End of Epoch
        save_checkpoint(sd_model, optimizer, epoch, global_step, current_loss, scaler)

    # Close WandB
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stable Diffusion model")
    parser.add_argument(
        "--datasource",
        type=str,
        default="mixture",
        choices=["mixture", "dresscode"],
        help="Data source to use for training (default: mixture)"
    )
    args = parser.parse_args()
    
    main(args)
