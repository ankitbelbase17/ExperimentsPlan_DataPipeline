import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import wandb
import os
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel

import config
from model import get_dit_model
from dataloader import get_dataloader
from utils import save_checkpoint

def get_text_encoder(device):
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    text_encoder.to(device).eval()
    text_encoder.requires_grad_(False)
    return text_encoder, tokenizer

def main():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Objective: Mean Flow Velocity (Global Displacement)")
    
    wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        config={
            "learning_rate": config.LEARNING_RATE,
            "architecture": "DiT-Base (250M)",
            "objective": "mean_flow_velocity",
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
    
    # 3. Training Loop
    global_step = 0
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        progress_bar = tqdm(dataloader)
        
        for step, batch in enumerate(progress_bar):
            dit_model.train()
            
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            
            with torch.no_grad():
                # Encode Image -> Latents z_1 (Data)
                latents = vae.encode(pixel_values).latent_dist.sample()
                z1 = latents * 0.18215
                
                # Encode Text
                encoder_hidden_states = text_encoder(input_ids)[0]
                
            loss = 0.0
            
            # --- Mean Flow Velocity Objective ---
            # We predict the global displacement vector v = z1 - z0
            # This represents the "average" velocity required to traverse from noise to data in time 1.
            
            bsz = z1.shape[0]
            z0 = torch.randn_like(z1) # Noise (t=0)
            
            # Interpolation time t ~ Uniform[0, 1]
            # Even though we want to predict the global mean velocity, we train on points along the path
            # to ensure the vector field is defined everywhere (or at least along the trajectories).
            t = torch.rand(bsz, device=device).view(bsz, 1, 1, 1)
            
            # Straight path interpolation: z_t
            zt = t * z1 + (1 - t) * z0
            
            # Target: The Global Mean Velocity (z1 - z0)
            target_v = z1 - z0
            
            # Timestep input scaling (0-1 -> 0-1000)
            timestep_input = t.view(bsz) * 1000 
            
            with autocast():
                # Model predicts v_theta(z_t, t)
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

            global_step += 1
            
        save_checkpoint(dit_model, optimizer, epoch, global_step, current_loss)

if __name__ == "__main__":
    main()
