"""
Test script to verify gradient backpropagation using DressCODE dataset
This tests a single batch to ensure the training pipeline works correctly.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import os
from tqdm import tqdm

import config
from model import StableDiffusionModel
from dataloader_dresscode_s3 import get_dresscode_dataloader
from utils import save_checkpoint, load_latest_checkpoint


def test_single_batch():
    """
    Test gradient backpropagation on a single batch from DressCODE dataset
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Model
    print("\n[1] Loading Stable Diffusion Model...")
    sd_model = StableDiffusionModel()
    sd_model.to(device)
    print("✓ Model loaded successfully")
    
    # 2. Load DressCODE Dataset
    print("\n[2] Loading DressCODE dataset from S3...")
    try:
        dataloader = get_dresscode_dataloader(batch_size=1, split='train', categories=['dresses'])
        print(f"✓ DataLoader created with {len(dataloader.dataset)} samples")
    except Exception as e:
        print(f"✗ Error loading dataloader: {e}")
        return False
    
    # 3. Get a single batch
    print("\n[3] Fetching a single batch...")
    try:
        batch = next(iter(dataloader))
        print(f"✓ Batch loaded successfully")
        print(f"  - Ground truth shape: {batch['ground_truth'].shape}")
        print(f"  - Cloth shape: {batch['cloth'].shape}")
        print(f"  - Mask shape: {batch['mask'].shape}")
    except Exception as e:
        print(f"✗ Error loading batch: {e}")
        return False
    
    # 4. Setup optimizer and scaler
    print("\n[4] Setting up optimizer and mixed precision...")
    optimizer = AdamW(sd_model.unet.parameters(), lr=config.LEARNING_RATE)
    scaler = GradScaler()
    print("✓ Optimizer and scaler initialized")
    
    # 5. Forward pass with ground truth image
    print("\n[5] Running forward pass...")
    try:
        ground_truth = batch["ground_truth"].to(device)
        
        # Convert image to latent space
        with torch.no_grad():
            latents = sd_model.vae.encode(ground_truth).latent_dist.sample()
            latents = latents * 0.18215
        
        print(f"✓ Image encoded to latent space: {latents.shape}")
        
        # Sample noise and timesteps
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
        print(f"✓ Noise added to latents: {noisy_latents.shape}")
        
        # Get text embedding (dummy caption)
        caption = "a person wearing cloth"
        inputs = sd_model.tokenizer(
            caption,
            max_length=sd_model.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(device)
        
        with torch.no_grad():
            encoder_hidden_states = sd_model.text_encoder(input_ids)[0]
        
        print(f"✓ Text encoded: {encoder_hidden_states.shape}")
        
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. Backward pass
    print("\n[6] Running backward pass...")
    try:
        optimizer.zero_grad()
        
        with autocast():
            # Predict noise residual
            noise_pred = sd_model.unet(
                noisy_latents, 
                timesteps, 
                encoder_hidden_states
            ).sample
            loss = F.mse_loss(noise_pred, noise, reduction="mean")
        
        print(f"✓ Loss computed: {loss.item():.6f}")
        
        # Backward
        scaler.scale(loss).backward()
        print("✓ Backward pass completed")
        
        # Check gradients
        total_grad_norm = 0.0
        for param in sd_model.unet.parameters():
            if param.grad is not None:
                total_grad_norm += torch.norm(param.grad).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        print(f"✓ Total gradient norm: {total_grad_norm:.6f}")
        
        if total_grad_norm == 0:
            print("✗ WARNING: Gradient norm is zero! Gradients may not be backpropagating.")
            return False
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        print("✓ Optimizer step completed")
        
    except Exception as e:
        print(f"✗ Error in backward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 7. Summary
    print("\n" + "="*60)
    print("✓ GRADIENT BACKPROPAGATION TEST PASSED!")
    print("="*60)
    print(f"Loss: {loss.item():.6f}")
    print(f"Gradient norm: {total_grad_norm:.6f}")
    print("\nThe training pipeline is working correctly.")
    print("You can now use this dataloader with train.py")
    
    return True


if __name__ == "__main__":
    success = test_single_batch()
    exit(0 if success else 1)
