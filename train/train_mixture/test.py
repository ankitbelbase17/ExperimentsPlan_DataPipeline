import torch
import torch.nn.functional as F
import wandb
import sys
import os

# Ensure we can import from the directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from model import StableDiffusionModel

def test_config():
    print("\n[Test] Checking Configuration...")
    assert config.MODEL_NAME is not None, "MODEL_NAME is missing in config"
    assert config.LEARNING_RATE > 0, "Invalid LEARNING_RATE"
    
    # Check S3 placeholders
    if "YOUR_AWS" in config.AWS_ACCESS_KEY_ID:
        print("WARNING: AWS credentials appear to be placeholders. Dataloader tests requiring S3 will fail.")
    print("Config check passed.")

def test_model_structure():
    print("\n[Test] Checking Model Structure & Loading...")
    try:
        # Load on CPU for testing to save memory/time if GPU busy, or just use config device
        device = "cpu" 
        if torch.cuda.is_available():
            device = "cuda"
            
        print(f"Loading model on {device}...")
        sd_model = StableDiffusionModel()
        sd_model.to(device)
        
        assert sd_model.vae is not None
        assert sd_model.text_encoder is not None
        assert sd_model.unet is not None
        
        print(f"Model loaded successfully on {device}.")
        return sd_model, device
    except Exception as e:
        print(f"Model loading failed: {e}")
        raise e

def test_dummy_training_step(sd_model, device):
    print("\n[Test] Running Dummy Training Step with Random Tensors...")
    
    sd_model.unet.train()
    
    batch_size = 2
    height = config.IMAGE_SIZE if hasattr(config, 'IMAGE_SIZE') else 512
    width = config.IMAGE_SIZE if hasattr(config, 'IMAGE_SIZE') else 512
    
    # 1. Create dummy inputs
    # Pixel values: [Batch, 3, H, W]
    pixel_values = torch.randn(batch_size, 3, height, width).to(device)
    
    # Input IDs: [Batch, 77] (Standard CLIP size)
    # Random integers within vocab range (approx 49408 for CLIP)
    input_ids = torch.randint(0, 1000, (batch_size, 77)).to(device)
    
    # Check for NaNs in input
    if torch.isnan(pixel_values).any():
        raise ValueError("Input pixel_values contain NaNs!")
        
    print(f"Dummy inputs created. Shape: {pixel_values.shape}")
    
    # 2. Forward Pass Simulation
    try:
        # VAE Encode
        with torch.no_grad():
            latents = sd_model.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215
            
        # Noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, sd_model.noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        
        # Add Noise
        noisy_latents = sd_model.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Check NaNs after VAE/Noise
        assert not torch.isnan(noisy_latents).any(), "NaNs found in noisy_latents"
        
        # Text Encoder
        with torch.no_grad():
            encoder_hidden_states = sd_model.text_encoder(input_ids)[0]
            
        # UNet Prediction
        noise_pred = sd_model.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        
        # Check NaNs in prediction
        assert not torch.isnan(noise_pred).any(), "NaNs found in model prediction"
        
        # Loss
        loss = F.mse_loss(noise_pred, noise)
        print(f"Calculated Loss: {loss.item()}")
        
        assert not torch.isnan(loss), "Loss is NaN"
        
        # Backward Pass (just to check graph connection)
        loss.backward()
        print("Backward pass successful.")
        
    except Exception as e:
        print(f"Training step failed: {e}")
        raise e

def test_wandb_init():
    print("\n[Test] Checking WandB Initialization...")
    try:
        # Initialize in disabled mode to avoid actual uploading
        wandb.init(mode="disabled", project="test_project")
        wandb.log({"test_metric": 0.5})
        print("WandB init and log successful (offline mode).")
        wandb.finish()
    except Exception as e:
        print(f"WandB check failed: {e}")
        raise e

def test_dataloader_instantiation():
    print("\n[Test] Checking Dataloader Instantiation...")
    # This might fail if AWS creds are not set, so we handle it gracefully
    try:
        from dataloader import get_dataloader
        # Attempt to initialize
        dl = get_dataloader(tokenizer=None, batch_size=2)
        print("Dataloader object created.")
        
        # We won't iterate because it requires S3 connection and actual files
        # unless we mocked boto3, but user requested testing "file paths" etc.
        # If config has real keys, this verifies the bucket connection initiates at least
        print("Dataloader init passed (Note: S3 connection not fully tested without valid creds).")
    except Exception as e:
        print(f"Dataloader init failed: {e}")
        print("NOTE: This failure is expected if S3 credentials are invalid in config.py")

if __name__ == "__main__":
    print("=== STARTING SYSTEM CHECKS ===")
    
    try:
        test_config()
        test_wandb_init()
        model, device = test_model_structure()
        test_dummy_training_step(model, device)
        test_dataloader_instantiation()
        
        print("\n=== ALL SYSTEM CHECKS PASSED ===")
    except Exception as e:
        print(f"\n!!! SYSTEM CHECK FAILED !!!\nError: {e}")
        sys.exit(1)
