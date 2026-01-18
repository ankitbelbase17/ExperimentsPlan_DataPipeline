# Configuration for DiT Base (Train from Scratch)
import os

# WandB
WANDB_PROJECT = "synthetic-experiments-dit-base"
WANDB_ENTITY = "your-entity"
WANDB_API_KEY = "your-api-key"

# Training
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 50
GRADIENT_ACCUMULATION_STEPS = 1
MIXED_PRECISION = "fp16" # 'fp16', 'bf16', 'no'

# Logging
LOG_INTERVAL = 1
METRICS_INTERVAL = 100
INFERENCE_INTERVAL = 100
SAVE_INTERVAL = 1000

# DiT Architecture (~250M Parameters approximation)
# DiT-B is ~130M (768 dim, 12 layers). 
# We'll scale up depth or width slightly.
# Let's use hidden_size=896, depth=18 or similar, or just DiT-L/2 scaled down.
# We will define this precisely in model.py
DIT_CONFIG = {
    "sample_size": 64,  # For 512x512 images with VAE f=8 -> 64x64 latents
    "patch_size": 2,
    "in_channels": 4,   # Latent channels
    "hidden_size": 768, # Base width
    "num_layers": 24,   # Increased depth to hit ~250M
    "num_attention_heads": 12,
    "dropout": 0.1,
    "learn_sigma": True
}

# Paths
DATASET_ROOT = "dataset_train_mixture" # S3 Prefix
OUTPUT_DIR = "checkpoints"

# AWS S3
AWS_ACCESS_KEY_ID = "YOUR_AWS_ACCESS_KEY"
AWS_SECRET_ACCESS_KEY = "YOUR_AWS_SECRET_KEY"
AWS_REGION = "us-east-1"
S3_BUCKET_NAME = "your-s3-bucket-name"
