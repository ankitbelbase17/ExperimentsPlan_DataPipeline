# Configuration for DiT Fast (Mean Flow / 1-Step)
import os

# WandB
WANDB_PROJECT = "synthetic-experiments-dit-fast"
WANDB_ENTITY = "your-entity"
WANDB_API_KEY = "your-api-key"

# Training
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 50
GRADIENT_ACCUMULATION_STEPS = 1
MIXED_PRECISION = "fp16" 

# Logging
LOG_INTERVAL = 1
METRICS_INTERVAL = 100
INFERENCE_INTERVAL = 100
SAVE_INTERVAL = 1000

# DiT Architecture (~250M Parameters)
DIT_CONFIG = {
    "sample_size": 64,  
    "patch_size": 2,
    "in_channels": 4,   
    "hidden_size": 768, 
    "num_layers": 24,   
    "num_attention_heads": 12,
    "dropout": 0.1,
    "learn_sigma": True 
}

# Paths
DATASET_ROOT = "dataset_train_mixture" # Using the mixture dataset
OUTPUT_DIR = "checkpoints"

# AWS S3
AWS_ACCESS_KEY_ID = "YOUR_AWS_ACCESS_KEY"
AWS_SECRET_ACCESS_KEY = "YOUR_AWS_SECRET_KEY"
AWS_REGION = "us-east-1"
S3_BUCKET_NAME = "your-s3-bucket-name"
