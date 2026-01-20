# Configuration parameters
import os

# WandB Configuration
WANDB_PROJECT = "synthetic-experiments-mixture"
WANDB_ENTITY = "your-entity" 
WANDB_API_KEY = "your-api-key"

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 10
GRADIENT_ACCUMULATION_STEPS = 1
MIXED_PRECISION = "fp16" 

# Logging & Checkpointing
LOG_INTERVAL = 1 
METRICS_INTERVAL = 100
INFERENCE_INTERVAL = 100 
SAVE_INTERVAL = 1000 

# Paths
DATASET_ROOT = "train_mixture" # Assumed S3 prefix
OUTPUT_DIR = "checkpoints"

# Model
MODEL_NAME = "runwayml/stable-diffusion-v1-5" 
IMAGE_SIZE = 512

# AWS S3 Configuration
# NOTE: Set these environment variables before running:
# export AWS_ACCESS_KEY_ID="your-access-key"
# export AWS_SECRET_ACCESS_KEY="your-secret-key"
# Or configure in Colab: os.environ['AWS_ACCESS_KEY_ID'] = "your-key"

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.environ.get("AWS_REGION", "eu-north-1")  # Stockholm region
S3_BUCKET_NAME = "dipan-dresscode-s3-bucket"
DRESSCODE_ROOT = "dresscode/dresscode"  # S3 prefix path to dresscode dataset
