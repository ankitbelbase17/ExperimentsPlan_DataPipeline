# Configuration parameters
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    print("python-dotenv not installed. Using environment variables from system.")

# WandB Configuration
WANDB_PROJECT = "synthetic-experiments-mixture"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")

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
# Option 1: Create .env file in this directory with your credentials
# Option 2: Set environment variables in shell:
#   export AWS_ACCESS_KEY_ID="your-access-key"
#   export AWS_SECRET_ACCESS_KEY="your-secret-key"
#   export AWS_REGION="eu-north-1"
#   export S3_BUCKET_NAME="your-bucket-name"

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.environ.get("AWS_REGION", "eu-north-1")  # Stockholm region
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "")
