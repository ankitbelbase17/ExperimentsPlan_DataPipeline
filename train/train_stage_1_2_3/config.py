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
WANDB_PROJECT = "synthetic-experiments-stage-1-2-3"
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

# Datasets
DATASET_STAGE_1_ROOT = "dataset_stage_1"
DATASET_STAGE_2_ROOT = "dataset_stage_2"
DATASET_STAGE_3_ROOT = "dataset_stage_3"
DATASET_WEIGHTS = [0.25, 0.25, 0.50] # 25% Stage 1, 25% Stage 2, 50% Stage 3

OUTPUT_DIR = "checkpoints"

# Model
MODEL_NAME = "runwayml/stable-diffusion-v1-5" 
IMAGE_SIZE = 512

# AWS S3 Configuration
# NOTE: Set these environment variables before running:
# Option 1: Create .env file in this directory with your credentials
# Option 2: Set environment variables in shell

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "")
