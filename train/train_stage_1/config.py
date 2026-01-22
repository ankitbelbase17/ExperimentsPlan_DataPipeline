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
WANDB_PROJECT = "synthetic-experiments-stage-1"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 10
GRADIENT_ACCUMULATION_STEPS = 1
MIXED_PRECISION = "fp16" # 'no', 'fp16', 'bf16'

# Logging & Checkpointing
LOG_INTERVAL = 1 # steps
METRICS_INTERVAL = 100 # steps
INFERENCE_INTERVAL = 100 # steps
SAVE_INTERVAL = 1000 # steps

# Paths
DATASET_ROOT = "dataset_stage_1" # Relative or absolute path
OUTPUT_DIR = "checkpoints"

# Model
MODEL_NAME = "runwayml/stable-diffusion-v1-5" # HuggingFace model ID

# AWS S3 Configuration
# NOTE: Set these environment variables before running:
# Option 1: Create .env file in this directory with your credentials
# Option 2: Set environment variables in shell

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "")

