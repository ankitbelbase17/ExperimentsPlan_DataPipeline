# Configuration parameters
import os

# WandB Configuration
WANDB_PROJECT = "synthetic-experiments-stage-1"
WANDB_ENTITY = "your-entity" # Update this
WANDB_API_KEY = "your-api-key" # Update this or set via env var

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
AWS_ACCESS_KEY_ID = "YOUR_AWS_ACCESS_KEY"
AWS_SECRET_ACCESS_KEY = "YOUR_AWS_SECRET_KEY"
AWS_REGION = "us-east-1"
S3_BUCKET_NAME = "p1-to-ep1"

