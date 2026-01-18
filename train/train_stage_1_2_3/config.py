# Configuration parameters
import os

# WandB Configuration
WANDB_PROJECT = "synthetic-experiments-stage-1-2-3"
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
AWS_ACCESS_KEY_ID = "YOUR_AWS_ACCESS_KEY"
AWS_SECRET_ACCESS_KEY = "YOUR_AWS_SECRET_KEY"
AWS_REGION = "us-east-1"
S3_BUCKET_NAME = "your-s3-bucket-name"
