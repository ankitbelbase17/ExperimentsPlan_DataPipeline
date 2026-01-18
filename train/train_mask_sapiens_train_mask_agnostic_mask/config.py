# Configuration for Masked Sapiens Training (VITON)
import os

# WandB
WANDB_PROJECT = "synthetic-experiments-mask-sapiens"
WANDB_ENTITY = "your-entity"
WANDB_API_KEY = "your-api-key"

# Training
LEARNING_RATE = 5e-5
BATCH_SIZE = 4
NUM_EPOCHS = 20
GRADIENT_ACCUMULATION_STEPS = 1
MIXED_PRECISION = "fp16" 

# Logging
LOG_INTERVAL = 1
METRICS_INTERVAL = 100
INFERENCE_INTERVAL = 100
SAVE_INTERVAL = 1000

# Model
# We use the Inpainting variant of SD1.5 as the base because it facilitates 9-channel input
# (Noisy Latents + Mask + Masked Image Latents)
MODEL_NAME = "runwayml/stable-diffusion-inpainting" 
IMAGE_SIZE = 512

# Conditioning
# We use CLIP Vision Model to encode the Cloth Image and pass it as "encoder_hidden_states"
# to the UNet (replacing text).
USE_CLOTH_CONDITIONING = True

# Paths
DATASET_ROOT = "dataset_train_mixture" 
OUTPUT_DIR = "checkpoints"

# AWS S3
AWS_ACCESS_KEY_ID = "YOUR_AWS_ACCESS_KEY"
AWS_SECRET_ACCESS_KEY = "YOUR_AWS_SECRET_KEY"
AWS_REGION = "us-east-1"
S3_BUCKET_NAME = "your-s3-bucket-name"
