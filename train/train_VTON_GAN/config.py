# VTON-GAN Configuration
# GAN-based Virtual Try-On with adversarial training
import os

# WandB Configuration
WANDB_PROJECT = "vton-gan-experiments"
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "your-entity")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "your-api-key")

# Training Hyperparameters
LEARNING_RATE_G = 2e-4  # Generator
LEARNING_RATE_D = 2e-4  # Discriminator
BATCH_SIZE = 32
NUM_EPOCHS = 100
GRADIENT_ACCUMULATION_STEPS = 1
MIXED_PRECISION = "fp16"

# Logging & Checkpointing
LOG_INTERVAL = 10
METRICS_INTERVAL = 250
INFERENCE_INTERVAL = 250
SAVE_INTERVAL = 250
CHECKPOINT_S3_PREFIX = "checkpoints/vtongan"

# Model Configuration
IMAGE_SIZE = 256
LATENT_DIM = 512
GENERATOR_CHANNELS = 64
DISCRIMINATOR_CHANNELS = 64

# GAN Specific
USE_SPECTRAL_NORM = True
USE_SELF_ATTENTION = True
GAN_MODE = "lsgan"  # Options: "vanilla", "lsgan", "wgangp"

# Loss Weights
LOSS_ADV_WEIGHT = 1.0
LOSS_L1_WEIGHT = 100.0
LOSS_PERCEPTUAL_WEIGHT = 10.0
LOSS_STYLE_WEIGHT = 250.0

# Dataset Configuration
DATASET_ROOT = "vton_gan_dataset"
TRAIN_PAIRS_FILE = "train_pairs.txt"
VAL_PAIRS_FILE = "val_pairs.txt"

# Output
OUTPUT_DIR = "checkpoints_vtongan"
INFERENCE_OUTPUT_DIR = "inference_outputs_vtongan"

# AWS S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "YOUR_AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "YOUR_AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = "p1-to-ep1"
