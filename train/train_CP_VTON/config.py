# CP-VTON Configuration
# Paper: "Toward Characteristic-Preserving Image-based Virtual Try-On Network"
import os

# WandB Configuration
WANDB_PROJECT = "cp-vton-experiments"
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "your-entity")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "your-api-key")

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 200
GRADIENT_ACCUMULATION_STEPS = 1
MIXED_PRECISION = "fp16"

# Logging & Checkpointing
LOG_INTERVAL = 10
METRICS_INTERVAL = 250
INFERENCE_INTERVAL = 250
SAVE_INTERVAL = 250
CHECKPOINT_S3_PREFIX = "checkpoints/cpvton"

# Model Configuration
IMAGE_SIZE = 256  # CP-VTON typically uses 256x256
LATENT_CHANNELS = 64

# CP-VTON Specific - Two-Stage Architecture
# Stage 1: Geometric Matching Module (GMM)
GMM_FEATURE_CHANNELS = 256
GMM_NUM_LAYERS = 6
USE_TPS_TRANSFORMATION = True

# Stage 2: Try-On Module (TOM)
TOM_FEATURE_CHANNELS = 96
TOM_NUM_LAYERS = 6
USE_COMPOSITION_MASK = True

# Loss Weights
LOSS_L1_WEIGHT = 1.0
LOSS_VGG_WEIGHT = 1.0
LOSS_MASK_WEIGHT = 1.0

# Dataset Configuration
DATASET_ROOT = "cp_vton_dataset"
TRAIN_PAIRS_FILE = "train_pairs.txt"
VAL_PAIRS_FILE = "val_pairs.txt"

# Output
OUTPUT_DIR = "checkpoints_cpvton"
INFERENCE_OUTPUT_DIR = "inference_outputs_cpvton"

# AWS S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "YOUR_AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "YOUR_AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = "p1-to-ep1"
