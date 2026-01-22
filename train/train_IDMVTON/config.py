# IDM-VTON Configuration
# Paper: "Improving Diffusion Models for Authentic Virtual Try-on in the Wild"
import os

# WandB Configuration
WANDB_PROJECT = "idm-vton-experiments"
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "your-entity")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "your-api-key")

# Training Hyperparameters
LEARNING_RATE = 5e-5
BATCH_SIZE = 32
NUM_EPOCHS = 50
GRADIENT_ACCUMULATION_STEPS = 2
MIXED_PRECISION = "fp16"

# Logging & Checkpointing
LOG_INTERVAL = 10
METRICS_INTERVAL = 250
INFERENCE_INTERVAL = 250
SAVE_INTERVAL = 250
CHECKPOINT_S3_PREFIX = "checkpoints/idmvton"

# Model Configuration
MODEL_NAME = "stabilityai/stable-diffusion-2-base"  # SD2 base model (not inpainting)
IMAGE_SIZE = 512
LATENT_CHANNELS = 4

# IDM-VTON Specific
USE_GARMENT_ENCODER = True
USE_DENSEPOSE = True  # DensePose for better body understanding
GARMENT_FEATURE_DIM = 768
FUSION_STRATEGY = "gated_attention"  # Options: "concat", "add", "gated_attention"

# Two-stage training
STAGE_1_EPOCHS = 20  # Train garment encoder
STAGE_2_EPOCHS = 30  # Fine-tune full model

# Dataset Configuration
DATASET_ROOT = "idm_vton_dataset"
TRAIN_PAIRS_FILE = "train_pairs.txt"
VAL_PAIRS_FILE = "val_pairs.txt"

# Data Augmentation
USE_HORIZONTAL_FLIP = False  # IDM-VTON typically doesn't flip
USE_COLOR_JITTER = True
COLOR_JITTER_PARAMS = {
    "brightness": 0.1,
    "contrast": 0.1,
    "saturation": 0.1,
    "hue": 0.05
}

# Output
OUTPUT_DIR = "checkpoints_idmvton"
INFERENCE_OUTPUT_DIR = "inference_outputs_idmvton"

# AWS S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "YOUR_AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "YOUR_AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = "p1-to-ep1"
