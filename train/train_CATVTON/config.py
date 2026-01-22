# CATVTON Configuration
# Paper: "Concatenation-based Attentive Try-On Network"
import os

# WandB Configuration
WANDB_PROJECT = "catvton-experiments"
WANDB_ENTITY = "your-entity"
WANDB_API_KEY = "your-api-key"

# Training Hyperparameters
LEARNING_RATE = 5e-5
BATCH_SIZE = 32
NUM_EPOCHS = 100
GRADIENT_ACCUMULATION_STEPS = 1
MIXED_PRECISION = "fp16"

# Logging & Checkpointing
LOG_INTERVAL = 10
METRICS_INTERVAL = 250
INFERENCE_INTERVAL = 250
SAVE_INTERVAL = 250
CHECKPOINT_S3_PREFIX = "checkpoints/catvton"

# Model Configuration
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
IMAGE_SIZE = 512
LATENT_CHANNELS = 4

# CATVTON Specific
USE_WARPING_MODULE = True
USE_SEGMENTATION = True
USE_POSE_ESTIMATION = True
GARMENT_ENCODER_LAYERS = 3
PERSON_ENCODER_LAYERS = 3

# Dataset Configuration
DATASET_ROOT = "vton_dataset"
TRAIN_PAIRS_FILE = "train_pairs.txt"
VAL_PAIRS_FILE = "val_pairs.txt"

# Data Augmentation
USE_HORIZONTAL_FLIP = True
USE_COLOR_JITTER = False
USE_RANDOM_CROP = False

# Output
OUTPUT_DIR = "checkpoints_catvton"
INFERENCE_OUTPUT_DIR = "inference_outputs_catvton"

# AWS S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "YOUR_AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "YOUR_AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = "p1-to-ep1"
