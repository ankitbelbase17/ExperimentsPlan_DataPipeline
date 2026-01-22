# OOTDiffusion Configuration
# Paper: "Outfitting Fusion based Latent Diffusion for Controllable Virtual Try-on"
import os

# WandB Configuration
WANDB_PROJECT = "ootdiffusion-experiments"
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "your-entity")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "your-api-key")

# Training Hyperparameters
LEARNING_RATE = 5e-5
BATCH_SIZE = 32
NUM_EPOCHS = 50
GRADIENT_ACCUMULATION_STEPS = 4
MIXED_PRECISION = "fp16"

# Logging & Checkpointing
LOG_INTERVAL = 10
METRICS_INTERVAL = 250
INFERENCE_INTERVAL = 250
SAVE_INTERVAL = 250
CHECKPOINT_S3_PREFIX = "checkpoints/ootdiffusion"

# Model Configuration
MODEL_NAME = "stabilityai/stable-diffusion-2-base"  # SD2 base model (not inpainting)
IMAGE_SIZE = 768  # OOTDiffusion uses higher resolution (768x1024)
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 768
LATENT_CHANNELS = 4

# OOTDiffusion Specific Architecture
USE_OUTFITTING_UNET = True  # Specialized UNet for garment features
USE_FUSION_BLOCKS = True    # Cross-attention fusion blocks
NUM_FUSION_LAYERS = 4

# Garment Feature Extraction
GARMENT_ENCODER_TYPE = "vae"  # Options: "vae", "clip", "dinov2"
GARMENT_FEATURE_DIM = 768
USE_GARMENT_ATTENTION = True

# Conditioning
USE_POSE_GUIDANCE = True
USE_DENSEPOSE = True
USE_OPENPOSE = True
POSE_FEATURE_DIM = 256

# Training Strategy
TRAIN_UNET_ONLY = False  # If True, freeze VAE and text encoder
TRAIN_FUSION_ONLY = False  # If True, only train fusion blocks
USE_LORA = False  # Use LoRA for efficient fine-tuning
LORA_RANK = 64

# Loss Configuration
LOSS_RECONSTRUCTION_WEIGHT = 1.0
LOSS_PERCEPTUAL_WEIGHT = 0.1
LOSS_GARMENT_CONSISTENCY_WEIGHT = 0.5
LOSS_POSE_ALIGNMENT_WEIGHT = 0.3

# Data Augmentation
USE_HORIZONTAL_FLIP = False  # OOTDiffusion typically doesn't flip
USE_COLOR_JITTER = True
USE_RANDOM_CROP = False

# Dataset Configuration
DATASET_ROOT = "ootdiffusion_dataset"
TRAIN_PAIRS_FILE = "train_pairs.txt"
VAL_PAIRS_FILE = "val_pairs.txt"

# Inference Configuration
INFERENCE_STEPS = 50
GUIDANCE_SCALE = 2.0
SEED = 42

# Output
OUTPUT_DIR = "checkpoints_ootdiffusion"
INFERENCE_OUTPUT_DIR = "inference_outputs_ootdiffusion"

# AWS S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "YOUR_AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "YOUR_AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = "p1-to-ep1"
