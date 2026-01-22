# DiT (Diffusion Transformer) Configuration
# Scalable Diffusion Models with Transformers
import os

# WandB Configuration
WANDB_PROJECT = "dit-experiments"
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "your-entity")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "your-api-key")

# Training Hyperparameters
LEARNING_RATE = 5e-5
BATCH_SIZE = 32  # DiT uses large batch sizes
NUM_EPOCHS = 400
GRADIENT_ACCUMULATION_STEPS = 1
MIXED_PRECISION = "bf16"  # DiT typically uses bfloat16

# Logging & Checkpointing
LOG_INTERVAL = 10
METRICS_INTERVAL = 250
INFERENCE_INTERVAL = 250
SAVE_INTERVAL = 250
CHECKPOINT_S3_PREFIX = "checkpoints/dit"

# Model Configuration - Custom (~215M parameters)
DIT_CONFIG = {
    "patch_size": 2,
    "hidden_size": 768,  # Reduced from 1152 (XL) -> 768 (Base+)
    "depth": 20,         # Increased from 12 (Base) -> 20 to hit ~200M
    "num_heads": 12,     # 768 / 64 = 12
    "mlp_ratio": 4.0,
    "class_dropout_prob": 0.1,
    "num_classes": 1000,
    "learn_sigma": True,
}

# Image Configuration
IMAGE_SIZE = 256  # DiT typically trains on 256x256 or 512x512
LATENT_SIZE = IMAGE_SIZE // 8  # VAE downsampling factor
IN_CHANNELS = 4  # VAE latent channels

# Training Objective
OBJECTIVE = "diffusion"  # Options: "diffusion", "rectified_flow", "flow_matching"

# Diffusion Configuration
NUM_DIFFUSION_STEPS = 1000
NOISE_SCHEDULE = "linear"  # Options: "linear", "cosine"
PREDICTION_TYPE = "epsilon"  # Options: "epsilon", "v_prediction", "sample"

# Dataset Configuration
DATASET_ROOT = "imagenet_dataset"
USE_IMAGENET = True

# Optimization
USE_EMA = True
EMA_DECAY = 0.9999
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0

# Output
OUTPUT_DIR = "checkpoints_dit"
INFERENCE_OUTPUT_DIR = "inference_outputs_dit"

# AWS S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "YOUR_AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "YOUR_AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = "p1-to-ep1"
