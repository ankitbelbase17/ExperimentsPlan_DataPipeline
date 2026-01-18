# Train Stage 1

This directory contains the training pipeline for **Stage 1** of the synthetic dataset experiments.

## Overview
- **Dataset**: `dataset_stage_1` (sampled from S3).
- **Model**: Stable Diffusion v1.5 (Fine-tuning).
- **Objective**: Standard Diffusion Denoising Score Matching.

## Structure
- `train.py`: Main training loop with WandB logging and mixed precision.
- `dataloader.py`: S3-based dataloader.
- `config.py`: Configuration parameters (Hyperparams, AWS Credentials).
- `model.py`: SD1.5 wrapper.
- `inference.py`: Inference script.

## Usage
1. Update `config.py` with your AWS and WandB credentials.
2. Run training:
   ```bash
   ./train.sh
   # or
   python train.py
   ```
3. Run inference:
   ```bash
   ./inference.sh
   ```
