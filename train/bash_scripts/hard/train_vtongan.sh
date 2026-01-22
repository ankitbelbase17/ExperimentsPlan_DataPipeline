#!/bin/bash
# Training script for VTON-GAN - Configurable Difficulty

DIFFICULTY=${1:-medium}

echo "Training VTON-GAN - Difficulty: HARD"

cd ../../train_VTON_GAN

python train.py \
  --difficulty hard \
  --batch_size 32 \
  --resume_from_checkpoint checkpoints_vtongan/latest_checkpoint.pt \
  --learning_rate 2e-4 \
  --output_dir checkpoints_vtongan_${DIFFICULTY} \
  --wandb_project vtongan-${DIFFICULTY} \
  --use_wandb

echo "VTON-GAN training completed!"
