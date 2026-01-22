#!/bin/bash
# Training script for VTON-GAN - Configurable Difficulty

DIFFICULTY=${1:-medium}

echo "Training VTON-GAN - Difficulty: $DIFFICULTY"

cd ../train_VTON_GAN

python train.py \
  --difficulty $DIFFICULTY \
  --batch_size 4 \
  --num_epochs 50 \
  --learning_rate 2e-4 \
  --output_dir checkpoints_vtongan_${DIFFICULTY} \
  --wandb_project vtongan-${DIFFICULTY} \
  --use_wandb

echo "VTON-GAN training completed!"
