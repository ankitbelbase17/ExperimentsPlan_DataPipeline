#!/bin/bash
# Training script for DiT - Configurable Difficulty

DIFFICULTY=${1:-medium}

echo "Training DiT - Difficulty: $DIFFICULTY"

cd ../train_DIT

python train.py \
  --difficulty $DIFFICULTY \
  --batch_size 256 \
  --num_epochs 400 \
  --learning_rate 1e-4 \
  --output_dir checkpoints_dit_${DIFFICULTY} \
  --wandb_project dit-${DIFFICULTY} \
  --use_wandb

echo "DiT training completed!"
