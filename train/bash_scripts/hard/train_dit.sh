#!/bin/bash
# Training script for DiT - Configurable Difficulty

DIFFICULTY=${1:-hard}

echo "Training DiT - Difficulty: HARD"

cd ../../train_DIT

python train.py \
  --difficulty hard \
  --batch_size 32 \
  --learning_rate 5e-5 \
  --num_epochs 400 \
  --output_dir checkpoints_dit_${DIFFICULTY} \
  --wandb_project dit-${DIFFICULTY} \
  --use_wandb

echo "DiT training completed!"
