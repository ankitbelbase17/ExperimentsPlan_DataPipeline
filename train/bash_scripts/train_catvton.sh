#!/bin/bash
# Training script for CATVTON - Configurable Difficulty

DIFFICULTY=${1:-medium}

echo "Training CATVTON - Difficulty: $DIFFICULTY"

cd ../train_CATVTON

python train.py \
  --difficulty $DIFFICULTY \
  --batch_size 4 \
  --num_epochs 50 \
  --learning_rate 1e-5 \
  --output_dir checkpoints_catvton_${DIFFICULTY} \
  --wandb_project catvton-${DIFFICULTY} \
  --use_wandb

echo "CATVTON training completed!"
