#!/bin/bash
# Training script for IDM-VTON - Configurable Difficulty

DIFFICULTY=${1:-medium}

echo "Training IDM-VTON - Difficulty: $DIFFICULTY"

cd ../train_IDMVTON

python train.py \
  --difficulty $DIFFICULTY \
  --batch_size 4 \
  --num_epochs 50 \
  --learning_rate 1e-5 \
  --output_dir checkpoints_idmvton_${DIFFICULTY} \
  --wandb_project idmvton-${DIFFICULTY} \
  --use_wandb

echo "IDM-VTON training completed!"
