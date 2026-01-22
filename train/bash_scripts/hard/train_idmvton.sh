#!/bin/bash
# Training script for IDM-VTON - Configurable Difficulty

DIFFICULTY=${1:-medium}

echo "Training IDM-VTON - Difficulty: HARD"

cd ../../train_IDMVTON

python train.py \
  --difficulty hard \
  --batch_size 32 \
  --learning_rate 5e-5 \
  --output_dir checkpoints_idmvton_${DIFFICULTY} \
  --wandb_project idmvton-${DIFFICULTY} \
  --use_wandb

echo "IDM-VTON training completed!"
