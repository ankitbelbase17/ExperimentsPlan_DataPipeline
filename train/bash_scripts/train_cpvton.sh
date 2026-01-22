#!/bin/bash
# Training script for CP-VTON - Configurable Difficulty

DIFFICULTY=${1:-medium}

echo "Training CP-VTON - Difficulty: $DIFFICULTY"

cd ../train_CP_VTON

python train.py \
  --difficulty $DIFFICULTY \
  --batch_size 4 \
  --num_epochs 50 \
  --learning_rate 1e-4 \
  --output_dir checkpoints_cpvton_${DIFFICULTY} \
  --wandb_project cpvton-${DIFFICULTY} \
  --use_wandb

echo "CP-VTON training completed!"
