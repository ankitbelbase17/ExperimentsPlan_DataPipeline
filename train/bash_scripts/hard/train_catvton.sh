#!/bin/bash
# Training script for CATVTON - Configurable Difficulty

DIFFICULTY=${1:-medium}

echo "Training CATVTON - Difficulty: HARD"

cd ../../train_CATVTON

python train.py \
  --difficulty hard \
  --batch_size 32 \
  --learning_rate 5e-5 \
  --resume_from_checkpoint checkpoints_catvton/latest_checkpoint.pt \
  --use_wandbrning_rate 1e-5 \
  --output_dir checkpoints_catvton_${DIFFICULTY} \
  --wandb_project catvton-${DIFFICULTY} \
  --use_wandb

echo "CATVTON training completed!"
