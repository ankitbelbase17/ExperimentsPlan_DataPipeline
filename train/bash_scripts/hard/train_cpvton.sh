#!/bin/bash
# Training script for CP-VTON - Configurable Difficulty

DIFFICULTY=${1:-medium}

echo "Training CP-VTON - Difficulty: HARD"

cd ../../train_CP_VTON

python train.py \
  --difficulty hard \
  --batch_size 32 \
  --resume_from_checkpoint checkpoints_cpvton/latest_checkpoint.pt \
  --use_wandbrning_rate 1e-4 \
  --output_dir checkpoints_cpvton_${DIFFICULTY} \
  --wandb_project cpvton-${DIFFICULTY} \
  --use_wandb

echo "CP-VTON training completed!"
