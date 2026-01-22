#!/bin/bash
# Training script for OOTDiffusion - Configurable Difficulty

DIFFICULTY=${1:-medium}

echo "Training OOTDiffusion - Difficulty: HARD"

cd ../../train_OOTDiffusion

python train.py \
  --difficulty hard \
  --batch_size 32 \
  --learning_rate 5e-5 \
  --output_dir checkpoints_ootdiffusion_${DIFFICULTY} \
  --wandb_project ootdiffusion-${DIFFICULTY} \
  --use_wandb

echo "OOTDiffusion training completed!"
