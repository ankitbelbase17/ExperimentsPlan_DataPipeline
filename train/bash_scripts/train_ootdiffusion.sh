#!/bin/bash
# Training script for OOTDiffusion - Configurable Difficulty

DIFFICULTY=${1:-medium}

echo "Training OOTDiffusion - Difficulty: $DIFFICULTY"

cd ../train_OOTDiffusion

python train.py \
  --difficulty $DIFFICULTY \
  --batch_size 2 \
  --num_epochs 50 \
  --learning_rate 1e-5 \
  --output_dir checkpoints_ootdiffusion_${DIFFICULTY} \
  --wandb_project ootdiffusion-${DIFFICULTY} \
  --use_wandb

echo "OOTDiffusion training completed!"
