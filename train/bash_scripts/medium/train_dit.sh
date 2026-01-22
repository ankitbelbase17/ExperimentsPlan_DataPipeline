#!/bin/bash
# Training script for DiT - Medium Difficulty
# Uses 30% easy + 70% medium samples

echo "=========================================="
echo "Training DiT - Medium Difficulty"
echo "Dataset: 30% easy + 70% medium"
echo "=========================================="

cd ../../train_DIT

python train.py \
  --difficulty medium \
  --batch_size 256 \
  --num_epochs 400 \
  --learning_rate 1e-4 \
  --output_dir checkpoints_dit_medium \
  --wandb_project dit-medium \
  --s3_prefixes \
    "dataset_ultimate/easy/female/" \
    "dataset_ultimate/easy/male/" \
    "dataset_ultimate/medium/female/" \
    "dataset_ultimate/medium/male/" \
  --use_wandb

echo "DiT Medium training completed!"
