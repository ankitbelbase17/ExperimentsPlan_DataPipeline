#!/bin/bash
# Training script for CATVTON - Medium Difficulty
# Uses 30% easy + 70% medium samples

echo "=========================================="
echo "Training CATVTON - Medium Difficulty"
echo "Dataset: 30% easy + 70% medium"
echo "=========================================="

cd ../../train_CATVTON

python train.py \
  --difficulty medium \
  --batch_size 4 \
  --num_epochs 50 \
  --learning_rate 1e-5 \
  --output_dir checkpoints_catvton_medium \
  --wandb_project catvton-medium \
  --s3_prefixes \
    "dataset_ultimate/easy/female/" \
    "dataset_ultimate/easy/male/" \
    "dataset_ultimate/medium/female/" \
    "dataset_ultimate/medium/male/" \
  --use_wandb

echo "CATVTON Medium training completed!"
