#!/bin/bash
# Training script for VTON-GAN - Medium Difficulty
# Uses 30% easy + 70% medium samples

echo "=========================================="
echo "Training VTON-GAN - Medium Difficulty"
echo "Dataset: 30% easy + 70% medium"
echo "=========================================="

cd ../../train_VTON_GAN

python train.py \
  --difficulty medium \
  --batch_size 4 \
  --num_epochs 50 \
  --learning_rate 2e-4 \
  --output_dir checkpoints_vtongan_medium \
  --wandb_project vtongan-medium \
  --s3_prefixes \
    "dataset_ultimate/easy/female/" \
    "dataset_ultimate/easy/male/" \
    "dataset_ultimate/medium/female/" \
    "dataset_ultimate/medium/male/" \
  --use_wandb

echo "VTON-GAN Medium training completed!"
