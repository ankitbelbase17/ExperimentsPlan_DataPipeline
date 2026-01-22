#!/bin/bash
# Training script for VTON-GAN - Easy Difficulty
# Uses 100% easy samples

echo "=========================================="
echo "Training VTON-GAN - Easy Difficulty"
echo "Dataset: 100% easy samples"
echo "=========================================="

cd ../../train_VTON_GAN

python train.py \
  --difficulty easy \
  --batch_size 4 \
  --num_epochs 50 \
  --learning_rate 2e-4 \
  --output_dir checkpoints_vtongan_easy \
  --wandb_project vtongan-easy \
  --s3_prefixes \
    "dataset_ultimate/easy/female/" \
    "dataset_ultimate/easy/male/" \
  --use_wandb

echo "VTON-GAN Easy training completed!"
