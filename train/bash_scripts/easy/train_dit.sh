#!/bin/bash
# Training script for DiT - Easy Difficulty
# Uses 100% easy samples

echo "=========================================="
echo "Training DiT - Easy Difficulty"
echo "Dataset: 100% easy samples"
echo "=========================================="

cd ../../train_DIT

python train.py \
  --difficulty easy \
  --batch_size 256 \
  --num_epochs 400 \
  --learning_rate 1e-4 \
  --output_dir checkpoints_dit_easy \
  --wandb_project dit-easy \
  --s3_prefixes \
    "dataset_ultimate/easy/female/" \
    "dataset_ultimate/easy/male/" \
  --use_wandb

echo "DiT Easy training completed!"
