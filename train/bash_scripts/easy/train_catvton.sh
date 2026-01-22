#!/bin/bash
# Training script for CATVTON - Easy Difficulty
# Uses 100% easy samples

echo "=========================================="
echo "Training CATVTON - Easy Difficulty"
echo "Dataset: 100% easy samples"
echo "=========================================="

cd ../../train_CATVTON

python train.py \
  --difficulty easy \
  --batch_size 4 \
  --num_epochs 50 \
  --learning_rate 1e-5 \
  --output_dir checkpoints_catvton_easy \
  --wandb_project catvton-easy \
  --s3_prefixes \
    "dataset_ultimate/easy/female/" \
    "dataset_ultimate/easy/male/" \
  --use_wandb

echo "CATVTON Easy training completed!"
