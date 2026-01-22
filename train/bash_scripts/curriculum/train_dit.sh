#!/bin/bash
# Curriculum Learning Training Script for DiT
# Progressive difficulty: Easy → Medium → Hard

echo "=========================================="
echo "DiT - Curriculum Learning Training"
echo "Progressive Difficulty: Easy → Medium → Hard"
echo "=========================================="

cd ../../train_DIT

# Stage 1: Easy (Epochs 0-100)
echo ""
echo "Stage 1/3: Training on EASY samples (Epochs 0-100)"
echo "Dataset: 100% easy"
python train.py \
  --difficulty easy \
  --batch_size 256 \
  --learning_rate 1e-4 \
  --num_epochs 100 \
  --output_dir checkpoints_dit_curriculum \
  --wandb_project dit-curriculum \
  --wandb_run_name "dit-curriculum-stage1-easy" \
  --resume_from_checkpoint checkpoints_dit_curriculum/latest_checkpoint.pt \
  --use_wandb

echo ""
echo "Stage 1 completed! Switching to MEDIUM difficulty..."
sleep 2

# Stage 2: Medium (Epochs 100-250)
echo ""
echo "Stage 2/3: Training on MEDIUM samples (Epochs 100-250)"
echo "Dataset: 30% easy + 70% medium"
python train.py \
  --difficulty medium \
  --batch_size 256 \
  --learning_rate 5e-5 \
  --num_epochs 250 \
  --output_dir checkpoints_dit_curriculum \
  --wandb_project dit-curriculum \
  --wandb_run_name "dit-curriculum-stage2-medium" \
  --resume_from_checkpoint checkpoints_dit_curriculum/latest_checkpoint.pt \
  --use_wandb

echo ""
echo "Stage 2 completed! Switching to HARD difficulty..."
sleep 2

# Stage 3: Hard (Epochs 250-400)
echo ""
echo "Stage 3/3: Training on HARD samples (Epochs 250-400)"
echo "Dataset: 25% easy + 25% medium + 50% hard"
python train.py \
  --difficulty hard \
  --batch_size 256 \
  --learning_rate 2e-5 \
  --num_epochs 400 \
  --output_dir checkpoints_dit_curriculum \
  --wandb_project dit-curriculum \
  --wandb_run_name "dit-curriculum-stage3-hard" \
  --resume_from_checkpoint checkpoints_dit_curriculum/latest_checkpoint.pt \
  --use_wandb

echo ""
echo "=========================================="
echo "DiT Curriculum Learning Completed!"
echo "Total Epochs: 400"
echo "  - Easy: 0-100 (100 epochs)"
echo "  - Medium: 100-250 (150 epochs)"
echo "  - Hard: 250-400 (150 epochs)"
echo "=========================================="
