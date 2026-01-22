#!/bin/bash
# Curriculum Learning Training Script for VTON-GAN
# Progressive difficulty: Easy → Medium → Hard

echo "=========================================="
echo "VTON-GAN - Curriculum Learning Training"
echo "Progressive Difficulty: Easy → Medium → Hard"
echo "=========================================="

cd ../../train_VTON_GAN

# Stage 1: Easy (Steps 0-8000)
echo ""
echo "Stage 1/3: Training on EASY samples (Steps 0-8000)"
echo "Dataset: 100% easy"
python train.py \
  --difficulty easy \
  --batch_size 4 \
  --learning_rate 2e-4 \
  --max_train_steps 8000 \
  --output_dir checkpoints_vtongan_curriculum \
  --wandb_project vtongan-curriculum \
  --wandb_run_name "vtongan-curriculum-stage1-easy" \
  --resume_from_checkpoint checkpoints_vtongan_curriculum/latest_checkpoint.pt \
  --use_wandb

echo ""
echo "Stage 1 completed! Switching to MEDIUM difficulty..."
sleep 2

# Stage 2: Medium (Steps 8000-20000)
echo ""
echo "Stage 2/3: Training on MEDIUM samples (Steps 8000-20000)"
echo "Dataset: 30% easy + 70% medium"
python train.py \
  --difficulty medium \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --max_train_steps 20000 \
  --output_dir checkpoints_vtongan_curriculum \
  --wandb_project vtongan-curriculum \
  --wandb_run_name "vtongan-curriculum-stage2-medium" \
  --resume_from_checkpoint checkpoints_vtongan_curriculum/latest_checkpoint.pt \
  --use_wandb

echo ""
echo "Stage 2 completed! Switching to HARD difficulty..."
sleep 2

# Stage 3: Hard (Steps 20000-35000)
echo ""
echo "Stage 3/3: Training on HARD samples (Steps 20000-35000)"
echo "Dataset: 25% easy + 25% medium + 50% hard"
python train.py \
  --difficulty hard \
  --batch_size 4 \
  --learning_rate 5e-5 \
  --max_train_steps 35000 \
  --output_dir checkpoints_vtongan_curriculum \
  --wandb_project vtongan-curriculum \
  --wandb_run_name "vtongan-curriculum-stage3-hard" \
  --resume_from_checkpoint checkpoints_vtongan_curriculum/latest_checkpoint.pt \
  --use_wandb

echo ""
echo "=========================================="
echo "VTON-GAN Curriculum Learning Completed!"
echo "Total Steps: 35000"
echo "  - Easy: 0-8000 (8k steps)"
echo "  - Medium: 8000-20000 (12k steps)"
echo "  - Hard: 20000-35000 (15k steps)"
echo "=========================================="
