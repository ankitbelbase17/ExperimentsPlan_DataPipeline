#!/bin/bash
# Curriculum Learning Training Script for CATVTON
# Progressive difficulty: Easy → Medium → Hard

echo "=========================================="
echo "CATVTON - Curriculum Learning Training"
echo "Progressive Difficulty: Easy → Medium → Hard"
echo "=========================================="

cd ../../train_CATVTON

# Stage 1: Easy (Steps 0-10000)
echo ""
echo "Stage 1/3: Training on EASY samples (Steps 0-10000)"
echo "Dataset: 100% easy"
python train.py \
  --difficulty easy \
  --batch_size 4 \
  --learning_rate 1e-5 \
  --max_train_steps 10000 \
  --output_dir checkpoints_catvton_curriculum \
  --wandb_project catvton-curriculum \
  --wandb_run_name "catvton-curriculum-stage1-easy" \
  --resume_from_checkpoint checkpoints_catvton_curriculum/latest_checkpoint.pt \
  --use_wandb

echo ""
echo "Stage 1 completed! Switching to MEDIUM difficulty..."
sleep 2

# Stage 2: Medium (Steps 10000-25000)
echo ""
echo "Stage 2/3: Training on MEDIUM samples (Steps 10000-25000)"
echo "Dataset: 30% easy + 70% medium"
python train.py \
  --difficulty medium \
  --batch_size 4 \
  --learning_rate 5e-6 \
  --max_train_steps 25000 \
  --output_dir checkpoints_catvton_curriculum \
  --wandb_project catvton-curriculum \
  --wandb_run_name "catvton-curriculum-stage2-medium" \
  --resume_from_checkpoint checkpoints_catvton_curriculum/latest_checkpoint.pt \
  --use_wandb

echo ""
echo "Stage 2 completed! Switching to HARD difficulty..."
sleep 2

# Stage 3: Hard (Steps 25000-40000)
echo ""
echo "Stage 3/3: Training on HARD samples (Steps 25000-40000)"
echo "Dataset: 25% easy + 25% medium + 50% hard"
python train.py \
  --difficulty hard \
  --batch_size 4 \
  --learning_rate 2e-6 \
  --max_train_steps 40000 \
  --output_dir checkpoints_catvton_curriculum \
  --wandb_project catvton-curriculum \
  --wandb_run_name "catvton-curriculum-stage3-hard" \
  --resume_from_checkpoint checkpoints_catvton_curriculum/latest_checkpoint.pt \
  --use_wandb

echo ""
echo "=========================================="
echo "CATVTON Curriculum Learning Completed!"
echo "Total Steps: 40000"
echo "  - Easy: 0-10000 (10k steps)"
echo "  - Medium: 10000-25000 (15k steps)"
echo "  - Hard: 25000-40000 (15k steps)"
echo "=========================================="
