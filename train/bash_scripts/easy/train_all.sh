#!/bin/bash
# Master training script - Easy Difficulty
# Trains all models sequentially on easy dataset

echo "=========================================="
echo "Master Training Script - Easy Difficulty"
echo "Training: CATVTON, DiT, VTON-GAN"
echo "Dataset: 100% easy samples"
echo "=========================================="

# Train CATVTON
echo ""
echo "Step 1/3: Training CATVTON..."
bash train_catvton.sh

# Train DiT
echo ""
echo "Step 2/3: Training DiT..."
bash train_dit.sh

# Train VTON-GAN
echo ""
echo "Step 3/3: Training VTON-GAN..."
bash train_vtongan.sh

echo ""
echo "=========================================="
echo "All Easy difficulty training completed!"
echo "=========================================="
