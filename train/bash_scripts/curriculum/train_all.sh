#!/bin/bash
# Master Curriculum Learning Script
# Trains all models with progressive difficulty: Easy → Medium → Hard

echo "=========================================="
echo "Master Curriculum Learning Training"
echo "Models: CATVTON, DiT, VTON-GAN"
echo "Strategy: Easy → Medium → Hard"
echo "=========================================="

# Train CATVTON with curriculum
echo ""
echo "=========================================="
echo "Training 1/3: CATVTON Curriculum Learning"
echo "=========================================="
bash train_catvton.sh

# Train DiT with curriculum
echo ""
echo "=========================================="
echo "Training 2/3: DiT Curriculum Learning"
echo "=========================================="
bash train_dit.sh

# Train VTON-GAN with curriculum
echo ""
echo "=========================================="
echo "Training 3/3: VTON-GAN Curriculum Learning"
echo "=========================================="
bash train_vtongan.sh

echo ""
echo "=========================================="
echo "All Curriculum Learning Training Completed!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  CATVTON: 40k steps (Easy→Medium→Hard)"
echo "  DiT: 400 epochs (Easy→Medium→Hard)"
echo "  VTON-GAN: 35k steps (Easy→Medium→Hard)"
echo ""
echo "Checkpoints saved to:"
echo "  - checkpoints_catvton_curriculum/"
echo "  - checkpoints_dit_curriculum/"
echo "  - checkpoints_vtongan_curriculum/"
echo "=========================================="
