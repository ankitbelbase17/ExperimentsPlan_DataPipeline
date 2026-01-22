#!/bin/bash
# Train with ATTENTION ONLY (only self-attention weights trainable)
# Run: bash train_attention_only.sh

echo "=============================================="
echo "Training DressCode with ATTENTION ONLY"
echo "=============================================="

python train_dresscode.py \
    --train_mode attention_only \
    --batch_size 32 \
    --num_workers 4 \
    --epochs 10 \
    --lr 1e-4 \
    --save_interval 100 \
    --image_log_interval 100 \
    --num_inference_steps 50

echo "Training complete!"
