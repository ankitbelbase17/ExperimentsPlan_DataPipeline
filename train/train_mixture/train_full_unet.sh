#!/bin/bash
# Train with FULL UNet (all parameters trainable)
# Run: bash train_full_unet.sh

echo "=============================================="
echo "Training DressCode with FULL UNet"
echo "=============================================="

python train_dresscode.py \
    --train_mode full_unet \
    --batch_size 32 \
    --num_workers 4 \
    --epochs 10 \
    --lr 1e-4 \
    --save_interval 100 \
    --image_log_interval 100 \
    --num_inference_steps 50

echo "Training complete!"
