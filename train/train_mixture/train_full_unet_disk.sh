#!/bin/bash
# Train with FULL UNet (all parameters trainable) using LOCAL DISK dataset
# Run: bash train_full_unet_disk.sh

echo "=============================================="
echo "Training DressCode with FULL UNet (DISK)"
echo "=============================================="

python train_dresscode.py \
    --train_mode full_unet \
    --data_source disk \
    --local_data_path ./data/dresscode \
    --batch_size 32 \
    --num_workers 4 \
    --epochs 10 \
    --lr 1e-4 \
    --save_interval 100 \
    --image_log_interval 100 \
    --num_inference_steps 50

echo "Training complete!"
