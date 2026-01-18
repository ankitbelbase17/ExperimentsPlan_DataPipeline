#!/bin/bash
# Inference script for DiT Fast (1-Step)

CHECKPOINT="checkpoints/checkpoint-step-1000.ckpt"

echo "Running 1-step inference for DiT Fast..."
python inference.py --checkpoint "$CHECKPOINT" --prompt "a photo of a synthetic object"
