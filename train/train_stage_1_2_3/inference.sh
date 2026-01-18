#!/bin/bash
# Inference script for Stage 1

# Default checkpoint path (modify as needed)
CHECKPOINT="checkpoints/checkpoint-step-1000.ckpt"

echo "Running inference..."
python inference.py --checkpoint "$CHECKPOINT" --prompt "a photo of a synthetic object"
