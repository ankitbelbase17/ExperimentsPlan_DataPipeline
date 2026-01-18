#!/bin/bash
# Inference script for DiT Base

OBJECTIVE=${1:-diffusion}
CHECKPOINT="checkpoints/checkpoint-step-1000.ckpt"

echo "Running inference for DiT Base with objective: $OBJECTIVE"
python inference.py --checkpoint "$CHECKPOINT" --objective $OBJECTIVE --prompt "a photo of a synthetic object"
