#!/bin/bash
# Inference script for Sapiens VITON
# Ensure you have sample images: person.jpg, cloth.jpg, mask.jpg in current dir or adjust paths

CHECKPOINT="checkpoints/checkpoint-step-1000.ckpt"

# Creating dummy files for demonstration if they don't exist
# In real usage, point to real files.

echo "Running inference..."
python inference.py --checkpoint "$CHECKPOINT" --person "person.jpg" --cloth "cloth.jpg" --mask "mask.jpg"
