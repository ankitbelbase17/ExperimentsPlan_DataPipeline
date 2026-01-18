#!/bin/bash
# Train script for DiT Base
# Supported objectives: diffusion, rectified_flow

OBJECTIVE=${1:-diffusion}
echo "Starting training for DiT Base with objective: $OBJECTIVE"
python train.py --objective $OBJECTIVE
