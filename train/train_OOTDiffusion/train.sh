#!/bin/bash
# Training script for OOTDiffusion

python train.py \
  --config config.py \
  --output_dir checkpoints_ootdiffusion \
  --logging_dir logs_ootdiffusion
