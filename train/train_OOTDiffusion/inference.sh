#!/bin/bash
# Inference script for OOTDiffusion

python inference.py \
  --checkpoint checkpoints_ootdiffusion/latest_checkpoint.pt \
  --person_image path/to/person.jpg \
  --garment_image path/to/garment.jpg \
  --pose_map path/to/pose.npy \
  --output_dir inference_outputs_ootdiffusion \
  --num_inference_steps 50 \
  --guidance_scale 2.0
