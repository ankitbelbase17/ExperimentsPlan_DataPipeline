# Masked Sapiens Training (VITON)

This project implements a Virtual Try-On (VITON) style training pipeline using **Mask Agnostic** approach.

## Overview
- **Task**: Generate a person wearing a specific cloth, given:
  1. **Person Image** (Source)
  2. **Cloth Image** (Condition)
  3. **Agnostic Mask** (Defines region to repaint/keep)
- **Model**: Stable Diffusion v1.5 Inpainting.
- **Conditioning**: CLIP Vision Model (ViT-L/14) encodes the cloth image, passed via Cross-Attention.
- **Input Channels**: 9 (Noisy Latents + Mask + Masked Latents).

## Structure
- `train.py`: Custom training loop handling triplet inputs.
- `dataloader.py`: Triplet loading logic from S3.
- `model.py`: SapiensModel wrapper (UNet + CLIP Vision).

## Usage
1. Update `config.py`.
2. Run training:
   ```bash
   ./train.sh
   ```
3. Run inference:
   ```bash
   python inference.py --person p.jpg --cloth c.jpg --mask m.jpg
   ```
