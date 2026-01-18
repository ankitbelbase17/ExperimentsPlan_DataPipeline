# DiT Base Pretraining

This project trains a **Diffusion Transformer (DiT)** (~250M parameters) from scratch.

## Overview
- **Architecture**: DiT (Transformer2DModel from diffusers) with Xavier Initialization.
- **Dataset**: `dataset_train_mixture`.
- **Objectives** (Selectable):
  1. `diffusion`: Standard DDPM/IDDPM noise prediction.
  2. `rectified_flow`: Flow matching objective ($v = z_1 - z_0$).

## Usage
1. Update `config.py`.
2. Run training (specify objective):
   ```bash
   ./train.sh diffusion
   # or
   ./train.sh rectified_flow
   ```
