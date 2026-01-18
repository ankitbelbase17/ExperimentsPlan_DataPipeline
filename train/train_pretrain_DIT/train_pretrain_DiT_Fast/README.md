# DiT Fast Pretraining (Mean Flow)

This project trains a **Diffusion Transformer (DiT)** using the **Mean Flow Velocity** objective for fast inference.

## Overview
- **Architecture**: DiT-Base (~250M params).
- **Objective**: **Mean Flow Velocity**.
  - Targets the global displacement vector $v = z_1 - z_0$.
  - Allows for **1-Step Inference**.
- **Dataset**: `dataset_train_mixture`.

## Inference
The inference script performs a single Euler step ($dt=1$):
$$ z_1 = z_0 + v_\theta(z_0, 0) \cdot 1 $$

## Usage
1. Update `config.py`.
2. Run training:
   ```bash
   ./train.sh
   ```
3. Run 1-step inference:
   ```bash
   ./inference.sh
   ```
