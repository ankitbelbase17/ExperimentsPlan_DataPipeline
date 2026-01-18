# Train Stage 1 & 2 (Mixed)

This directory contains the training pipeline for **Stage 1 + Stage 2** mixed training.

## Overview
- **Dataset**: `dataset_stage_1` (30%) + `dataset_stage_2` (70%).
- **Sampling**: WeightedRandomSampler used to maintain the 30/70 ratio per batch.
- **Model**: Stable Diffusion v1.5 (Fine-tuning).

## Structure
- `train.py`: Main training loop.
- `dataloader.py`: Custom dataloader concatenating and weighting two S3 datasets.
- `config.py`: Configuration parameters.

## Usage
1. Update `config.py` with credentials.
2. Run training:
   ```bash
   ./train.sh
   ```
