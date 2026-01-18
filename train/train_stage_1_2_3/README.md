# Train Stage 1, 2 & 3 (Mixed)

This directory contains the training pipeline for **Stage 1 + 2 + 3** mixed training.

## Overview
- **Dataset**: `dataset_stage_1` (25%) + `dataset_stage_2` (25%) + `dataset_stage_3` (50%).
- **Sampling**: WeightedRandomSampler used to maintain the 25/25/50 ratio.
- **Model**: Stable Diffusion v1.5 (Fine-tuning).

## Structure
- `train.py`: Main training loop.
- `dataloader.py`: Custom dataloader concatenating and weighting three S3 datasets.
- `config.py`: Configuration parameters.

## Usage
1. Update `config.py` with credentials.
2. Run training:
   ```bash
   ./train.sh
   ```
