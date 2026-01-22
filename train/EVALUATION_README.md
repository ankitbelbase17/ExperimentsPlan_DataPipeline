# VTON Models Evaluation Guide

## Overview
This directory contains comprehensive evaluation tools for all VTON models with automatic checkpoint discovery and metric computation.

## Metrics Computed

### Image Quality Metrics
- **LPIPS** (Learned Perceptual Image Patch Similarity): Perceptual similarity
- **SSIM** (Structural Similarity Index): Structural similarity
- **PSNR** (Peak Signal-to-Noise Ratio): Pixel-level accuracy

### Masked Metrics
- **Masked LPIPS**: LPIPS computed only in garment region
- **Masked SSIM**: SSIM computed only in garment region

### Semantic Metrics
- **mIOU** (mean Intersection over Union): Segmentation accuracy
- **PCK** (Percentage of Correct Keypoints): Pose alignment accuracy

## Quick Start

### 1. Evaluate All Models
```bash
python evaluate_all_models.py \
    --test_dir dataset_test \
    --output_dir evaluation_results \
    --device cuda
```

### 2. Evaluate Single Model
```bash
python -m common.metrics \
    --model catvton \
    --checkpoint s3://p1-to-ep1/checkpoints/catvton/checkpoint_latest.pt \
    --test_dir dataset_test \
    --output_dir metrics_results
```

### 3. Run Inference Only (CATVTON Example)
```bash
cd train_CATVTON
python inference.py \
    --checkpoint s3://p1-to-ep1/checkpoints/catvton/checkpoint_latest.pt \
    --test_dir ../dataset_test \
    --output_dir inference_outputs \
    --num_inference_steps 50 \
    --guidance_scale 7.5
```

## Test Dataset Structure

Your `dataset_test` directory should follow this structure:

```
dataset_test/
├── initial_image/          # Person images
│   ├── 001_person.png
│   ├── 002_person.png
│   └── ...
├── cloth_image/            # Garment images
│   ├── 001_cloth_shirt.png
│   ├── 002_cloth_dress.png
│   └── ...
├── try_on_image/           # Ground truth try-on results
│   ├── 001_vton.png
│   ├── 002_vton.png
│   └── ...
├── mask/                   # (Optional) Garment region masks
│   ├── 001_mask.png
│   └── ...
├── segmentation/           # (Optional) Body part segmentation
│   ├── 001_seg.png
│   └── ...
└── keypoints/              # (Optional) Pose keypoints
    ├── 001_keypoints.json
    └── ...
```

## Checkpoint Discovery

The evaluation scripts automatically find the latest checkpoint:

1. **S3 Checkpoints**: Searches `s3://p1-to-ep1/checkpoints/{model_name}/`
2. **Local Checkpoints**: Searches `checkpoints_{model_name}/`

You can manually specify a checkpoint with `--checkpoint` flag.

## Output Format

### Individual Model Results
Each model generates a JSON file with detailed metrics:

```json
{
  "lpips": {
    "mean": 0.1234,
    "std": 0.0456,
    "min": 0.0789,
    "max": 0.2345
  },
  "ssim": {
    "mean": 0.8765,
    "std": 0.0234,
    "min": 0.7890,
    "max": 0.9456
  },
  ...
}
```

### Combined Summary
`all_models_summary.json` contains results for all models in a single file.

## Advanced Usage

### Find Latest Checkpoint
```bash
python -m common.checkpoint_utils \
    --model catvton \
    --s3_bucket p1-to-ep1 \
    --s3_prefix checkpoints/catvton
```

### Custom Inference Parameters
```bash
python train_CATVTON/inference.py \
    --checkpoint path/to/checkpoint.pt \
    --test_dir dataset_test \
    --num_inference_steps 100 \
    --guidance_scale 10.0 \
    --batch_size 4
```

## Supported Models

- ✅ **CATVTON**: Fully implemented
- 🔄 **IDM-VTON**: Inference pending
- 🔄 **CP-VTON**: Inference pending
- 🔄 **VTON-GAN**: Inference pending
- 🔄 **OOTDiffusion**: Inference pending
- ✅ **DiT**: Works with triplets

## Requirements

```bash
pip install torch torchvision
pip install lpips scikit-image
pip install boto3 pillow tqdm
pip install diffusers transformers
```

## Troubleshooting

### No checkpoint found
- Verify S3 credentials are set: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- Check checkpoint directory exists
- Ensure checkpoint files end with `.pt` or `.pth`

### CUDA out of memory
- Reduce `--batch_size`
- Reduce `--num_inference_steps`
- Use smaller image size in config

### Missing test data
- Verify test directory structure matches expected format
- Ensure file naming follows pattern: `{stem}_{type}.{ext}`

## Citation

If you use this evaluation framework, please cite the original papers for each model.
