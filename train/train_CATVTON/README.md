# CATVTON: Concatenation-based Attentive Virtual Try-On Network

## Overview
CATVTON is a virtual try-on model that uses concatenation-based attention mechanisms to synthesize realistic images of people wearing different garments.

## Architecture

### Input Tensors (Training)
| Tensor Name | Shape | Description |
|-------------|-------|-------------|
| `person_img` | `[B, 3, H, W]` | Target person image |
| `garment_img` | `[B, 3, H, W]` | Garment to try on |
| `pose_map` | `[B, 3, H, W]` | Pose keypoints visualization |
| `segmentation` | `[B, 3, H, W]` | Body part segmentation mask |
| `input_ids` | `[B, 77]` | Text prompt tokens |

### Intermediate Tensors
| Tensor Name | Shape | Description |
|-------------|-------|-------------|
| `person_latents` | `[B, 4, H/8, W/8]` | VAE-encoded person |
| `garment_latents` | `[B, 4, H/8, W/8]` | VAE-encoded warped garment |
| `pose_latents` | `[B, 4, H/8, W/8]` | VAE-encoded pose |
| `seg_latents` | `[B, 4, H/8, W/8]` | VAE-encoded segmentation |
| `unet_input` | `[B, 16, H/8, W/8]` | Concatenated latents |
| `text_embeddings` | `[B, 77, 768]` | CLIP text embeddings |

### Output Tensors
| Tensor Name | Shape | Description |
|-------------|-------|-------------|
| `noise_pred` | `[B, 4, H/8, W/8]` | Predicted noise |
| `tps_params` | `[B, K*4]` | TPS transformation parameters |

## Key Features
1. **Warping Module**: Thin-Plate Spline (TPS) transformation to align garment with person pose
2. **Multi-Input Concatenation**: Combines person, garment, pose, and segmentation in latent space
3. **Modified UNet**: 16-channel input (4 modalities × 4 latent channels each)
4. **Attention Fusion**: Cross-attention with text conditioning

## Training Command
```bash
python train.py
```

## Dataset Structure
```
vton_dataset/
├── person/          # Person images
├── garment/         # Garment images
├── pose/            # Pose keypoint visualizations
├── segmentation/    # Body part masks
├── train_pairs.txt  # person_id garment_id pairs
└── val_pairs.txt    # Validation pairs
```

## Configuration
Edit `config.py` to adjust:
- Learning rate, batch size, epochs
- Enable/disable warping module
- S3 bucket settings
- WandB project name
