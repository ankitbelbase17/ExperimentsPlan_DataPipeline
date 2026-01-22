# CP-VTON: Characteristic-Preserving Virtual Try-On

## Overview
CP-VTON uses a two-stage architecture with geometric matching and try-on synthesis modules.

## Architecture

### Stage 1: Geometric Matching Module (GMM)
**Input Tensors:**
| Tensor | Shape | Description |
|--------|-------|-------------|
| `person_repr` | `[B, 3, H, W]` | Person representation (pose/segmentation) |
| `garment` | `[B, 3, H, W]` | Garment image |

**Output Tensors:**
| Tensor | Shape | Description |
|--------|-------|-------------|
| `tps_params` | `[B, 50]` | TPS transformation parameters (5x5 grid) |
| `warped_garment` | `[B, 3, H, W]` | Warped garment aligned to person |

### Stage 2: Try-On Module (TOM)
**Input Tensors:**
| Tensor | Shape | Description |
|--------|-------|-------------|
| `person` | `[B, 3, H, W]` | Person image |
| `warped_garment` | `[B, 3, H, W]` | Warped garment from GMM |
| `person_repr` | `[B, 3, H, W]` | Person representation |

**Output Tensors:**
| Tensor | Shape | Description |
|--------|-------|-------------|
| `tryon_result` | `[B, 3, H, W]` | Final try-on image |
| `composition_mask` | `[B, 1, H, W]` | Soft composition mask |

## Loss Functions
1. **L1 Loss**: Pixel-wise reconstruction
2. **VGG Perceptual Loss**: High-level feature matching
3. **Mask Regularization**: Encourages balanced composition

## Training
```bash
python train.py
```

## Dataset Structure
```
cp_vton_dataset/
├── person/          # Person images
├── garment/         # Garment images
├── person_repr/     # Person representations
├── target/          # Ground truth try-on results
├── train_pairs.txt
└── val_pairs.txt
```
