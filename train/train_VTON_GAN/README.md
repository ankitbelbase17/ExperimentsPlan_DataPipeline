# VTON-GAN: GAN-based Virtual Try-On

## Overview
VTON-GAN uses adversarial training with a ResNet-based generator and PatchGAN discriminator.

## Architecture

### Generator
**Input Tensors:**
| Tensor | Shape | Description |
|--------|-------|-------------|
| `person` | `[B, 3, H, W]` | Person image |
| `garment` | `[B, 3, H, W]` | Garment image |
| `pose` | `[B, 3, H, W]` | Pose keypoints |

**Output Tensors:**
| Tensor | Shape | Description |
|--------|-------|-------------|
| `tryon_result` | `[B, 3, H, W]` | Generated try-on image |

### Discriminator
**Input Tensors:**
| Tensor | Shape | Description |
|--------|-------|-------------|
| `image` | `[B, 3, H, W]` | Real or fake try-on image |

**Output Tensors:**
| Tensor | Shape | Description |
|--------|-------|-------------|
| `prediction` | `[B, 1, H', W']` | Patch-wise real/fake predictions |

## Loss Functions
1. **Adversarial Loss**: LSGAN, vanilla GAN, or WGAN-GP
2. **L1 Reconstruction Loss**: Pixel-wise matching
3. **Perceptual Loss**: VGG-based feature matching (optional)
4. **Style Loss**: Gram matrix matching (optional)

## Training
```bash
python train.py
```

## Key Features
- Spectral normalization for training stability
- PatchGAN discriminator for high-frequency details
- Residual blocks in generator
- Separate learning rates for G and D

## Dataset Structure
```
vton_gan_dataset/
├── person/
├── garment/
├── pose/
├── target/
├── train_pairs.txt
└── val_pairs.txt
```
