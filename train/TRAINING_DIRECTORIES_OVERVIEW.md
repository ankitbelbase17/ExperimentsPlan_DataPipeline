# Training Directories Overview

This document provides a comprehensive overview of all training directories in the SyntheticDatasetExperiments project.

## Directory Structure

```
train/
├── train_CATVTON/              # Concatenation-based Attentive Virtual Try-On
├── train_IDMVTON/              # Image-based Diffusion Model VTON
├── train_CP_VTON/              # Characteristic-Preserving VTON
├── train_VTON_GAN/             # GAN-based Virtual Try-On
├── train_DIT/                  # Diffusion Transformer
├── train_stage_1/              # Stable Diffusion Stage 1
├── train_stage_1_2/            # Stable Diffusion Stages 1-2
├── train_stage_1_2_3/          # Stable Diffusion Stages 1-2-3
├── train_mixture/              # Mixed Dataset Training
├── train_pretrain_DIT/         # DiT Pretraining (Base & Fast variants)
├── train_mask_sapiens_train_mask_agnostic_mask/  # Mask-agnostic training
└── train_contrastive_diffusion/  # Contrastive learning for diffusion
```

---

## 🎨 Virtual Try-On Methods

### 1. CATVTON (Concatenation-based Attentive VTON)

**Directory:** `train_CATVTON/`

**Architecture:**
- Warping Module: TPS-based garment alignment
- Modified UNet: 16-channel input (person + garment + pose + segmentation)
- Attention-based feature fusion

**Input Tensors:**
| Tensor | Shape | Description |
|--------|-------|-------------|
| `person_img` | `[B, 3, 512, 512]` | Target person |
| `garment_img` | `[B, 3, 512, 512]` | Garment to try on |
| `pose_map` | `[B, 3, 512, 512]` | Pose keypoints |
| `segmentation` | `[B, 3, 512, 512]` | Body segmentation |

**Output Tensors:**
| Tensor | Shape | Description |
|--------|-------|-------------|
| `noise_pred` | `[B, 4, 64, 64]` | Predicted noise in latent space |
| `tps_params` | `[B, 50]` | TPS transformation parameters |

**Key Features:**
- Multi-modal concatenation in latent space
- Thin-Plate Spline warping
- Diffusion-based synthesis

---

### 2. IDM-VTON (Improving Diffusion Models for VTON)

**Directory:** `train_IDMVTON/`

**Architecture:**
- CLIP-based Garment Encoder
- SD2-Inpainting UNet (9-channel input)
- Gated Attention Fusion
- DensePose conditioning

**Input Tensors:**
| Tensor | Shape | Description |
|--------|-------|-------------|
| `person_img` | `[B, 3, 512, 512]` | Target person |
| `garment_img` | `[B, 3, 512, 512]` | Garment image |
| `mask` | `[B, 1, 512, 512]` | Inpainting mask (1=keep, 0=generate) |
| `densepose` | `[B, 3, 512, 512]` | DensePose visualization |

**Output Tensors:**
| Tensor | Shape | Description |
|--------|-------|-------------|
| `noise_pred` | `[B, 4, 64, 64]` | Predicted noise |
| `garment_features` | `[B, 768]` | CLIP garment embeddings |

**Key Features:**
- Inpainting-based approach
- CLIP garment understanding
- Body-aware generation via DensePose
- Two-stage training (garment encoder → full model)

---

### 3. CP-VTON (Characteristic-Preserving VTON)

**Directory:** `train_CP_VTON/`

**Architecture:**
- Two-stage pipeline:
  1. **GMM (Geometric Matching Module):** Warps garment to person shape
  2. **TOM (Try-On Module):** Synthesizes final result with composition mask

**Stage 1 - GMM:**
| Input | Shape | Output | Shape |
|-------|-------|--------|-------|
| `person_repr` | `[B, 3, 256, 256]` | `tps_params` | `[B, 50]` |
| `garment` | `[B, 3, 256, 256]` | `warped_garment` | `[B, 3, 256, 256]` |

**Stage 2 - TOM:**
| Input | Shape | Output | Shape |
|-------|-------|--------|-------|
| `person` | `[B, 3, 256, 256]` | `tryon_result` | `[B, 3, 256, 256]` |
| `warped_garment` | `[B, 3, 256, 256]` | `composition_mask` | `[B, 1, 256, 256]` |
| `person_repr` | `[B, 3, 256, 256]` | | |

**Loss Functions:**
- L1 reconstruction loss
- VGG perceptual loss
- Composition mask regularization

**Key Features:**
- Explicit geometric matching
- Soft composition masks
- Characteristic preservation

---

### 4. VTON-GAN (GAN-based Virtual Try-On)

**Directory:** `train_VTON_GAN/`

**Architecture:**
- **Generator:** ResNet-based with 6 residual blocks
- **Discriminator:** PatchGAN with spectral normalization

**Generator:**
| Input | Shape | Output | Shape |
|-------|-------|--------|-------|
| `person` | `[B, 3, 256, 256]` | `tryon_result` | `[B, 3, 256, 256]` |
| `garment` | `[B, 3, 256, 256]` | | |
| `pose` | `[B, 3, 256, 256]` | | |

**Discriminator:**
| Input | Shape | Output | Shape |
|-------|-------|--------|-------|
| `image` | `[B, 3, 256, 256]` | `prediction` | `[B, 1, H', W']` |

**Loss Functions:**
- Adversarial loss (LSGAN/vanilla/WGAN-GP)
- L1 reconstruction loss
- Perceptual loss (optional)
- Style loss (optional)

**Key Features:**
- Adversarial training for photorealism
- Spectral normalization for stability
- PatchGAN for high-frequency details

---

## 🤖 Diffusion Transformer (DiT)

### DiT Training

**Directory:** `train_DIT/`

**Architecture:**
- Vision Transformer backbone
- Adaptive Layer Norm (adaLN-Zero)
- Patchification of latent space
- Classifier-free guidance support

**Model Variants:**
| Model | Depth | Hidden Size | Heads | Parameters |
|-------|-------|-------------|-------|------------|
| DiT-S | 12 | 384 | 6 | 33M |
| DiT-B | 12 | 768 | 12 | 130M |
| DiT-L | 24 | 1024 | 16 | 458M |
| DiT-XL | 28 | 1152 | 16 | 675M |

**Input Tensors (Diffusion Objective):**
| Tensor | Shape | Description |
|--------|-------|-------------|
| `images` | `[B, 3, 256, 256]` | RGB images |
| `labels` | `[B]` | Class labels (0-999 for ImageNet) |
| `timesteps` | `[B]` | Diffusion timesteps [0, 1000) |

**Intermediate Tensors:**
| Tensor | Shape | Description |
|--------|-------|-------------|
| `latents` | `[B, 4, 32, 32]` | VAE-encoded (256/8 = 32) |
| `noisy_latents` | `[B, 4, 32, 32]` | Latents + noise |
| `patches` | `[B, 256, 1152]` | Patchified (32/2)² = 256 patches |
| `timestep_emb` | `[B, 1152]` | Sinusoidal timestep embeddings |
| `label_emb` | `[B, 1152]` | Learned label embeddings |
| `conditioning` | `[B, 1152]` | timestep_emb + label_emb |

**Output Tensors:**
| Tensor | Shape | Description |
|--------|-------|-------------|
| `noise_pred` | `[B, 4, 32, 32]` | Predicted noise |
| `sigma_pred` | `[B, 4, 32, 32]` | Predicted variance (if learn_sigma=True) |

**Training Objectives:**
1. **Diffusion:** `loss = MSE(noise_pred, noise)`
2. **Rectified Flow:** `loss = MSE(v_pred, z1 - z0)`
3. **Flow Matching:** `loss = MSE(v_pred, z1 - z0)`

**Key Features:**
- Transformer-based diffusion
- Classifier-free guidance
- EMA (Exponential Moving Average)
- Large batch training (256+)
- BFloat16 mixed precision
- Gradient clipping

---

## 📚 Stable Diffusion Training

### Stage 1, 1-2, 1-2-3, Mixture

**Directories:** `train_stage_1/`, `train_stage_1_2/`, `train_stage_1_2_3/`, `train_mixture/`

**Architecture:**
- Stable Diffusion v1.5 UNet
- VAE encoder/decoder
- CLIP text encoder

**Input Tensors:**
| Tensor | Shape | Description |
|--------|-------|-------------|
| `pixel_values` | `[B, 3, 512, 512]` | RGB images |
| `input_ids` | `[B, 77]` | CLIP tokenized text |

**Intermediate Tensors:**
| Tensor | Shape | Description |
|--------|-------|-------------|
| `latents` | `[B, 4, 64, 64]` | VAE-encoded (scaled by 0.18215) |
| `noise` | `[B, 4, 64, 64]` | Gaussian noise |
| `timesteps` | `[B]` | Random timesteps [0, 1000) |
| `noisy_latents` | `[B, 4, 64, 64]` | Latents + noise |
| `encoder_hidden_states` | `[B, 77, 768]` | CLIP text embeddings |

**Output Tensors:**
| Tensor | Shape | Description |
|--------|-------|-------------|
| `noise_pred` | `[B, 4, 64, 64]` | UNet predicted noise |

**Training Objective:**
```python
loss = MSE(noise_pred, noise)
```

---

## 🎯 Comparison Table: Input/Output Tensors

| Method | Input Modalities | Latent Size | Output | Special Features |
|--------|------------------|-------------|--------|------------------|
| **CATVTON** | Person, Garment, Pose, Seg | `[B, 16, 64, 64]` | Noise | Multi-modal concat, TPS warp |
| **IDM-VTON** | Person, Garment, Mask, DensePose | `[B, 9, 64, 64]` | Noise | Inpainting, CLIP garment |
| **CP-VTON** | Person, Garment, Person Repr | `[B, 9, 256, 256]` | Image + Mask | Two-stage, pixel-space |
| **VTON-GAN** | Person, Garment, Pose | `[B, 9, 256, 256]` | Image | Adversarial, PatchGAN |
| **DiT** | Image, Label | `[B, 256, 1152]` | Noise/Velocity | Transformer, CFG |
| **SD Stages** | Image, Text | `[B, 4, 64, 64]` | Noise | Text-to-image |

---

## 🚀 Quick Start Guide

### CATVTON
```bash
cd train_CATVTON
python train.py
```

### IDM-VTON
```bash
cd train_IDMVTON
python train.py
```

### CP-VTON
```bash
cd train_CP_VTON
python train.py
```

### VTON-GAN
```bash
cd train_VTON_GAN
python train.py
```

### DiT
```bash
cd train_DIT
python train.py
```

---

## 📊 Dataset Requirements

### VTON Methods (CATVTON, IDM-VTON, CP-VTON, VTON-GAN)
```
dataset/
├── person/          # Person images
├── garment/         # Garment images
├── pose/            # Pose keypoints (CATVTON, VTON-GAN)
├── segmentation/    # Body segmentation (CATVTON)
├── mask/            # Inpainting masks (IDM-VTON)
├── densepose/       # DensePose maps (IDM-VTON)
├── person_repr/     # Person representations (CP-VTON)
├── target/          # Ground truth try-on results (CP-VTON, VTON-GAN)
├── train_pairs.txt  # Format: person_id garment_id
└── val_pairs.txt
```

### DiT
```
imagenet_dataset/
├── train_labels.txt  # Format: image_path class_id
├── val_labels.txt
└── images/
    ├── 0/  # Class folders
    ├── 1/
    └── ...
```

---

## 🔧 Configuration

Each training directory contains a `config.py` file with:
- WandB settings
- Hyperparameters (learning rate, batch size, epochs)
- Model architecture settings
- Dataset paths
- AWS S3 configuration
- Loss weights

---

## 📝 Files in Each Directory

Standard files across all directories:
- `config.py` - Configuration parameters
- `model.py` - Model architecture
- `train.py` - Training script
- `dataloader.py` - Dataset and dataloader
- `utils.py` - Checkpoint and utility functions
- `README.md` - Method-specific documentation

---

## 🎓 References

- **CATVTON:** Concatenation-based Attentive Virtual Try-On Network
- **IDM-VTON:** Improving Diffusion Models for Authentic Virtual Try-on in the Wild
- **CP-VTON:** Toward Characteristic-Preserving Image-based Virtual Try-On Network
- **VTON-GAN:** GAN-based approaches for virtual try-on
- **DiT:** Scalable Diffusion Models with Transformers (Peebles & Xie, 2023)
- **Stable Diffusion:** High-Resolution Image Synthesis with Latent Diffusion Models

---

**Last Updated:** 2026-01-22
