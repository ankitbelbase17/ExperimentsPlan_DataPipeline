# DiT: Scalable Diffusion Models with Transformers

## Overview
DiT (Diffusion Transformer) replaces the U-Net backbone in diffusion models with a Vision Transformer architecture, achieving state-of-the-art image generation quality.

## Architecture

### Model Components

#### 1. Patchification
**Input:** `[B, 4, H/8, W/8]` latents from VAE  
**Process:** Split into patches of size `patch_size × patch_size`  
**Output:** `[B, N, D]` where `N = (H/8/patch_size)²` tokens

#### 2. Timestep & Label Conditioning
**Timestep Embedding:**
- Input: `[B]` timesteps
- Output: `[B, D]` sinusoidal embeddings

**Label Embedding:**
- Input: `[B]` class labels (0-999 for ImageNet)
- Output: `[B, D]` learned embeddings
- Supports classifier-free guidance via dropout

**Combined Conditioning:** `c = timestep_emb + label_emb` → `[B, D]`

#### 3. DiT Blocks (Adaptive Layer Norm)
Each block contains:
- **Self-Attention:** Multi-head attention over patch tokens
- **MLP:** Feed-forward network
- **adaLN-Zero:** Adaptive layer normalization conditioned on `c`

**Modulation:** `output = LayerNorm(x) * (1 + scale) + shift`

#### 4. Final Layer
**Input:** `[B, N, D]` tokens  
**Output:** `[B, patch_size² * out_channels, N]`  
**Unpatchify:** Reshape to `[B, out_channels, H/8, W/8]`

### Tensor Flow

#### Training (Diffusion Objective)
```
Input Tensors:
├── images: [B, 3, H, W] - RGB images
├── labels: [B] - Class labels
└── timesteps: [B] - Sampled timesteps [0, 1000)

Intermediate:
├── latents: [B, 4, H/8, W/8] - VAE encoded
├── noise: [B, 4, H/8, W/8] - Gaussian noise
├── noisy_latents: [B, 4, H/8, W/8] - latents + noise
├── patches: [B, N, D] - Patchified latents
├── conditioning: [B, D] - timestep + label embeddings
└── transformer_out: [B, N, D] - After DiT blocks

Output Tensors:
├── noise_pred: [B, 4, H/8, W/8] - Predicted noise
└── (optional) sigma_pred: [B, 4, H/8, W/8] - Predicted variance
```

#### Training (Rectified Flow Objective)
```
Input Tensors:
├── images: [B, 3, H, W]
├── labels: [B]
└── t: [B, 1, 1, 1] - Interpolation time ~ Uniform[0, 1]

Intermediate:
├── z1 (latents): [B, 4, H/8, W/8] - Data distribution
├── z0 (noise): [B, 4, H/8, W/8] - Noise distribution
├── zt: [B, 4, H/8, W/8] - Interpolated: t*z1 + (1-t)*z0
└── target_v: [B, 4, H/8, W/8] - Velocity: z1 - z0

Output Tensors:
└── v_pred: [B, 4, H/8, W/8] - Predicted velocity
```

## Configuration

### Model Sizes
| Model | Depth | Hidden Size | Heads | Params |
|-------|-------|-------------|-------|--------|
| DiT-S | 12 | 384 | 6 | 33M |
| DiT-B | 12 | 768 | 12 | 130M |
| DiT-L | 24 | 1024 | 16 | 458M |
| DiT-XL | 28 | 1152 | 16 | 675M |

Default configuration uses **DiT-XL/2** (patch_size=2).

## Training Objectives

### 1. Diffusion (DDPM)
```python
loss = MSE(noise_pred, noise)
```

### 2. Rectified Flow
```python
loss = MSE(v_pred, z1 - z0)
```

### 3. Flow Matching
```python
loss = MSE(v_pred, z1 - z0)  # Similar to rectified flow
```

## Key Features
- **Classifier-Free Guidance:** Label dropout during training
- **EMA:** Exponential moving average of model weights
- **Large Batch Training:** Batch size 256+ recommended
- **Mixed Precision:** BFloat16 for stability
- **Gradient Clipping:** Prevents exploding gradients

## Training Command
```bash
python train.py
```

## Dataset Structure
```
imagenet_dataset/
├── train_labels.txt  # Format: image_path class_id
├── val_labels.txt
└── images/
    ├── 0/  # Class 0
    ├── 1/  # Class 1
    └── ...
```

## Inference
DiT supports classifier-free guidance during sampling:
```python
# Unconditional: y = num_classes
# Conditional: y = class_label
# CFG: output = (1 + guidance_scale) * cond - guidance_scale * uncond
```

## References
- Paper: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
- Original implementation: https://github.com/facebookresearch/DiT
