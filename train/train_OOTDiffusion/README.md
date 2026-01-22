# OOTDiffusion: Outfitting Fusion based Latent Diffusion for Controllable Virtual Try-on

## Overview
OOTDiffusion is a state-of-the-art controllable virtual try-on method that uses outfitting fusion blocks to seamlessly integrate garment features with person representations in the latent diffusion space.

## Architecture

### Key Components

#### 1. Garment Encoder
**Purpose:** Extract garment-specific features for fusion

**Input:** `[B, 3, H, W]` - Garment image  
**Process:**
- VAE encoding → `[B, 4, H/8, W/8]`
- Garment-specific processing (Conv layers)
- Global feature projection → `[B, 768]`

**Output:**
- `garment_latents`: `[B, 4, H/8, W/8]` - Spatial features
- `garment_features`: `[B, 768]` - Global features for cross-attention

#### 2. Pose Encoder
**Purpose:** Encode pose information (OpenPose + DensePose)

**Input:** `[B, 18, H, W]` - Combined pose map  
**Process:** Multi-scale CNN encoder  
**Output:** `[B, 256, H/8, W/8]` - Pose features

#### 3. Fusion Blocks
**Purpose:** Cross-attention fusion between garment and person features

**Architecture:**
- Self-attention on person features
- Cross-attention with garment features
- Feed-forward network
- Applied at multiple UNet resolutions (320, 640, 1280 channels)

**Operation:**
```
person_features [B, C, H, W] + garment_features [B, 768]
    ↓ Self-Attention
    ↓ Cross-Attention with garment
    ↓ Feed-Forward
    → fused_features [B, C, H, W]
```

#### 4. Outfitting UNet
**Base:** SD2-Inpainting UNet (9-channel input)  
**Modification:** Integrated fusion blocks at multiple layers

### Tensor Flow

#### Training Forward Pass

**Input Tensors:**
| Tensor | Shape | Description |
|--------|-------|-------------|
| `person_img` | `[B, 3, 1024, 768]` | Target person (high-res) |
| `garment_img` | `[B, 3, 1024, 768]` | Garment to try on |
| `pose_map` | `[B, 18, 1024, 768]` | OpenPose + DensePose |
| `mask` | `[B, 1, 1024, 768]` | Inpainting mask |
| `input_ids` | `[B, 77]` | Text tokens |
| `timesteps` | `[B]` | Diffusion timesteps |

**Intermediate Tensors:**
| Tensor | Shape | Description |
|--------|-------|-------------|
| `person_latents` | `[B, 4, 128, 96]` | VAE-encoded person (1024/8 × 768/8) |
| `garment_latents` | `[B, 4, 128, 96]` | VAE-encoded garment |
| `garment_features` | `[B, 768]` | Global garment features |
| `pose_features` | `[B, 256, 128, 96]` | Encoded pose |
| `noisy_latents` | `[B, 4, 128, 96]` | Person + noise |
| `masked_latents` | `[B, 4, 128, 96]` | Masked person |
| `mask_latent` | `[B, 1, 128, 96]` | Downsampled mask |
| `unet_input` | `[B, 9, 128, 96]` | Concatenated input |
| `text_embeddings` | `[B, 77, 768]` | CLIP text embeddings |

**Output Tensors:**
| Tensor | Shape | Description |
|--------|-------|-------------|
| `noise_pred` | `[B, 4, 128, 96]` | Predicted noise |

### Loss Functions

#### 1. Reconstruction Loss (Main)
```python
loss_recon = MSE(noise_pred, noise)
```
**Weight:** 1.0

#### 2. Perceptual Loss (Optional)
```python
loss_perceptual = VGG_L1(noise_pred, noise)
```
**Weight:** 0.1

#### 3. Garment Consistency Loss
Ensures garment features are preserved during fusion
**Weight:** 0.5

#### 4. Pose Alignment Loss (Optional)
Ensures generated result aligns with pose
**Weight:** 0.3

## Training Strategies

### 1. Full Training (Default)
Train all components:
- UNet with fusion blocks
- Garment encoder
- Pose encoder
- Feature projectors

```python
TRAIN_UNET_ONLY = False
TRAIN_FUSION_ONLY = False
```

### 2. Fusion-Only Training
Only train fusion blocks and encoders (freeze base UNet)
```python
TRAIN_FUSION_ONLY = True
```

### 3. UNet-Only Training
Only train UNet (freeze encoders)
```python
TRAIN_UNET_ONLY = True
```

### 4. LoRA Fine-tuning (Optional)
Efficient fine-tuning with Low-Rank Adaptation
```python
USE_LORA = True
LORA_RANK = 64
```

## Key Features

✅ **High Resolution:** 768×1024 images (vs 512×512 in other methods)  
✅ **Fusion Blocks:** Cross-attention between garment and person  
✅ **Multi-Modal Conditioning:** Pose + Garment + Text  
✅ **Controllable:** Explicit garment feature control  
✅ **Inpainting-Based:** Preserves person identity  

## Training Command
```bash
python train.py
```

## Dataset Structure
```
ootdiffusion_dataset/
├── person/          # Person images (768x1024)
├── garment/         # Garment images
├── pose/            # Pose maps (.npy files, 18 channels)
│   └── {id}.npy     # Shape: [18, 1024, 768]
├── mask/            # Inpainting masks
├── train_pairs.txt  # Format: person_id garment_id
└── val_pairs.txt
```

### Pose Map Channels (18 total)
- **Channels 0-17:** OpenPose keypoints (18 body keypoints)
- Can be extended with DensePose UV maps

## Inference

### Basic Inference
```python
from model import get_ootdiffusion_model
import torch

model = get_ootdiffusion_model()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate try-on result
with torch.no_grad():
    result = model.generate(
        person_img=person,
        garment_img=garment,
        pose_map=pose,
        mask=mask,
        prompt="a person wearing the garment",
        num_inference_steps=50,
        guidance_scale=2.0
    )
```

### Controllable Generation
- **Garment Control:** Adjust garment feature strength
- **Pose Control:** Modify pose guidance weight
- **Text Control:** Use descriptive prompts

## Comparison with Other Methods

| Method | Resolution | Fusion | Pose | Base Model |
|--------|-----------|--------|------|------------|
| **OOTDiffusion** | 768×1024 | ✅ Cross-Attn | ✅ 18-ch | SD2-Inpainting |
| CATVTON | 512×512 | Concat | ✅ RGB | SD1.5 |
| IDM-VTON | 512×512 | Gated Attn | ✅ DensePose | SD2-Inpainting |
| CP-VTON | 256×256 | Two-stage | ✅ RGB | Custom |
| VTON-GAN | 256×256 | None | ✅ RGB | GAN |

## Advantages

1. **Higher Quality:** 768×1024 resolution for detailed results
2. **Better Fusion:** Cross-attention fusion vs simple concatenation
3. **Controllability:** Explicit control over garment features
4. **Flexibility:** Multiple training strategies
5. **State-of-the-art:** Based on latest diffusion techniques

## Configuration

Key parameters in `config.py`:
- `IMAGE_HEIGHT = 1024` - Image height
- `IMAGE_WIDTH = 768` - Image width
- `USE_FUSION_BLOCKS = True` - Enable fusion blocks
- `NUM_FUSION_LAYERS = 4` - Number of fusion layers
- `GARMENT_ENCODER_TYPE = "vae"` - Garment encoder type
- `USE_POSE_GUIDANCE = True` - Enable pose conditioning

## References
- Based on Stable Diffusion 2 Inpainting
- Inspired by ControlNet for conditional generation
- Uses cross-attention fusion similar to IP-Adapter

---

**Last Updated:** 2026-01-22
