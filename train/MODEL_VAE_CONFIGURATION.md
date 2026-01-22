# Model & VAE Configuration Summary - Final

## ✅ Complete Model Assignments

All models have been configured with the appropriate Stable Diffusion versions and VAEs.

---

## Model Configurations

### **SD v1.5 Models** (6 models)

| Model | SD Version | VAE Source | Text Embedding Dim |
|-------|-----------|------------|-------------------|
| **CATVTON** | v1.5 | SD v1.5 | 768 |
| **CP-VTON** | v1.5 | SD v1.5 | 768 |
| **Stage 1** | v1.5 | SD v1.5 | 768 |
| **Stage 1-2** | v1.5 | SD v1.5 | 768 |
| **Stage 1-2-3** | v1.5 | SD v1.5 | 768 |
| **Mixture** | v1.5 | SD v1.5 | 768 |

**Model ID:** `runwayml/stable-diffusion-v1-5`

---

### **SD2 Base Models** (2 models)

| Model | SD Version | VAE Source | Text Embedding Dim |
|-------|-----------|------------|-------------------|
| **IDM-VTON** | 2-base | **SD2** | 1024 |
| **OOTDiffusion** | 2-base | **SD2** | 1024 |

**Model ID:** `stabilityai/stable-diffusion-2-base`

---

### **Custom Models**

#### **DiT** (Diffusion Transformer)
- **Architecture:** Custom DiT Transformer
- **VAE:** **SD2 Base VAE** ✅
- **VAE Source:** `stabilityai/stable-diffusion-2-base` (subfolder: `vae`)
- **No Text Encoder** (class-conditional only)

```python
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder="vae")
```

#### **VTON-GAN**
- **Architecture:** Custom GAN (ResNet Generator + PatchGAN Discriminator)
- **No pretrained diffusion model**
- **No VAE** (operates in pixel space)

---

## VAE Specifications

### SD v1.5 VAE
```python
Source: "runwayml/stable-diffusion-v1-5" subfolder "vae"
Architecture: AutoencoderKL
Latent Channels: 4
Downsampling Factor: 8
Scaling Factor: 0.18215

# Encoding
Image: [B, 3, 512, 512] → Latent: [B, 4, 64, 64]

# Decoding
Latent: [B, 4, 64, 64] → Image: [B, 3, 512, 512]
```

### SD2 Base VAE
```python
Source: "stabilityai/stable-diffusion-2-base" subfolder "vae"
Architecture: AutoencoderKL (same as SD v1.5)
Latent Channels: 4
Downsampling Factor: 8
Scaling Factor: 0.18215

# Encoding
Image: [B, 3, 512, 512] → Latent: [B, 4, 64, 64]

# Decoding
Latent: [B, 4, 64, 64] → Image: [B, 3, 512, 512]
```

**Note:** SD v1.5 and SD2 use the **same VAE architecture**, but SD2's VAE may have slightly different weights.

---

## Why SD2 VAE for DiT?

1. **Better quality** - SD2 VAE has improved training
2. **Consistency** - Matches SD2-based VTON models (IDM-VTON, OOTDiffusion)
3. **Future-proof** - SD2 is more recent
4. **Same architecture** - Compatible with SD v1.5 (4 channels, 8x downsampling)

---

## Loading Examples

### SD v1.5 VAE
```python
from diffusers import AutoencoderKL

# Option 1: Load from SD v1.5
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")

# Option 2: Load standalone
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
```

### SD2 Base VAE
```python
from diffusers import AutoencoderKL

# Load from SD2 Base
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder="vae")
```

### DiT with SD2 VAE
```python
from train.train_DIT.model import get_dit_model

# Automatically loads SD2 VAE
dit, vae = get_dit_model()
```

---

## Complete Model Summary

| Model | Base Model | VAE | Text Encoder | Embedding Dim |
|-------|-----------|-----|--------------|---------------|
| **CATVTON** | SD v1.5 | SD v1.5 | CLIP ViT-L/14 | 768 |
| **IDM-VTON** | SD2 Base | **SD2** | OpenCLIP ViT-H/14 | 1024 |
| **CP-VTON** | SD v1.5 | SD v1.5 | CLIP ViT-L/14 | 768 |
| **VTON-GAN** | Custom GAN | None | None | - |
| **OOTDiffusion** | SD2 Base | **SD2** | OpenCLIP ViT-H/14 | 1024 |
| **DiT** | Custom DiT | **SD2** ✅ | None (class-only) | - |
| **Stage 1/2/3** | SD v1.5 | SD v1.5 | CLIP ViT-L/14 | 768 |
| **Mixture** | SD v1.5 | SD v1.5 | CLIP ViT-L/14 | 768 |

---

## Key Points

✅ **DiT now uses SD2 VAE** (updated from sd-vae-ft-mse)  
✅ **IDM-VTON and OOTDiffusion use SD2 Base** (not SD2-inpainting)  
✅ **All other diffusion models use SD v1.5**  
✅ **VAE architecture is the same** (4 channels, 8x downsampling)  
✅ **VTON-GAN uses no VAE** (pixel-space GAN)  

---

## Configuration Files

### DiT (`train_DIT/model.py`)
```python
def get_dit_model():
    dit = DiTModel(...)
    
    # Load VAE from SD2 Base
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-2-base", 
        subfolder="vae"
    )
    vae.requires_grad_(False)
    
    return dit, vae
```

### IDM-VTON (`train_IDMVTON/config.py`)
```python
MODEL_NAME = "stabilityai/stable-diffusion-2-base"
```

### OOTDiffusion (`train_OOTDiffusion/config.py`)
```python
MODEL_NAME = "stabilityai/stable-diffusion-2-base"
```

### CATVTON (`train_CATVTON/config.py`)
```python
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
```

---

**All configurations are now correct and consistent!** ✅

**Last Updated:** 2026-01-22
