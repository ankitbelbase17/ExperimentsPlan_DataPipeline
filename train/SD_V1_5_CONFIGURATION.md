# Model Configuration Summary - Final

## ✅ Stable Diffusion Model Assignments

All models have been configured with the appropriate Stable Diffusion versions.

---

## Model Configurations

### **SD v1.5 Models** (5 models)

#### 1. **CATVTON** (`train_CATVTON/config.py`)
```python
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
```

#### 2. **CP-VTON** (`train_CP_VTON/config.py`)
```python
MODEL_NAME = "runwayml/stable-diffusion-v1-5"  # (If using diffusion)
```

#### 3. **Stage 1** (`train_stage_1/config.py`)
```python
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
```

#### 4. **Stage 1-2** (`train_stage_1_2/config.py`)
```python
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
```

#### 5. **Stage 1-2-3** (`train_stage_1_2_3/config.py`)
```python
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
```

#### 6. **Mixture** (`train_mixture/config.py`)
```python
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
```

---

### **SD2 Base Models** (2 models)

#### 7. **IDM-VTON** (`train_IDMVTON/config.py`)
```python
MODEL_NAME = "stabilityai/stable-diffusion-2-base"  # SD2 base (not inpainting)
```

#### 8. **OOTDiffusion** (`train_OOTDiffusion/config.py`)
```python
MODEL_NAME = "stabilityai/stable-diffusion-2-base"  # SD2 base (not inpainting)
```

---

### **GAN-based Models** (1 model)

#### 9. **VTON-GAN** (`train_VTON_GAN/`)
```python
# No pretrained diffusion model - custom GAN architecture
```

---

### **Transformer Models** (1 model)

#### 10. **DiT** (`train_DIT/config.py`)
```python
# Uses custom DiT architecture with SD VAE
# VAE from: "stabilityai/sd-vae-ft-mse"
```

---

## Specifications Comparison

### SD v1.5 vs SD2 Base

| Feature | SD v1.5 | SD2 Base |
|---------|---------|----------|
| **Model ID** | `runwayml/stable-diffusion-v1-5` | `stabilityai/stable-diffusion-2-base` |
| **UNet Input Channels** | 4 | 4 |
| **Text Encoder** | CLIP ViT-L/14 | OpenCLIP ViT-H/14 |
| **Text Embedding Dim** | 768 | 1024 |
| **Max Text Length** | 77 | 77 |
| **Training Resolution** | 512×512 | 512×512 (base), 768×768 (v2.1) |
| **VAE** | Same architecture | Same architecture |
| **Latent Channels** | 4 | 4 |
| **Latent Size** | H/8 × W/8 | H/8 × W/8 |

---

## Why Different Models?

### **SD v1.5** (CATVTON, CP-VTON, Stages, Mixture)
- ✅ Most widely used and tested
- ✅ Better community support
- ✅ Smaller text embeddings (768 vs 1024)
- ✅ Proven for 512×512 generation
- ✅ Good for general VTON tasks

### **SD2 Base** (IDM-VTON, OOTDiffusion)
- ✅ Better text understanding (OpenCLIP)
- ✅ Higher quality text conditioning
- ✅ More recent architecture improvements
- ✅ Better for complex conditioning (garment features, pose)
- ✅ Can scale to higher resolutions (768×768+)

---

## Tensor Specifications

### SD v1.5 Tensors

```python
# VAE
Image: [B, 3, 512, 512] → Latent: [B, 4, 64, 64]
Scaling: 0.18215

# UNet
Input: [B, 4, 64, 64]  # Noisy latents
Conditioning: [B, 77, 768]  # CLIP ViT-L/14
Output: [B, 4, 64, 64]  # Predicted noise

# Text Encoder
Input: [B, 77]  # Token IDs
Output: [B, 77, 768]  # Embeddings
```

### SD2 Base Tensors

```python
# VAE (same as SD v1.5)
Image: [B, 3, 512, 512] → Latent: [B, 4, 64, 64]
Scaling: 0.18215

# UNet
Input: [B, 4, 64, 64]  # Noisy latents
Conditioning: [B, 77, 1024]  # OpenCLIP ViT-H/14 (larger!)
Output: [B, 4, 64, 64]  # Predicted noise

# Text Encoder
Input: [B, 77]  # Token IDs
Output: [B, 77, 1024]  # Embeddings (1024-dim, not 768!)
```

---

## Important Notes for Implementation

### For SD2 Models (IDM-VTON, OOTDiffusion)

**Text Embedding Dimension Changed:**
```python
# SD v1.5
cross_attention_dim = 768

# SD2 Base
cross_attention_dim = 1024  # ⚠️ Different!
```

**Loading SD2 Components:**
```python
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

model_id = "stabilityai/stable-diffusion-2-base"

# Load components
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

# ⚠️ SD2 uses OpenCLIP, not standard CLIP
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
```

**Cross-Attention Compatibility:**
```python
# When adding custom cross-attention layers for garment features
# Make sure to match the text embedding dimension

# For SD v1.5
garment_proj = nn.Linear(garment_dim, 768)  # Match CLIP ViT-L/14

# For SD2
garment_proj = nn.Linear(garment_dim, 1024)  # Match OpenCLIP ViT-H/14
```

---

## Mask-Free VTON with SD2

### IDM-VTON (SD2 Base)

```python
# Encode inputs
person_latents = vae.encode(person_img)  # [B, 4, 64, 64]
target_latents = vae.encode(target_img)  # [B, 4, 64, 64]
cloth_features = clip_encoder(cloth_img)  # [B, 768] from CLIP

# Project cloth features to SD2 dimension
cloth_features_proj = proj_layer(cloth_features)  # [B, 1024]

# Text embeddings from SD2 text encoder
text_emb = text_encoder(input_ids)  # [B, 77, 1024]

# Combine conditioning
conditioning = torch.cat([
    text_emb,                           # [B, 77, 1024]
    cloth_features_proj.unsqueeze(1),  # [B, 1, 1024]
], dim=1)  # [B, 78, 1024]

# UNet forward
noise_pred = unet(
    noisy_latents,  # [B, 4, 64, 64]
    timesteps,
    encoder_hidden_states=conditioning  # [B, 78, 1024]
).sample
```

### OOTDiffusion (SD2 Base, High-Res)

```python
# Higher resolution
person_latents = vae.encode(person_img)  # [B, 4, 128, 96] (1024/8 × 768/8)
target_latents = vae.encode(target_img)  # [B, 4, 128, 96]

# Garment features
cloth_features = garment_encoder(cloth_img)  # [B, 768]
cloth_features_proj = proj_layer(cloth_features)  # [B, 1024] for SD2

# Fusion blocks with SD2 dimensions
fusion_output = fusion_blocks(
    person_latents,      # [B, 4, 128, 96]
    cloth_features_proj  # [B, 1024]
)

# UNet with SD2 text encoder
text_emb = text_encoder(input_ids)  # [B, 77, 1024]
noise_pred = unet(
    noisy_latents,
    timesteps,
    encoder_hidden_states=text_emb  # [B, 77, 1024]
).sample
```

---

## Summary Table

| Model | SD Version | Text Encoder | Embedding Dim | Resolution | Use Case |
|-------|-----------|--------------|---------------|------------|----------|
| **CATVTON** | v1.5 | CLIP ViT-L/14 | 768 | 512×512 | Multi-modal VTON |
| **IDM-VTON** | **2-base** | **OpenCLIP ViT-H/14** | **1024** | 512×512 | Diffusion VTON |
| **CP-VTON** | v1.5 | CLIP ViT-L/14 | 768 | 256×256 | Two-stage VTON |
| **VTON-GAN** | - | - | - | 256×256 | GAN-based VTON |
| **OOTDiffusion** | **2-base** | **OpenCLIP ViT-H/14** | **1024** | 768×1024 | High-res VTON |
| **Stage 1/2/3** | v1.5 | CLIP ViT-L/14 | 768 | 512×512 | General training |
| **Mixture** | v1.5 | CLIP ViT-L/14 | 768 | 512×512 | Mixed training |
| **DiT** | Custom | - | - | 256×256 | Transformer diffusion |

---

## Loading Models

### SD v1.5
```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
```

### SD2 Base
```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")
```

---

**All models are now correctly configured!** ✅

- **SD v1.5:** CATVTON, CP-VTON, Stage 1/2/3, Mixture
- **SD2 Base:** IDM-VTON, OOTDiffusion
- **Custom:** VTON-GAN, DiT

**Last Updated:** 2026-01-22
