# Input/Output Tensors for Mask-Free VTON Training

## Training Setup

**Mask-Free Training** means we only have three images per sample:
1. **Person in different cloth** (reference person image)
2. **Target cloth image** (garment to try on)
3. **Ground truth** (person wearing target cloth)

**No masks, no pose maps, no segmentation required!**

---

## Data Format

### Available Data (from S3VTONDataset)
```python
{
    "initial_image": [B, 3, H, W],    # Person in different cloth
    "cloth_image": [B, 3, H, W],      # Target garment
    "try_on_image": [B, 3, H, W],     # Ground truth (person wearing target cloth)
    "difficulty": str,                # "easy", "medium", or "hard"
    "stem": str                       # Sample identifier
}
```

---

## Model-by-Model Tensor Flow (Mask-Free)

### 1. **CP-VTON** (Simplest for Mask-Free)

#### Input Tensors:
```python
person_img = batch['initial_image']     # [B, 3, 256, 256] - Person in different cloth
cloth_img = batch['cloth_image']        # [B, 3, 256, 256] - Target garment
target_img = batch['try_on_image']      # [B, 3, 256, 256] - Ground truth
```

#### Stage 1 - GMM (Geometric Matching):
```python
Input:
  - person_img: [B, 3, 256, 256]
  - cloth_img: [B, 3, 256, 256]

Processing:
  - Extract features from both
  - Compute correlation
  - Predict TPS parameters: [B, 50]
  
Output:
  - warped_cloth: [B, 3, 256, 256]  # Cloth aligned to person shape
```

#### Stage 2 - TOM (Try-On Module):
```python
Input:
  - person_img: [B, 3, 256, 256]
  - warped_cloth: [B, 3, 256, 256]
  - Concatenated: [B, 6, 256, 256]  # Simple concatenation

Processing:
  - UNet encoder-decoder
  
Output:
  - tryon_result: [B, 3, 256, 256]  # Final try-on result
  - composition_mask: [B, 1, 256, 256]  # Soft blending mask

Loss:
  L = ||tryon_result - target_img||_1 + VGG_loss(tryon_result, target_img)
```

**Total Flow:**
```
person_img [B,3,256,256] ──┐
                           ├──> GMM ──> warped_cloth [B,3,256,256] ──┐
cloth_img [B,3,256,256] ───┘                                         │
                                                                      ├──> TOM ──> result [B,3,256,256]
person_img [B,3,256,256] ────────────────────────────────────────────┘

Target: try_on_image [B,3,256,256]
```

---

### 2. **VTON-GAN** (GAN-based, Mask-Free)

#### Input Tensors:
```python
person_img = batch['initial_image']     # [B, 3, 256, 256]
cloth_img = batch['cloth_image']        # [B, 3, 256, 256]
target_img = batch['try_on_image']      # [B, 3, 256, 256]
```

#### Generator:
```python
Input:
  - Concatenated: [person_img, cloth_img] = [B, 6, 256, 256]

Processing:
  - ResNet-based generator with residual blocks
  
Output:
  - fake_tryon: [B, 3, 256, 256]  # Generated try-on result
```

#### Discriminator:
```python
Input (Real):
  - target_img: [B, 3, 256, 256]
  
Input (Fake):
  - fake_tryon: [B, 3, 256, 256]

Output:
  - prediction: [B, 1, H', W']  # PatchGAN predictions
```

#### Loss:
```python
# Generator Loss
L_G = L_adv + λ_L1 * ||fake_tryon - target_img||_1 + λ_perc * VGG_loss

# Discriminator Loss
L_D = MSE(D(target_img), 1) + MSE(D(fake_tryon.detach()), 0)
```

**Total Flow:**
```
person_img [B,3,256,256] ──┐
                           ├──> concat [B,6,256,256] ──> Generator ──> fake_tryon [B,3,256,256]
cloth_img [B,3,256,256] ───┘                                                │
                                                                             ├──> Discriminator ──> real/fake
target_img [B,3,256,256] ────────────────────────────────────────────────────┘
```

---

### 3. **IDM-VTON** (Diffusion-based, Mask-Free Adaptation)

#### Input Tensors:
```python
person_img = batch['initial_image']     # [B, 3, 512, 512]
cloth_img = batch['cloth_image']        # [B, 3, 512, 512]
target_img = batch['try_on_image']      # [B, 3, 512, 512]
```

#### Processing:
```python
# 1. Encode to latent space
person_latents = vae.encode(person_img).latent_dist.sample() * 0.18215  # [B, 4, 64, 64]
target_latents = vae.encode(target_img).latent_dist.sample() * 0.18215  # [B, 4, 64, 64]

# 2. Extract garment features
cloth_features = garment_encoder(cloth_img)  # [B, 768] - CLIP features

# 3. Add noise (diffusion)
noise = torch.randn_like(target_latents)  # [B, 4, 64, 64]
timesteps = torch.randint(0, 1000, (B,))  # [B]
noisy_latents = scheduler.add_noise(target_latents, noise, timesteps)  # [B, 4, 64, 64]

# 4. Create conditioning (mask-free: use person latents as condition)
# Instead of masked latents, use person latents directly
unet_input = torch.cat([
    noisy_latents,      # [B, 4, 64, 64] - Noisy target
    person_latents,     # [B, 4, 64, 64] - Person condition
], dim=1)  # [B, 8, 64, 64]

# 5. UNet prediction with garment cross-attention
noise_pred = unet(
    unet_input,                    # [B, 8, 64, 64]
    timesteps,                     # [B]
    encoder_hidden_states=text_emb,  # [B, 77, 768]
    garment_features=cloth_features  # [B, 768] - Cross-attention
).sample  # [B, 4, 64, 64]
```

#### Loss:
```python
L = ||noise_pred - noise||_2^2
```

**Total Flow:**
```
person_img [B,3,512,512] ──> VAE ──> person_latents [B,4,64,64] ──┐
                                                                   │
cloth_img [B,3,512,512] ──> CLIP ──> cloth_features [B,768] ──────┼──> UNet ──> noise_pred [B,4,64,64]
                                                                   │              │
target_img [B,3,512,512] ──> VAE ──> target_latents [B,4,64,64] ──┤              │
                                     + noise ──> noisy_latents ────┘              │
                                                                                   │
Target: noise [B,4,64,64] <────────────────────────────────────────────────────────┘
```

---

### 4. **OOTDiffusion** (High-Res Diffusion, Mask-Free Adaptation)

#### Input Tensors:
```python
person_img = batch['initial_image']     # [B, 3, 1024, 768]
cloth_img = batch['cloth_image']        # [B, 3, 1024, 768]
target_img = batch['try_on_image']      # [B, 3, 1024, 768]
```

#### Processing:
```python
# 1. Encode to latent space (higher resolution)
person_latents = vae.encode(person_img).latent_dist.sample() * 0.18215  # [B, 4, 128, 96]
target_latents = vae.encode(target_img).latent_dist.sample() * 0.18215  # [B, 4, 128, 96]

# 2. Garment encoding (spatial + global)
cloth_latents = garment_encoder(cloth_img)  # [B, 4, 128, 96] - Spatial features
cloth_features = global_proj(cloth_latents.flatten())  # [B, 768] - Global features

# 3. Diffusion process
noise = torch.randn_like(target_latents)  # [B, 4, 128, 96]
timesteps = torch.randint(0, 1000, (B,))
noisy_latents = scheduler.add_noise(target_latents, noise, timesteps)

# 4. Conditioning (mask-free: use person latents)
unet_input = torch.cat([
    noisy_latents,      # [B, 4, 128, 96]
    person_latents,     # [B, 4, 128, 96]
], dim=1)  # [B, 8, 128, 96]

# 5. UNet with fusion blocks
noise_pred = outfitting_unet(
    unet_input,
    timesteps,
    encoder_hidden_states=text_emb,
    garment_features=cloth_features  # Fused via cross-attention
).sample  # [B, 4, 128, 96]
```

#### Loss:
```python
L = ||noise_pred - noise||_2^2 + λ * garment_consistency_loss
```

**Total Flow:**
```
person_img [B,3,1024,768] ──> VAE ──> person_latents [B,4,128,96] ──┐
                                                                     │
cloth_img [B,3,1024,768] ──> Garment Encoder ──> cloth_features ────┼──> Outfitting UNet ──> noise_pred
                                                  [B,768]            │
target_img [B,3,1024,768] ──> VAE ──> target_latents [B,4,128,96] ──┤
                                      + noise ──> noisy_latents ─────┘

Target: noise [B,4,128,96]
```

---

### 5. **CATVTON** (Mask-Free Adaptation)

**Note:** Original CATVTON uses pose/segmentation, but for mask-free:

#### Input Tensors:
```python
person_img = batch['initial_image']     # [B, 3, 512, 512]
cloth_img = batch['cloth_image']        # [B, 3, 512, 512]
target_img = batch['try_on_image']      # [B, 3, 512, 512]
```

#### Simplified Processing (Mask-Free):
```python
# 1. Encode to latent space
person_latents = vae.encode(person_img).latent_dist.sample() * 0.18215  # [B, 4, 64, 64]
cloth_latents = vae.encode(cloth_img).latent_dist.sample() * 0.18215    # [B, 4, 64, 64]
target_latents = vae.encode(target_img).latent_dist.sample() * 0.18215  # [B, 4, 64, 64]

# 2. Concatenate (simplified, no pose/seg)
unet_input = torch.cat([
    person_latents,  # [B, 4, 64, 64]
    cloth_latents,   # [B, 4, 64, 64]
], dim=1)  # [B, 8, 64, 64]

# 3. Add noise to target
noise = torch.randn_like(target_latents)
timesteps = torch.randint(0, 1000, (B,))
noisy_target = scheduler.add_noise(target_latents, noise, timesteps)

# 4. UNet prediction
unet_input_full = torch.cat([noisy_target, unet_input], dim=1)  # [B, 12, 64, 64]
noise_pred = unet(unet_input_full, timesteps, text_emb).sample  # [B, 4, 64, 64]
```

#### Loss:
```python
L = ||noise_pred - noise||_2^2
```

---

## Summary Table: Mask-Free VTON

| Model | Input Concat | Latent/Pixel | Resolution | Special Features |
|-------|--------------|--------------|------------|------------------|
| **CP-VTON** | `[person, cloth]` | Pixel | 256×256 | Two-stage (GMM+TOM) |
| **VTON-GAN** | `[person, cloth]` | Pixel | 256×256 | GAN discriminator |
| **IDM-VTON** | `[noisy, person]` | Latent | 512×512 | CLIP garment features |
| **OOTDiffusion** | `[noisy, person]` | Latent | 1024×768 | Fusion blocks, high-res |
| **CATVTON** | `[noisy, person, cloth]` | Latent | 512×512 | Simplified concatenation |

---

## Recommended Approach for Mask-Free Training

### **Best Choice: IDM-VTON or OOTDiffusion**

**Why?**
- ✅ Designed for diffusion-based generation
- ✅ Can work without explicit masks
- ✅ Use person latents as conditioning (instead of masked latents)
- ✅ CLIP garment features provide strong guidance

### **Simplified Training Flow:**

```python
# Pseudocode for mask-free VTON training
for batch in dataloader:
    person_img = batch['initial_image']    # Person in different cloth
    cloth_img = batch['cloth_image']       # Target garment
    target_img = batch['try_on_image']     # Ground truth
    
    # Encode
    person_latents = vae.encode(person_img)
    target_latents = vae.encode(target_img)
    cloth_features = clip_encoder(cloth_img)
    
    # Diffusion
    noise = randn_like(target_latents)
    timesteps = randint(0, 1000)
    noisy_latents = add_noise(target_latents, noise, timesteps)
    
    # Condition on person (instead of mask)
    conditioning = person_latents
    
    # Predict
    noise_pred = unet(
        noisy_latents,
        timesteps,
        person_condition=conditioning,
        garment_features=cloth_features
    )
    
    # Loss
    loss = mse_loss(noise_pred, noise)
    loss.backward()
```

---

## Key Insight for Mask-Free Training

**Instead of:**
```python
masked_latents = target_latents * mask  # Requires mask
```

**Use:**
```python
person_latents = vae.encode(person_img)  # Person as condition
```

The **person image in different cloth** serves as the structural/pose reference, eliminating the need for explicit masks!

---

**This is much simpler and aligns perfectly with your S3VTONDataset structure!**
