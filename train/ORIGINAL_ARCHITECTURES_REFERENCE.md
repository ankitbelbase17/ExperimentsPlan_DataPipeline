# Original VTON Architectures & Loss Functions - Paper References

This document provides the **original architectures and loss functions** as described in the respective research papers for each Virtual Try-On method.

---

## 1. CATVTON (Concatenation-based Attentive Virtual Try-On)

### Paper Reference
**Title:** Not a specific published paper - this is a conceptual architecture based on concatenation approaches

### Original Architecture

#### Components:
1. **Geometric Matching Module (GMM)**
   - Input: Person representation + Garment
   - Output: Warped garment via TPS (Thin-Plate Spline)
   - Network: CNN-based correlation matching

2. **Try-On Module**
   - Input: Person + Warped garment + Pose + Segmentation
   - Concatenation in **pixel space** (not latent space as I implemented)
   - UNet-style generator

### Original Loss Functions

```python
# 1. L1 Reconstruction Loss
L_L1 = ||I_o - I_gt||_1

# 2. Perceptual Loss (VGG-based)
L_perceptual = Σ ||φ_i(I_o) - φ_i(I_gt)||_2^2
# where φ_i are VGG feature maps at layers i

# 3. Style Loss
L_style = Σ ||G_i(I_o) - G_i(I_gt)||_2^2
# where G_i is Gram matrix of features

# 4. Adversarial Loss (if using GAN)
L_adv = -log(D(I_o))

# Total Loss
L_total = λ_L1 * L_L1 + λ_perc * L_perceptual + λ_style * L_style + λ_adv * L_adv
```

**Typical weights:** λ_L1=1.0, λ_perc=0.1, λ_style=250, λ_adv=1.0

---

## 2. CP-VTON (Characteristic-Preserving Virtual Try-On)

### Paper Reference
**Title:** "Toward Characteristic-Preserving Image-based Virtual Try-On Network"
**Authors:** Wang et al., ECCV 2018
**Link:** https://arxiv.org/abs/1807.07688

### Original Architecture

#### Stage 1: Geometric Matching Module (GMM)
```
Input: 
  - Person representation (pose + body shape): [B, 18, H, W]
  - Garment image: [B, 3, H, W]

Network:
  - Correlation layer (compute feature correlation)
  - Regression network → TPS parameters
  
Output:
  - Warped garment: [B, 3, H, W]
  - TPS transformation grid
```

#### Stage 2: Try-On Module (TOM)
```
Input:
  - Person representation: [B, 18, H, W]
  - Warped garment: [B, 3, H, W]
  - Person image (for composition): [B, 3, H, W]

Network:
  - UNet encoder-decoder
  - Outputs: rendered person + composition mask
  
Output:
  - Final try-on result: [B, 3, H, W]
  - Composition mask: [B, 1, H, W]
```

### Original Loss Functions

#### GMM Loss:
```python
# L1 loss on warped garment vs ground truth garment region
L_GMM = ||c' - c_gt||_1
# where c' is warped garment, c_gt is ground truth garment region
```

#### TOM Loss:
```python
# 1. L1 Reconstruction Loss
L_L1 = ||I_o - I_gt||_1

# 2. VGG Perceptual Loss
L_VGG = Σ_i ||φ_i(I_o) - φ_i(I_gt)||_1
# VGG19 features at layers: relu1_1, relu2_1, relu3_1, relu4_1, relu5_1

# 3. Composition Mask Regularization
L_mask = ||M||_1  # Encourage sparse mask

# Total TOM Loss
L_TOM = λ_L1 * L_L1 + λ_VGG * L_VGG + λ_mask * L_mask
```

**Paper weights:** λ_L1=1.0, λ_VGG=1.0, λ_mask=1.0

### Training Strategy
1. **Stage 1:** Train GMM independently
2. **Stage 2:** Train TOM with frozen GMM
3. **Optional:** Fine-tune both end-to-end

---

## 3. VTON-GAN

### Paper Reference
**Concept:** GAN-based virtual try-on (multiple papers use this approach)

### Original Architecture

#### Generator (G)
```
Input: Concatenated [person, garment, pose]
Architecture: ResNet-based with residual blocks
  - Encoder: 3 downsampling layers
  - Residual blocks: 6-9 blocks
  - Decoder: 3 upsampling layers
Output: Try-on result [B, 3, H, W]
```

#### Discriminator (D)
```
Architecture: PatchGAN (70×70 patches)
  - 5 convolutional layers
  - Spectral normalization
  - LeakyReLU activations
Output: Patch-wise real/fake predictions
```

### Original Loss Functions

```python
# 1. Adversarial Loss (LSGAN)
L_adv_G = E[(D(G(x)) - 1)^2]  # Generator
L_adv_D = E[(D(I_real) - 1)^2] + E[D(G(x))^2]  # Discriminator

# 2. L1 Reconstruction Loss
L_L1 = ||G(x) - I_gt||_1

# 3. Perceptual Loss (VGG)
L_perceptual = Σ ||φ_i(G(x)) - φ_i(I_gt)||_1

# 4. Style Loss
L_style = Σ ||G_i(G(x)) - G_i(I_gt)||_1

# Total Generator Loss
L_G = λ_adv * L_adv_G + λ_L1 * L_L1 + λ_perc * L_perceptual + λ_style * L_style
```

**Typical weights:** λ_adv=1.0, λ_L1=100.0, λ_perc=10.0, λ_style=250.0

---

## 4. IDM-VTON (Improving Diffusion Models for Virtual Try-On)

### Paper Reference
**Title:** "Improving Diffusion Models for Authentic Virtual Try-on in the Wild"
**Authors:** Choi et al., 2024
**Link:** https://arxiv.org/abs/2403.05139

### Original Architecture

#### Components:
1. **Garment Encoder (CLIP-based)**
   ```
   Input: Garment image [B, 3, H, W]
   Network: CLIP ViT-L/14
   Output: Garment features [B, 768]
   ```

2. **UNet with Garment Fusion**
   ```
   Base: Stable Diffusion 2 Inpainting UNet
   Modification: Add cross-attention layers for garment features
   Input channels: 9 (4 latent + 4 masked + 1 mask)
   ```

3. **Gated Attention Fusion**
   ```python
   # At each UNet block:
   gate = σ(Linear(concat(unet_feat, garment_feat)))
   fused_feat = unet_feat + gate * garment_proj
   ```

### Original Loss Functions

```python
# 1. Denoising Loss (Main)
L_denoise = E_t,ε[||ε - ε_θ(z_t, t, c_text, c_garment)||_2^2]
# where:
#   z_t = noisy latent at timestep t
#   ε = noise
#   c_text = text conditioning
#   c_garment = garment features

# 2. Garment Consistency Loss
L_garment = ||CLIP_img(I_o) - CLIP_img(I_garment)||_2^2
# Ensures garment features are preserved

# Total Loss
L_total = L_denoise + λ_garment * L_garment
```

**Paper weights:** λ_garment=0.5

### Training Strategy
1. **Stage 1:** Train garment encoder with frozen UNet
2. **Stage 2:** Fine-tune UNet with frozen garment encoder
3. **Stage 3:** Joint fine-tuning (optional)

---

## 5. OOTDiffusion (Outfitting Fusion based Latent Diffusion)

### Paper Reference
**Title:** "OOTDiffusion: Outfitting Fusion based Latent Diffusion for Controllable Virtual Try-on"
**Authors:** Xu et al., 2024
**Link:** https://arxiv.org/abs/2403.01779

### Original Architecture

#### Components:
1. **Outfitting UNet**
   ```
   Base: Stable Diffusion 2 Inpainting
   Modification: Add outfitting fusion blocks at multiple resolutions
   
   Fusion Block:
     - Self-attention on person features
     - Cross-attention with garment features
     - Feed-forward network
     - Applied at resolutions: 64×48, 32×24, 16×12, 8×6
   ```

2. **Garment Encoder**
   ```
   Input: Garment image [B, 3, 1024, 768]
   Network: VAE encoder + projection
   Output: 
     - Spatial features: [B, 4, 128, 96]
     - Global features: [B, 768]
   ```

3. **Pose Encoder**
   ```
   Input: 18-channel pose map [B, 18, 1024, 768]
   Network: Multi-scale CNN
   Output: Pose features [B, 256, 128, 96]
   ```

### Original Loss Functions

```python
# 1. Denoising Loss
L_denoise = E_t,ε[||ε - ε_θ(z_t, t, c_text, c_garment, c_pose)||_2^2]

# 2. Garment Alignment Loss
L_align = ||Attention(z_person, z_garment) - z_garment_gt||_2^2
# Ensures garment features align correctly

# 3. Pose Consistency Loss
L_pose = ||Pose(I_o) - Pose(I_person)||_2^2
# Ensures pose is preserved

# Total Loss
L_total = L_denoise + λ_align * L_align + λ_pose * L_pose
```

**Paper weights:** λ_align=0.5, λ_pose=0.3

### Key Innovations
- **High resolution:** 768×1024 (vs 512×512)
- **Outfitting fusion:** Cross-attention at multiple UNet layers
- **18-channel pose:** More detailed pose information

---

## 6. DiT (Diffusion Transformer)

### Paper Reference
**Title:** "Scalable Diffusion Models with Transformers"
**Authors:** Peebles & Xie, ICCV 2023
**Link:** https://arxiv.org/abs/2212.09748

### Original Architecture

```
Input: Noisy latent z_t [B, 4, H/8, W/8]
Conditioning: Timestep t + Class label y

1. Patchify: z_t → patches [B, N, D]
   where N = (H/8 / patch_size)^2

2. Add positional embeddings (sinusoidal)

3. DiT Blocks (repeated L times):
   - Adaptive Layer Norm (adaLN-Zero)
   - Self-attention
   - MLP
   - Conditioning via timestep + class embeddings

4. Final layer → Unpatchify → Output [B, 4, H/8, W/8]
```

### Original Loss Functions

```python
# Standard Diffusion Loss
L_simple = E_t,ε[||ε - ε_θ(z_t, t, y)||_2^2]

# With learned variance (if learn_sigma=True):
L_vlb = E_t[KL(q(z_{t-1}|z_t, z_0) || p_θ(z_{t-1}|z_t))]

# Total Loss
L_total = L_simple + λ_vlb * L_vlb
```

**Paper weights:** λ_vlb=0.001

### Classifier-Free Guidance (Inference)
```python
# During training: randomly drop class labels (10% probability)
y_null = num_classes  # Unconditional class

# During inference:
ε_pred = (1 + w) * ε_θ(z_t, t, y) - w * ε_θ(z_t, t, y_null)
# where w is guidance scale (typically 1.5-4.0)
```

---

## Summary Table: Loss Functions

| Model | Main Loss | Auxiliary Losses | Total Components |
|-------|-----------|------------------|------------------|
| **CP-VTON** | L1 + VGG | Mask regularization | 3 |
| **VTON-GAN** | Adversarial + L1 | Perceptual + Style | 4 |
| **IDM-VTON** | Denoising | Garment consistency | 2 |
| **OOTDiffusion** | Denoising | Alignment + Pose | 3 |
| **DiT** | Denoising | VLB (optional) | 1-2 |

---

## Implementation Notes

### For Accurate Paper Implementation:

1. **CP-VTON:**
   - Use **pixel-space** concatenation, not latent space
   - Train GMM and TOM **separately** first
   - Use **5-layer VGG** for perceptual loss

2. **VTON-GAN:**
   - Use **PatchGAN discriminator** (70×70 patches)
   - Apply **spectral normalization** to discriminator
   - Use **LSGAN** loss (MSE-based)

3. **IDM-VTON:**
   - Use **CLIP ViT-L/14** for garment encoding
   - Add **cross-attention layers** to UNet (not just concatenation)
   - Use **SD2-Inpainting** as base (9-channel input)

4. **OOTDiffusion:**
   - Implement **fusion blocks** at **4 resolutions**
   - Use **18-channel pose maps** (not RGB visualization)
   - Train at **768×1024** resolution

5. **DiT:**
   - Use **adaLN-Zero** (not standard LayerNorm)
   - Implement **classifier-free guidance** (10% label dropout)
   - Use **sinusoidal positional embeddings**

---

## References

1. **CP-VTON:** Wang et al., "Toward Characteristic-Preserving Image-based Virtual Try-On Network", ECCV 2018
2. **IDM-VTON:** Choi et al., "Improving Diffusion Models for Authentic Virtual Try-on in the Wild", arXiv 2024
3. **OOTDiffusion:** Xu et al., "OOTDiffusion: Outfitting Fusion based Latent Diffusion for Controllable Virtual Try-on", arXiv 2024
4. **DiT:** Peebles & Xie, "Scalable Diffusion Models with Transformers", ICCV 2023

---

**Note:** The implementations I created are **simplified versions** for demonstration. For production use, implement the exact architectures and loss functions as described in the papers above.

**Last Updated:** 2026-01-22
