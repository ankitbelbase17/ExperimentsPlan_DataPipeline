# Model Architecture & Input/Output Details

This document details how each model handles input tensors, outputs, and loss functions.

## 1. CATVTON (Diffusion)
**Architecture:** Stable Diffusion UNet + ControlNet-like injection.
- **Inputs:**
  - `person_img` (Batch, 3, 512, 512): Target person image.
  - `garment_img` (Batch, 3, 512, 512): Garment to try on.
  - `pose_map` (Batch, 18, 512, 512): DensePose/OpenPose map.
  - `segmentation` (Batch, 1, 512, 512): Body part masks.
  - `noise` (Batch, 4, 64, 64): Gaussian noise for diffusion training.
  - `timesteps`: Diffusion step indices.
- **Outputs:**
  - `noise_pred` (Batch, 4, 64, 64): Predicted noise at current timestep.
- **Loss Function:**
  - `MSE Loss` between `noise_pred` and `noise`.

## 2. IDM-VTON (Diffusion)
**Architecture:** Improved Diffusion Model with densepose conditioning.
- **Inputs:**
  - `person_img`: Reference person.
  - `garment_img`: In-shop garment.
  - `mask`: Inpainting mask (1 = keep, 0 = regenerate).
  - `densepose`: DensePose visualizations for structure.
  - `text_embeddings`: CLIP embeddings of "a person wearing X".
- **Outputs:**
  - `noise_pred`: Predicted noise.
- **Loss Function:**
  - `MSE Loss` (Reconstruction).

## 3. CP-VTON (Geometric + Try-On)
**Architecture:** Two-stage (GMM Warping + Try-On Module).
- **Inputs:**
  - `person`: Reference person image.
  - `garment`: Target garment.
  - `person_repr`: Pose keypoints/segmentation map.
- **Outputs:**
  - `tryon_result`: Synthesized image.
  - `warped_garment`: Deformed garment to fit pose.
  - `composition_mask`: Alpha mask for blending.
- **Loss Functions:**
  - `L1 Loss`: Pixel-wise absolute difference.
  - `VGG Loss`: Perceptual feature difference (VGG19).
  - `Mask Loss`: Binary Cross Entropy on composition mask.

## 4. VTON-GAN (Adversarial)
**Architecture:** Generator (Try-On) vs Discriminator (Real/Fake).
- **Inputs:**
  - **Generator:** `person`, `garment`, `pose`.
  - **Discriminator:** `fake_tryon` (from G) or `target` (Ground Truth).
- **Outputs:**
  - `fake_tryon`: Generated image.
  - `validity`: Discriminator score (Real/Fake).
- **Loss Functions:**
  - **Generator:** Adversarial Loss + L1 Reconstruction Loss.
  - **Discriminator:** GAN Loss (MSE for LSGAN or BCE).

## 5. OOTDiffusion (Latent Diffusion)
**Architecture:** Latent Diffusion with 'Outfitting Fusion'.
- **Inputs:**
  - `person_img`, `garment_img`, `pose_map` (18-ch), `mask`.
  - `text_embeddings` (CLIP).
- **Outputs:**
  - `noise_pred`: Predicted noise.
  - `garment_features`: For consistency loss.
- **Loss Functions:**
  - `Reconstruction Loss` (MSE).
  - `Perceptual Loss` (L1/VGG).
  - `Garment Consistency Loss`.

## 6. DiT (Transformer)
**Architecture:** Diffusion Transformer (No UNet).
- **Inputs:**
  - `latents` (Batch, 4, 32, 32): VAE-encoded image latent.
  - `labels`: Class labels (or zeros).
  - `timesteps`.
- **Outputs:**
  - `model_pred`: Predicted noise (or velocity flow).
- **Loss Function:**
  - `MSE Loss`: Standard diffusion objective.
