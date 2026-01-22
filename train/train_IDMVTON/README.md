# IDM-VTON: Improving Diffusion Models for Virtual Try-On

## Overview
IDM-VTON uses Stable Diffusion 2 Inpainting with a specialized garment encoder and gated attention fusion for high-quality virtual try-on.

## Architecture

### Input Tensors
| Tensor Name | Shape | Description |
|-------------|-------|-------------|
| `person_img` | `[B, 3, H, W]` | Target person image |
| `garment_img` | `[B, 3, H, W]` | Garment image |
| `mask` | `[B, 1, H, W]` | Binary inpainting mask (1=keep, 0=inpaint) |
| `densepose` | `[B, 3, H, W]` | DensePose body representation |
| `input_ids` | `[B, 77]` | Text tokens |

### Intermediate Tensors
| Tensor Name | Shape | Description |
|-------------|-------|-------------|
| `garment_features` | `[B, 768]` | CLIP-encoded garment features |
| `person_latents` | `[B, 4, H/8, W/8]` | VAE-encoded person |
| `masked_latents` | `[B, 4, H/8, W/8]` | Masked person latents |
| `mask_latent` | `[B, 1, H/8, W/8]` | Downsampled mask |
| `unet_input` | `[B, 9, H/8, W/8]` | Concatenated inpainting input |
| `text_embeddings` | `[B, 77, 768]` | CLIP text embeddings |

### Output Tensors
| Tensor Name | Shape | Description |
|-------------|-------|-------------|
| `noise_pred` | `[B, 4, H/8, W/8]` | Predicted noise |

## Key Features
1. **CLIP Garment Encoder**: Extracts semantic garment features
2. **SD2-Inpainting Base**: Uses inpainting UNet (9-channel input)
3. **Gated Attention Fusion**: Fuses garment features into UNet layers
4. **DensePose Conditioning**: Body-aware generation
5. **Two-Stage Training**: First garment encoder, then full model

## Training Command
```bash
python train.py
```

## Dataset Structure
```
idm_vton_dataset/
├── person/          # Person images
├── garment/         # Garment images
├── mask/            # Inpainting masks (1=keep, 0=generate)
├── densepose/       # DensePose visualizations
├── train_pairs.txt
└── val_pairs.txt
```
