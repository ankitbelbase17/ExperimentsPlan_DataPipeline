# Model Simplification Status

## ✅ COMPLETED:
### CATVTON
- **Model**: Redesigned to use only Person + Cloth
- **Architecture**: Cross-attention between garment features and person latents
- **Dataloader**: Simplified to load only person + garment
- **Training**: Updated to use new signature

## 🔄 IN PROGRESS:
### IDM-VTON
- Status: Pending

### CP-VTON  
- Status: Pending

### VTON-GAN
- Status: Pending

### OOTDiffusion
- Status: Pending

### DiT
- Status: Already works with triplets only (no changes needed)

## Key Changes Made:
1. Removed pose_map, segmentation, mask, densepose parameters from model forward()
2. Added GarmentEncoder to extract features from garment latents
3. Use cross-attention conditioning instead of concatenation
4. Dataloader only loads person + cloth (no auxiliary files)
5. Training script handles S3 dataset keys (initial_image, cloth_image)
