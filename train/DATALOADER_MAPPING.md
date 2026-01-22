# Dataloader Usage by Training Model

## Overview

This document maps each training model to its corresponding dataloader implementation and expected data structure.

---

## 📊 Summary Table

| Training Model | Dataset Class | Data Structure | Input Keys | Notes |
|----------------|---------------|----------------|------------|-------|
| **CATVTON** | `CATVTONDataset` | Pairs-based | person, garment, pose, segmentation | 4 modalities |
| **IDM-VTON** | `IDMVTONDataset` | Pairs-based | person, garment, mask, densepose | Inpainting-based |
| **CP-VTON** | `CPVTONDataset` | Pairs-based | person, garment, person_repr, target | Two-stage |
| **VTON-GAN** | `VTONGANDataset` | Pairs-based | person, garment, pose, target | GAN training |
| **OOTDiffusion** | `OOTDiffusionDataset` | Pairs-based | person, garment, pose (18-ch), mask | High-res 768×1024 |
| **DiT** | `ImageNetDataset` | Class-based | image, label | ImageNet-style |
| **Stage 1** | `Stage1Dataset` | S3 listing | pixel_values | Simple image listing |
| **Stage 1-2** | `S3ImageDataset` | S3 listing | pixel_values | Simple image listing |
| **Stage 1-2-3** | `S3ImageDataset` | S3 listing | pixel_values | Simple image listing |
| **Mixture** | `MixtureDataset` | S3 listing | pixel_values | Simple image listing |

---

## 🎨 VTON Models (Virtual Try-On)

### 1. **CATVTON** (`train_CATVTON/dataloader.py`)

**Dataset Class:** `CATVTONDataset`

**Expected S3 Structure:**
```
{root_dir}/
├── person/{id}.jpg
├── garment/{id}.jpg
├── pose/{id}.jpg
├── segmentation/{id}.png
└── train_pairs.txt  # Format: person_id garment_id
```

**Output Format:**
```python
{
    "person_img": [3, 512, 512],      # Person image
    "garment_img": [3, 512, 512],     # Garment image
    "pose_map": [3, 512, 512],        # Pose keypoints (RGB visualization)
    "segmentation": [3, 512, 512],    # Body segmentation
    "input_ids": [77],                # CLIP tokens
    "person_id": str,
    "garment_id": str
}
```

**Usage:**
```python
from train.train_CATVTON.dataloader import get_dataloader

dataloader = get_dataloader(
    tokenizer=model.tokenizer,
    batch_size=4,
    split='train'
)
```

---

### 2. **IDM-VTON** (`train_IDMVTON/dataloader.py`)

**Dataset Class:** `IDMVTONDataset`

**Expected S3 Structure:**
```
{root_dir}/
├── person/{id}.jpg
├── garment/{id}.jpg
├── mask/{id}.png          # Inpainting mask (1=keep, 0=generate)
├── densepose/{id}.jpg     # DensePose visualization
└── train_pairs.txt
```

**Output Format:**
```python
{
    "person_img": [3, 512, 512],
    "garment_img": [3, 512, 512],
    "mask": [1, 512, 512],           # Binary mask
    "densepose": [3, 512, 512],      # DensePose
    "input_ids": [77]
}
```

**Key Feature:** Inpainting-based approach with DensePose for body awareness

---

### 3. **CP-VTON** (`train_CP_VTON/dataloader.py`)

**Dataset Class:** `CPVTONDataset`

**Expected S3 Structure:**
```
{root_dir}/
├── person/{id}.jpg
├── garment/{id}.jpg
├── person_repr/{id}.jpg   # Person representation (pose/segmentation)
├── target/{person_id}_{garment_id}.jpg  # Ground truth
└── train_pairs.txt
```

**Output Format:**
```python
{
    "person": [3, 256, 256],
    "garment": [3, 256, 256],
    "person_repr": [3, 256, 256],
    "target": [3, 256, 256]         # Ground truth try-on result
}
```

**Key Feature:** Two-stage architecture (GMM + TOM), pixel-space training

---

### 4. **VTON-GAN** (`train_VTON_GAN/dataloader.py`)

**Dataset Class:** `VTONGANDataset`

**Expected S3 Structure:**
```
{root_dir}/
├── person/{id}.jpg
├── garment/{id}.jpg
├── pose/{id}.jpg
├── target/{person_id}_{garment_id}.jpg
└── train_pairs.txt
```

**Output Format:**
```python
{
    "person": [3, 256, 256],
    "garment": [3, 256, 256],
    "pose": [3, 256, 256],
    "target": [3, 256, 256]
}
```

**Key Feature:** GAN-based training with adversarial loss

---

### 5. **OOTDiffusion** (`train_OOTDiffusion/dataloader.py`)

**Dataset Class:** `OOTDiffusionDataset`

**Expected S3 Structure:**
```
{root_dir}/
├── person/{id}.jpg
├── garment/{id}.jpg
├── pose/{id}.npy          # 18-channel pose map (OpenPose + DensePose)
├── mask/{id}.png
└── train_pairs.txt
```

**Output Format:**
```python
{
    "person_img": [3, 1024, 768],    # High resolution!
    "garment_img": [3, 1024, 768],
    "pose_map": [18, 1024, 768],     # 18-channel pose
    "mask": [1, 1024, 768],
    "input_ids": [77],
    "person_id": str,
    "garment_id": str
}
```

**Key Features:**
- **Highest resolution:** 768×1024 (vs 512×512 or 256×256)
- **18-channel pose maps:** More detailed pose information
- **NumPy pose files:** `.npy` format for multi-channel data

---

## 🤖 Diffusion Models

### 6. **DiT** (`train_DIT/dataloader.py`)

**Dataset Class:** `ImageNetDataset`

**Expected S3 Structure:**
```
{root_dir}/
├── train_labels.txt  # Format: image_path class_id
├── val_labels.txt
└── images/
    ├── 0/  # Class directories
    ├── 1/
    └── ...
```

**Output Format:**
```python
{
    "image": [3, 256, 256],
    "label": int  # Class label (0-999 for ImageNet)
}
```

**Key Feature:** Class-conditional generation for ImageNet-style datasets

---

## 📚 Stable Diffusion Models

### 7. **Stage 1** (`train_stage_1/dataloader.py`)

**Dataset Class:** `Stage1Dataset`

**Expected S3 Structure:**
```
{root_dir}/
└── *.png, *.jpg, *.jpeg, *.bmp  # All images in prefix
```

**Output Format:**
```python
{
    "pixel_values": [3, 512, 512],
    "input_ids": [77]  # Generic caption: "a photo of a synthetic object"
}
```

**Key Feature:** Simple S3 listing, no pairs needed

---

### 8. **Stage 1-2** & **Stage 1-2-3** (`train_stage_1_2/dataloader.py`, `train_stage_1_2_3/dataloader.py`)

**Dataset Class:** `S3ImageDataset`

**Expected S3 Structure:**
```
{root_dir}/
└── *.png, *.jpg, *.jpeg, *.bmp
```

**Output Format:**
```python
{
    "pixel_values": [3, 512, 512],
    "input_ids": [77]
}
```

**Key Feature:** Same as Stage 1, simple image listing

---

### 9. **Mixture** (`train_mixture/dataloader.py`)

**Dataset Class:** `MixtureDataset`

**Expected S3 Structure:**
```
{root_dir}/
└── *.png, *.jpg, *.jpeg, *.bmp
```

**Output Format:**
```python
{
    "pixel_values": [3, 512, 512],
    "input_ids": [77]
}
```

**Key Feature:** Mixed dataset from multiple sources

---

## 🆕 Shared S3 VTON Dataset (NEW!)

### **S3VTONDataset** (`train/common/dataset.py`)

**Three Variants:**
1. **Easy:** 100% easy samples
2. **Medium:** 30% easy + 70% medium
3. **Hard:** 25% easy + 25% medium + 50% hard

**Expected S3 Structure:**
```
dataset_ultimate/
├── easy/
│   ├── female/
│   │   ├── initial_image/{stem}_person.png
│   │   ├── cloth_image/{stem}_cloth_*.png
│   │   └── try_on_image/{stem}_vton.png
│   └── male/ (same structure)
├── medium/ (same structure)
└── hard/ (same structure)
```

**Output Format:**
```python
{
    "initial_image": [3, H, W],   # Person image
    "cloth_image": [3, H, W],     # Cloth image
    "try_on_image": [3, H, W],    # Target try-on result
    "difficulty": str,            # "easy", "medium", or "hard"
    "stem": str                   # Base identifier
}
```

**Usage:**
```python
from train.common.dataset import get_vton_dataset

dataset = get_vton_dataset(
    difficulty='medium',  # or 'easy', 'hard'
    s3_prefixes=['dataset_ultimate/easy/', 'dataset_ultimate/medium/'],
    transform=transform
)
```

**Key Features:**
- Automatic triplet discovery
- Difficulty-based sampling
- No pairs file needed (auto-discovers by stem)

---

## 🔄 Migration Guide: Using S3VTONDataset

The new `S3VTONDataset` can replace existing VTON dataloaders. Here's how to adapt it:

### Example: Adapting for CATVTON

```python
# Current CATVTON dataloader expects:
# - person_img, garment_img, pose_map, segmentation

# S3VTONDataset provides:
# - initial_image (person), cloth_image (garment), try_on_image (target)

# Adapter:
from train.common.dataset import get_vton_dataset

dataset = get_vton_dataset(difficulty='medium', s3_prefixes=[...], transform=transform)

# In training loop, map keys:
for batch in dataloader:
    person_img = batch['initial_image']    # Map initial_image → person_img
    garment_img = batch['cloth_image']     # Map cloth_image → garment_img
    target_img = batch['try_on_image']     # Target for supervision
    
    # Note: pose_map and segmentation need to be generated or loaded separately
```

---

## 📋 Comparison: Pairs-based vs Auto-discovery

| Feature | Pairs-based (Current) | Auto-discovery (S3VTONDataset) |
|---------|----------------------|-------------------------------|
| **Setup** | Requires `train_pairs.txt` | No pairs file needed |
| **Discovery** | Manual pairing | Automatic by stem |
| **Flexibility** | Fixed pairs | Dynamic sampling |
| **Difficulty Mix** | Single difficulty | Easy/Medium/Hard mix |
| **Maintenance** | Update pairs file | Just add images |

---

## 🎯 Recommendations

### For VTON Training:
- **Use S3VTONDataset** for new projects (automatic discovery, difficulty mixing)
- **Keep existing dataloaders** for backward compatibility

### For Stable Diffusion:
- **Stage 1/2/3/Mixture** are simple and work well for general image training
- No need to change unless you need VTON-specific features

### For DiT:
- **ImageNetDataset** is specialized for class-conditional generation
- Keep as-is for ImageNet-style training

---

## 📝 Quick Reference

**VTON Models (need pairs or triplets):**
- CATVTON, IDM-VTON, CP-VTON, VTON-GAN, OOTDiffusion

**Simple Image Models (just list images):**
- Stage 1, Stage 1-2, Stage 1-2-3, Mixture

**Class-based Models:**
- DiT (ImageNet-style)

**New Shared Dataset:**
- S3VTONDataset (auto-discovery, difficulty mixing)

---

**Last Updated:** 2026-01-22
