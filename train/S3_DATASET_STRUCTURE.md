# S3 VTON Dataset Structure

The `S3VTONDataset` expects a specific directory structure and file naming convention to automatically pair person images, garments, and target try-on results.

## 📂 Directory Structure

For every S3 prefix you provide (e.g., `dataset_ultimate/easy/female/`), the dataloader expects the following three subdirectories:

```
dataset_ultimate/easy/female/
│
├── initial_image/      # Original person images
│   ├── {stem}_person.png
│   └── ...
│
├── cloth_image/        # Garment/Cloth images
│   ├── {stem}_cloth_{variant}.png
│   └── ...
│
└── try_on_image/       # Ground truth target images (VTON result)
    ├── {stem}_vton.png
    └── ...
```

---

## 🏷️ Naming Convention (The "Stem")

The dataset automatically links images based on a unique **stem** extracted from the filename. All three files for a single training sample must share the exact same stem.

| Image Type | Folder Name | Required Suffix/Pattern | Example Filename | Extracted Stem |
|------------|-------------|-------------------------|------------------|----------------|
| **Person** | `initial_image` | `_person.png` | `01234_person.png` | `01234` |
| **Garment** | `cloth_image` | `_cloth_...` | `01234_cloth_front.png` | `01234` |
| **Target** | `try_on_image` | `_vton.png` | `01234_vton.png` | `01234` |

> **Note:** Supported extensions are `.png`, `.jpg`, and `.jpeg`.

---

## ❌ Common Issues

1. **Incomplete Triplets:** If a stem is found in `initial_image` but missing in `cloth_image`, that sample is **skipped**. You must have all three components.
2. **Incorrect Suffix:** If a file is named `01234.png` instead of `01234_person.png`, it will be ignored.
3. **Mismatched Stems:** `01234_person.png` and `01235_cloth_front.png` will **not** be paired.

## 🛠️ Example Layout

```
s3://your-bucket/dataset_ultimate/easy/female/
├── initial_image/
│   ├── 00001_person.png
│   ├── 00002_person.png
├── cloth_image/
│   ├── 00001_cloth_front.png
│   ├── 00002_cloth_flat.png
└── try_on_image/
    ├── 00001_vton.png
    ├── 00002_vton.png
```

This folder will yield **2 training samples** (for stems `00001` and `00002`).
