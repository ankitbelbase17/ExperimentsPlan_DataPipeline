# S3 VTON Dataset - Difficulty-based Sampling

## Overview

This module provides a flexible S3-based dataset for Virtual Try-On (VTON) training with three difficulty variants that automatically mix samples from easy, medium, and hard directories.

## Dataset Variants

### 1. **Easy Dataset** (`S3VTONDatasetEasy`)
- **Composition:** 100% easy samples
- **Use case:** Initial training, baseline models, quick prototyping

### 2. **Medium Dataset** (`S3VTONDatasetMedium`)
- **Composition:** 30% easy + 70% medium samples
- **Use case:** Intermediate training, balanced difficulty

### 3. **Hard Dataset** (`S3VTONDatasetHard`)
- **Composition:** 25% easy + 25% medium + 50% hard samples
- **Use case:** Advanced training, challenging scenarios, final model refinement

## Expected S3 Structure

```
s3://your-bucket/
└── dataset_ultimate/
    ├── easy/
    │   ├── female/
    │   │   ├── initial_image/
    │   │   │   ├── 001_person.png
    │   │   │   ├── 002_person.png
    │   │   │   └── ...
    │   │   ├── cloth_image/
    │   │   │   ├── 001_cloth_dress.png
    │   │   │   ├── 002_cloth_shirt.png
    │   │   │   └── ...
    │   │   └── try_on_image/
    │   │       ├── 001_vton.png
    │   │       ├── 002_vton.png
    │   │       └── ...
    │   └── male/
    │       └── ... (same structure)
    ├── medium/
    │   └── ... (same structure)
    └── hard/
        └── ... (same structure)
```

## Naming Conventions

The dataset automatically discovers triplets based on these naming patterns:

| Image Type | Pattern | Example |
|------------|---------|---------|
| **Person (Initial)** | `{stem}_person.png` | `001_person.png` |
| **Cloth** | `{stem}_cloth_*.png` | `001_cloth_dress.png` |
| **Try-on (Target)** | `{stem}_vton.png` | `001_vton.png` |

The `stem` (e.g., `001`) is used to match triplets together.

## Usage

### Basic Usage

```python
from train.common.dataset import get_vton_dataset
from torchvision import transforms

# Define transforms
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Create Easy dataset (100% easy samples)
dataset_easy = get_vton_dataset(
    difficulty='easy',
    s3_prefixes=[
        'dataset_ultimate/easy/female/',
        'dataset_ultimate/easy/male/',
        'dataset_ultimate/medium/female/',
        'dataset_ultimate/medium/male/',
        'dataset_ultimate/hard/female/',
        'dataset_ultimate/hard/male/',
    ],
    transform=transform
)

# Create Medium dataset (30% easy + 70% medium)
dataset_medium = get_vton_dataset(
    difficulty='medium',
    s3_prefixes=[
        'dataset_ultimate/easy/female/',
        'dataset_ultimate/medium/female/',
        'dataset_ultimate/hard/female/',
    ],
    transform=transform
)

# Create Hard dataset (25% easy + 25% medium + 50% hard)
dataset_hard = get_vton_dataset(
    difficulty='hard',
    s3_prefixes=[
        'dataset_ultimate/easy/female/',
        'dataset_ultimate/medium/female/',
        'dataset_ultimate/hard/female/',
    ],
    transform=transform
)
```

### Using with DataLoader

```python
from torch.utils.data import DataLoader

# Create dataset
dataset = get_vton_dataset(
    difficulty='medium',
    s3_prefixes=['dataset_ultimate/easy/', 'dataset_ultimate/medium/'],
    transform=transform
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Training loop
for batch in dataloader:
    person_img = batch['initial_image']      # [B, 3, H, W]
    cloth_img = batch['cloth_image']         # [B, 3, H, W]
    target_img = batch['try_on_image']       # [B, 3, H, W]
    difficulty = batch['difficulty']         # List of strings
    stem = batch['stem']                     # List of strings
    
    # Your training code here
    ...
```

### Direct Class Usage

```python
from train.common.dataset import S3VTONDatasetEasy, S3VTONDatasetMedium, S3VTONDatasetHard

# Easy variant
dataset_easy = S3VTONDatasetEasy(
    s3_prefixes=['dataset_ultimate/easy/', 'dataset_ultimate/medium/'],
    transform=transform
)

# Medium variant
dataset_medium = S3VTONDatasetMedium(
    s3_prefixes=['dataset_ultimate/easy/', 'dataset_ultimate/medium/'],
    transform=transform
)

# Hard variant
dataset_hard = S3VTONDatasetHard(
    s3_prefixes=['dataset_ultimate/easy/', 'dataset_ultimate/medium/', 'dataset_ultimate/hard/'],
    transform=transform
)
```

## Features

### ✅ Automatic Triplet Discovery
- Scans S3 directories and automatically groups images by stem
- Only includes complete triplets (person + cloth + try-on)
- Warns about incomplete triplets

### ✅ Difficulty-based Sampling
- Automatically detects difficulty from S3 path (`/easy/`, `/medium/`, `/hard/`)
- Samples according to predefined weights
- Handles cases where target count exceeds available samples (samples with replacement)

### ✅ Flexible S3 Paths
- Supports both `s3://bucket/prefix/` and `prefix/` formats
- Auto-detects bucket from s3:// URI or uses default bucket

### ✅ Worker-safe
- Lazy S3 client initialization per worker
- Avoids fork safety issues in multi-worker dataloaders

### ✅ Robust Error Handling
- Gracefully handles missing images
- Logs warnings for incomplete triplets
- Provides detailed error messages

## Output Format

Each sample returned by the dataset contains:

```python
{
    "initial_image": torch.Tensor,    # [3, H, W] - Person image
    "cloth_image": torch.Tensor,      # [3, H, W] - Cloth image
    "try_on_image": torch.Tensor,     # [3, H, W] - Target try-on result
    "difficulty": str,                # "easy", "medium", or "hard"
    "stem": str                       # Base identifier (e.g., "001")
}
```

## Difficulty Distribution Examples

### Easy Dataset
```
Total: 10,000 samples
├── Easy: 10,000 (100%)
├── Medium: 0 (0%)
└── Hard: 0 (0%)
```

### Medium Dataset
```
Total: 10,000 samples
├── Easy: 3,000 (30%)
├── Medium: 7,000 (70%)
└── Hard: 0 (0%)
```

### Hard Dataset
```
Total: 10,000 samples
├── Easy: 2,500 (25%)
├── Medium: 2,500 (25%)
└── Hard: 5,000 (50%)
```

## Environment Variables

The dataset uses these environment variables (with fallbacks):

```bash
export S3_REGION="us-east-1"
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export S3_BUCKET_NAME="your-bucket-name"
```

Or configure in `src/config.py`:

```python
S3_REGION = "us-east-1"
AWS_ACCESS_KEY_ID = "your-access-key"
AWS_SECRET_ACCESS_KEY = "your-secret-key"
S3_BUCKET_NAME = "your-bucket-name"
```

## Advanced Usage

### Custom Difficulty Weights

```python
from train.common.dataset import S3VTONDataset

# Custom difficulty distribution: 10% easy, 40% medium, 50% hard
dataset = S3VTONDataset(
    s3_prefixes=['dataset_ultimate/easy/', 'dataset_ultimate/medium/', 'dataset_ultimate/hard/'],
    transform=transform,
    difficulty_weights={
        "easy": 0.10,
        "medium": 0.40,
        "hard": 0.50
    }
)
```

### Filtering by Gender

```python
# Female only - Easy difficulty
dataset_female_easy = get_vton_dataset(
    difficulty='easy',
    s3_prefixes=[
        'dataset_ultimate/easy/female/',
        'dataset_ultimate/medium/female/',
        'dataset_ultimate/hard/female/',
    ],
    transform=transform
)

# Male only - Hard difficulty
dataset_male_hard = get_vton_dataset(
    difficulty='hard',
    s3_prefixes=[
        'dataset_ultimate/easy/male/',
        'dataset_ultimate/medium/male/',
        'dataset_ultimate/hard/male/',
    ],
    transform=transform
)
```

## Integration with Training Scripts

### Example: CATVTON Training

```python
# In train/train_CATVTON/train.py
from train.common.dataset import get_vton_dataset
from torchvision import transforms

# Define transforms
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Create dataset with medium difficulty
dataset = get_vton_dataset(
    difficulty='medium',  # 30% easy + 70% medium
    s3_prefixes=[
        'dataset_ultimate/easy/female/',
        'dataset_ultimate/medium/female/',
    ],
    transform=transform
)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# Training loop
for batch in dataloader:
    person_img = batch['initial_image']
    garment_img = batch['cloth_image']
    target_img = batch['try_on_image']
    
    # CATVTON training code...
```

## Troubleshooting

### Issue: "No samples available for difficulty 'hard'"
**Solution:** Ensure your S3 prefixes include directories with `/hard/` in the path.

### Issue: "Incomplete triplet for stem 'XXX'"
**Solution:** Check that all three images exist for that stem:
- `XXX_person.png` in `initial_image/`
- `XXX_cloth_*.png` in `cloth_image/`
- `XXX_vton.png` in `try_on_image/`

### Issue: "Target count exceeds available samples"
**Solution:** This is a warning, not an error. The dataset will sample with replacement. Consider adding more samples or adjusting difficulty weights.

## Performance Tips

1. **Use multiple workers:** Set `num_workers=4` or higher in DataLoader
2. **Enable pin_memory:** Set `pin_memory=True` for faster GPU transfer
3. **Cache S3 credentials:** Use IAM roles instead of access keys when possible
4. **Batch prefetch:** Use `prefetch_factor=2` in DataLoader

## License

Part of the SyntheticDatasetExperiments project.
