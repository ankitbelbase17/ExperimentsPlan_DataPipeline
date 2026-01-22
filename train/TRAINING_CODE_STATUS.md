# Training Code Status & Required Updates

## ⚠️ Current Status

The **bash scripts** are ready and pass various arguments, but the **Python training scripts** currently **do NOT support command-line arguments**. They only read from `config.py` files.

---

## What the Bash Scripts Expect

### Arguments Being Passed
```bash
python train.py \
  --difficulty medium \              # ❌ Not supported
  --batch_size 4 \                   # ❌ Not supported
  --learning_rate 1e-5 \             # ❌ Not supported
  --max_train_steps 10000 \          # ❌ Not supported
  --num_epochs 50 \                  # ❌ Not supported
  --output_dir checkpoints_xxx \     # ❌ Not supported
  --wandb_project xxx \              # ❌ Not supported
  --wandb_run_name xxx \             # ❌ Not supported
  --resume_from_checkpoint xxx \     # ❌ Not supported
  --s3_prefixes "prefix1" "prefix2" \# ❌ Not supported
  --use_wandb                        # ❌ Not supported
```

### What Currently Works
```python
# Training scripts only read from config.py
import config

batch_size = config.BATCH_SIZE  # ✅ Works
learning_rate = config.LEARNING_RATE  # ✅ Works
num_epochs = config.NUM_EPOCHS  # ✅ Works
```

---

## Current Training Script Structure

### Example: `train_CATVTON/train.py`
```python
import config
from model import get_catvton_model
from dataloader import get_dataloader

def main():
    # No argparse - only uses config.py
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hardcoded from config
    wandb.init(project=config.WANDB_PROJECT)  # From config.py
    dataloader = get_dataloader(...)  # No difficulty argument
    
    # Training loop uses config values
    for epoch in range(config.NUM_EPOCHS):  # From config.py
        ...
```

**Issues:**
- ❌ No `argparse` for command-line arguments
- ❌ No `--difficulty` support for dataloader
- ❌ No `--max_train_steps` support
- ❌ No `--resume_from_checkpoint` with custom path
- ❌ No dynamic S3 prefix configuration

---

## Required Updates

### Option 1: Add Argparse to All Training Scripts (Recommended)

Each `train.py` needs to be updated to support command-line arguments:

```python
import argparse
import config

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Difficulty for curriculum learning
    parser.add_argument("--difficulty", type=str, default="medium",
                       choices=["easy", "medium", "hard"])
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--num_epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--max_train_steps", type=int, default=None)
    
    # Paths
    parser.add_argument("--output_dir", type=str, default=config.OUTPUT_DIR)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    
    # WandB
    parser.add_argument("--wandb_project", type=str, default=config.WANDB_PROJECT)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    
    # S3 prefixes for dataloader
    parser.add_argument("--s3_prefixes", nargs="+", default=None)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Use args instead of config
    dataloader = get_dataloader(
        tokenizer=...,
        difficulty=args.difficulty,  # NEW!
        batch_size=args.batch_size,
        s3_prefixes=args.s3_prefixes
    )
    
    # Training loop with max_train_steps support
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(dataloader):
            # Training code...
            
            # Support max_train_steps
            if args.max_train_steps and global_step >= args.max_train_steps:
                break
```

---

### Option 2: Modify Bash Scripts to Use Config Files Only

Simplify bash scripts to just run training without arguments:

```bash
#!/bin/bash
# Simplified - no arguments

cd ../../train_CATVTON
python train.py  # Uses config.py only
```

**Drawbacks:**
- ❌ No curriculum learning (can't switch difficulty mid-training)
- ❌ No flexible configuration
- ❌ Must manually edit config.py for each run

---

## What Needs to Be Updated

### Files Requiring Updates (if using Option 1)

1. **`train_CATVTON/train.py`** - Add argparse
2. **`train_IDMVTON/train.py`** - Add argparse
3. **`train_CP_VTON/train.py`** - Add argparse
4. **`train_VTON_GAN/train.py`** - Add argparse
5. **`train_OOTDiffusion/train.py`** - Add argparse
6. **`train_DIT/train.py`** - Add argparse

### Dataloader Updates Required

All `get_dataloader()` functions need to accept:
```python
def get_dataloader(
    tokenizer,
    batch_size=None,
    split='train',
    difficulty='medium',  # NEW!
    s3_prefixes=None      # NEW!
):
    # Use S3VTONDataset with difficulty
    from train.common.dataset import get_vton_dataset
    
    dataset = get_vton_dataset(
        difficulty=difficulty,
        s3_prefixes=s3_prefixes or default_prefixes,
        transform=transform
    )
    
    return DataLoader(dataset, batch_size=batch_size, ...)
```

---

## Curriculum Learning Requirements

For curriculum learning to work, training scripts need:

### 1. Max Steps Support
```python
if args.max_train_steps:
    if global_step >= args.max_train_steps:
        print(f"Reached max steps: {args.max_train_steps}")
        break
```

### 2. Resume from Specific Checkpoint
```python
if args.resume_from_checkpoint:
    checkpoint = torch.load(args.resume_from_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    global_step = checkpoint['global_step']
```

### 3. Dynamic Dataloader Creation
```python
# Each stage creates new dataloader with different difficulty
dataloader = get_dataloader(
    difficulty=args.difficulty,  # Changes per stage
    s3_prefixes=args.s3_prefixes
)
```

---

## Current Workaround

Until training scripts are updated, you can:

### 1. Manually Edit Config Files

For each difficulty level, edit `config.py`:

```python
# train_CATVTON/config.py

# For easy training
DATASET_ROOT = "dataset_ultimate/easy/"
OUTPUT_DIR = "checkpoints_catvton_easy"

# For medium training (change manually)
DATASET_ROOT = "dataset_ultimate/medium/"
OUTPUT_DIR = "checkpoints_catvton_medium"
```

### 2. Run Training Directly

```bash
cd train_CATVTON
python train.py  # Uses config.py settings
```

### 3. No Curriculum Learning

Curriculum learning **requires** argparse support to switch dataloaders mid-training.

---

## Recommended Action Plan

### Phase 1: Add Basic Argparse Support
Update all 6 training scripts to accept:
- `--difficulty`
- `--batch_size`
- `--learning_rate`
- `--num_epochs`
- `--output_dir`

### Phase 2: Add Curriculum Learning Support
Add to all training scripts:
- `--max_train_steps`
- `--resume_from_checkpoint`
- `--wandb_run_name`

### Phase 3: Update Dataloaders
Modify all `get_dataloader()` functions to:
- Accept `difficulty` parameter
- Accept `s3_prefixes` parameter
- Use `S3VTONDataset` from `train/common/dataset.py`

---

## Summary

| Feature | Bash Scripts | Python Scripts | Status |
|---------|-------------|----------------|--------|
| **Difficulty selection** | ✅ Pass `--difficulty` | ❌ Not supported | **Needs update** |
| **Max steps** | ✅ Pass `--max_train_steps` | ❌ Not supported | **Needs update** |
| **Custom checkpoints** | ✅ Pass `--resume_from_checkpoint` | ⚠️ Partial (uses latest only) | **Needs update** |
| **WandB config** | ✅ Pass `--wandb_project` | ⚠️ Uses config.py | **Needs update** |
| **S3 prefixes** | ✅ Pass `--s3_prefixes` | ❌ Not supported | **Needs update** |
| **Curriculum learning** | ✅ Scripts ready | ❌ **Not possible** | **Needs update** |

---

## Conclusion

**Current State:**
- ✅ Bash scripts are well-structured and ready
- ✅ S3VTONDataset supports difficulty-based sampling
- ❌ Python training scripts **do not support** command-line arguments
- ❌ Curriculum learning **cannot work** without updates

**To Make Everything Work:**
1. Add `argparse` to all 6 training scripts
2. Update dataloaders to accept `difficulty` and `s3_prefixes`
3. Add `max_train_steps` support for curriculum learning
4. Add custom checkpoint resume support

**Estimated Work:**
- ~2-3 hours per training script
- ~12-18 hours total for all 6 models

---

**Would you like me to update the training scripts to support all these features?**

---

**Last Updated:** 2026-01-22
