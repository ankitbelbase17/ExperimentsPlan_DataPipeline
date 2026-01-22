# Training Scripts Update Progress

## ✅ Update Complete!

All 6 models have been updated to support `argparse`, `difficulty`, and `curriculum learning`.

### Summary of Updates

| Model | Argparse | Curriculum | Difficulty | S3 Dataset |
|-------|----------|------------|------------|------------|
| **CATVTON** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **IDM-VTON** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **CP-VTON** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **VTON-GAN** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **OOTDiffusion** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **DiT** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |

---

## 🚀 How to Run

### 1. Curriculum Learning (Best Quality)
Progressive training from easy to hard difficulty.

```bash
cd train/bash_scripts/curriculum
bash train_all.sh
```

### 2. Fixed Difficulty Training
Train on a specific difficulty level.

```bash
# Train CATVTON on medium difficulty
cd train_CATVTON
python train.py --difficulty medium --batch_size 4

# Or use bash script
cd train/bash_scripts/medium
bash train_catvton.sh
```

### 3. Mixed Training
Train all models with custom settings.

```bash
cd train/bash_scripts/mix
bash train_all.sh hard
```

---

## 🔍 Features Implemented

### 1. Command Line Arguments (All Models)
- `--difficulty`: Select dataset difficulty (easy/medium/hard)
- `--batch_size`: Override config batch size
- `--learning_rate`: Override config learning rate
- `--num_epochs`: Override config epochs
- `--max_train_steps`: Stop training after N steps (for curriculum stages)
- `--resume_from_checkpoint`: Resume from specific checkpoint
- `--output_dir`: Custom output directory
- `--wandb_run_name`: Custom WandB run name
- `--use_wandb`: Enable/disable logging
- `--s3_prefixes`: Custom S3 prefixes for dataset

### 2. Dataloader Logic
- Automatically maps difficulty to appropriate S3 prefixes
- Uses `S3VTONDataset` from `train/common/dataset.py`
- Fallback to original dataset if `S3VTONDataset` fails
- Proper image transforms for each model architecture

### 3. Checkpointing
- `save_checkpoint`: Supports custom paths and filenames
- `load_checkpoint`: Can load from any specific path (crucial for curriculum learning)
- `load_latest_checkpoint`: Backward compatible

---

**Everything is ready for training!** 🚀
