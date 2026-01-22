# Bash Scripts Documentation

## 📂 Directory Structure

### 1. `curriculum/` (Recommended ⭐️)
Implements **Progressive Difficulty Training** (Curriculum Learning). Models start on easy data and gradually tackle harder samples.

- **`train_all.sh`**: The master script. Sequentially runs curriculum training for CATVTON, DiT, and VTON-GAN.
- **`train_catvton.sh`** (and others): Implements 3-stage training for specific models.

**Usage:**
```bash
cd train/bash_scripts/curriculum
bash train_all.sh
```

---

### 2. General Training (Root Directory)
Scripts in the root `bash_scripts/` folder allow you to train models on a specific difficulty level.

- **`train_all.sh [difficulty]`**: Trains ALL 6 models sequentially.
  - Default difficulty: `medium`
  - Usage: `bash train_all.sh hard`
- **`train_catvton.sh [difficulty]`**: Trains specific model.
  - Usage: `bash train_catvton.sh easy`

---

### 3. `easy/`, `medium/`, `hard/` (Shortcuts)
Convenience folders containing fixed-configuration scripts.
- **`train_all.sh`**: Trains supported models at that specific difficulty.

**Usage:**
```bash
cd train/bash_scripts/hard
bash train_all.sh
```

## ⚙️ Common Features
All scripts automatically handle:
- **S3 Checkpointing:** Uploads checkpoints to `s3://p1-to-ep1/checkpoints/...`
- **Resume:** Automatically resumes from `latest_checkpoint.pt`.
- **WandB Logging:** Logs metrics/images every 250 steps.
