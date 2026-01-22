# Curriculum Learning Training Scripts

Progressive difficulty training for VTON models: **Easy → Medium → Hard**

---

## Overview

Curriculum learning trains models by gradually increasing difficulty, leading to:
- ✅ **Faster convergence** - Easy samples provide quick initial learning
- ✅ **Better generalization** - Progressive difficulty prevents overfitting
- ✅ **Higher final quality** - Hard samples refine the model
- ✅ **More stable training** - Gradual difficulty increase reduces instability

---

## Training Strategy

### CATVTON Curriculum
```
Stage 1: Easy      →  Steps 0-10,000   (10k steps, LR: 1e-5)
Stage 2: Medium    →  Steps 10,000-25,000  (15k steps, LR: 5e-6)
Stage 3: Hard      →  Steps 25,000-40,000  (15k steps, LR: 2e-6)
Total: 40,000 steps
```

### DiT Curriculum
```
Stage 1: Easy      →  Epochs 0-100     (100 epochs, LR: 1e-4)
Stage 2: Medium    →  Epochs 100-250   (150 epochs, LR: 5e-5)
Stage 3: Hard      →  Epochs 250-400   (150 epochs, LR: 2e-5)
Total: 400 epochs
```

### VTON-GAN Curriculum
```
Stage 1: Easy      →  Steps 0-8,000    (8k steps, LR: 2e-4)
Stage 2: Medium    →  Steps 8,000-20,000  (12k steps, LR: 1e-4)
Stage 3: Hard      →  Steps 20,000-35,000  (15k steps, LR: 5e-5)
Total: 35,000 steps
```

---

## Dataset Composition by Stage

### Stage 1: Easy
- **Easy:** 100%
- **Medium:** 0%
- **Hard:** 0%

### Stage 2: Medium
- **Easy:** 30%
- **Medium:** 70%
- **Hard:** 0%

### Stage 3: Hard
- **Easy:** 25%
- **Medium:** 25%
- **Hard:** 50%

---

## Usage

### Train All Models with Curriculum Learning

```bash
cd train/bash_scripts/curriculum
bash train_all.sh
```

This will sequentially train:
1. CATVTON (40k steps)
2. DiT (400 epochs)
3. VTON-GAN (35k steps)

Each model progresses through Easy → Medium → Hard automatically.

---

### Train Individual Models

#### CATVTON
```bash
cd train/bash_scripts/curriculum
bash train_catvton.sh
```

**Timeline:**
- **0-10k steps:** Easy samples only
- **10k-25k steps:** 30% easy + 70% medium
- **25k-40k steps:** 25% easy + 25% medium + 50% hard

#### DiT
```bash
cd train/bash_scripts/curriculum
bash train_dit.sh
```

**Timeline:**
- **0-100 epochs:** Easy samples only
- **100-250 epochs:** 30% easy + 70% medium
- **250-400 epochs:** 25% easy + 25% medium + 50% hard

#### VTON-GAN
```bash
cd train/bash_scripts/curriculum
bash train_vtongan.sh
```

**Timeline:**
- **0-8k steps:** Easy samples only
- **8k-20k steps:** 30% easy + 70% medium
- **20k-35k steps:** 25% easy + 25% medium + 50% hard

---

## How It Works

### Automatic Dataloader Switching

Each stage automatically switches to the appropriate difficulty dataloader:

```python
# Stage 1: Easy
dataset = get_vton_dataset(difficulty='easy', s3_prefixes=[...])

# Stage 2: Medium (script automatically resumes and switches)
dataset = get_vton_dataset(difficulty='medium', s3_prefixes=[...])

# Stage 3: Hard (script automatically resumes and switches)
dataset = get_vton_dataset(difficulty='hard', s3_prefixes=[...])
```

### Checkpoint Continuity

All stages share the same checkpoint directory, ensuring smooth progression:

```
checkpoints_catvton_curriculum/
├── latest_checkpoint.pt  # Always contains latest state
├── checkpoint_epoch_X_step_Y.pt
└── ...
```

The `--resume_from_checkpoint` flag ensures each stage continues from where the previous stage left off.

---

## Learning Rate Schedule

Learning rates decrease as difficulty increases:

| Model | Easy LR | Medium LR | Hard LR |
|-------|---------|-----------|---------|
| **CATVTON** | 1e-5 | 5e-6 | 2e-6 |
| **DiT** | 1e-4 | 5e-5 | 2e-5 |
| **VTON-GAN** | 2e-4 | 1e-4 | 5e-5 |

**Rationale:** Lower LR for harder samples prevents catastrophic forgetting and allows fine-tuning.

---

## WandB Tracking

Each stage is tracked separately in WandB:

### CATVTON
- Run 1: `catvton-curriculum-stage1-easy`
- Run 2: `catvton-curriculum-stage2-medium`
- Run 3: `catvton-curriculum-stage3-hard`
- Project: `catvton-curriculum`

### DiT
- Run 1: `dit-curriculum-stage1-easy`
- Run 2: `dit-curriculum-stage2-medium`
- Run 3: `dit-curriculum-stage3-hard`
- Project: `dit-curriculum`

### VTON-GAN
- Run 1: `vtongan-curriculum-stage1-easy`
- Run 2: `vtongan-curriculum-stage2-medium`
- Run 3: `vtongan-curriculum-stage3-hard`
- Project: `vtongan-curriculum`

---

## Advantages Over Fixed Difficulty

| Aspect | Fixed Difficulty | Curriculum Learning |
|--------|------------------|---------------------|
| **Convergence Speed** | Slower | **Faster** ✅ |
| **Training Stability** | Less stable | **More stable** ✅ |
| **Final Quality** | Good | **Better** ✅ |
| **Generalization** | Moderate | **Strong** ✅ |
| **Sample Efficiency** | Lower | **Higher** ✅ |

---

## Monitoring Progress

### Check Current Stage

```bash
# Check CATVTON progress
tail -f checkpoints_catvton_curriculum/training.log

# Check DiT progress
tail -f checkpoints_dit_curriculum/training.log

# Check VTON-GAN progress
tail -f checkpoints_vtongan_curriculum/training.log
```

### WandB Dashboard

Monitor all stages in real-time:
```
https://wandb.ai/your-entity/catvton-curriculum
https://wandb.ai/your-entity/dit-curriculum
https://wandb.ai/your-entity/vtongan-curriculum
```

---

## Customization

### Adjust Stage Durations

Edit the `--max_train_steps` or `--num_epochs` in each script:

```bash
# Example: Longer easy stage for CATVTON
python train.py \
  --difficulty easy \
  --max_train_steps 15000  # Changed from 10000
  ...
```

### Adjust Learning Rates

Modify the `--learning_rate` for each stage:

```bash
# Example: Higher LR for medium stage
python train.py \
  --difficulty medium \
  --learning_rate 8e-6  # Changed from 5e-6
  ...
```

### Add More Stages

You can add intermediate stages:

```bash
# Example: Add "very_easy" stage
python train.py \
  --difficulty very_easy \
  --max_train_steps 5000 \
  ...
```

---

## Troubleshooting

### Issue: Training doesn't resume between stages
**Solution:** Ensure `--resume_from_checkpoint` points to the correct path:
```bash
--resume_from_checkpoint checkpoints_catvton_curriculum/latest_checkpoint.pt
```

### Issue: Dataloader doesn't switch
**Solution:** Verify that the `--difficulty` argument is being passed correctly to the training script.

### Issue: WandB runs not linking
**Solution:** Use the same `--wandb_project` across all stages but different `--wandb_run_name`.

---

## Expected Training Time

Assuming single GPU (A100):

| Model | Easy Stage | Medium Stage | Hard Stage | Total |
|-------|-----------|--------------|------------|-------|
| **CATVTON** | ~3 hours | ~4.5 hours | ~4.5 hours | **~12 hours** |
| **DiT** | ~10 hours | ~15 hours | ~15 hours | **~40 hours** |
| **VTON-GAN** | ~2.5 hours | ~3.5 hours | ~4.5 hours | **~10.5 hours** |

**Total for all models:** ~62.5 hours (~2.6 days)

---

## Best Practices

1. **Monitor loss curves** - Ensure loss is decreasing in each stage
2. **Check samples** - Visually inspect generated samples at stage transitions
3. **Save stage checkpoints** - Keep checkpoints from each stage for comparison
4. **Use WandB** - Track metrics across all stages in one dashboard
5. **Adjust if needed** - If a stage converges early, you can move to the next stage

---

## Summary

**Curriculum learning** provides a structured approach to training VTON models:

```
Easy (Foundation) → Medium (Refinement) → Hard (Mastery)
```

This progressive strategy leads to:
- ✅ Faster convergence
- ✅ Better final quality
- ✅ More stable training
- ✅ Stronger generalization

**Start your curriculum learning:**
```bash
cd train/bash_scripts/curriculum
bash train_all.sh
```

---

**Last Updated:** 2026-01-22
