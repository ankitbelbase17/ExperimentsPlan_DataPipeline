# Bash Scripts Training Guide

Complete explanation of what each bash script trains, with dataset compositions and use cases.

---

## Directory Overview

```
bash_scripts/
├── easy/           # 100% easy samples
├── medium/         # 30% easy + 70% medium samples
├── hard/           # 25% easy + 25% medium + 50% hard samples
├── mix/            # All models, configurable difficulty
└── curriculum/     # Progressive difficulty (Easy→Medium→Hard)
```

---

## 1. Easy Directory (`bash_scripts/easy/`)

### Dataset Composition
- **Easy:** 100%
- **Medium:** 0%
- **Hard:** 0%

### Scripts

#### `train_catvton.sh`
**Trains:** CATVTON (Concatenation-based Attentive VTON)  
**Dataset:** 100% easy samples  
**Duration:** 50 epochs  
**Batch Size:** 4  
**Learning Rate:** 1e-5  
**Output:** `checkpoints_catvton_easy/`  
**Use Case:** Build foundation for CATVTON on simple try-on cases

#### `train_dit.sh`
**Trains:** DiT (Diffusion Transformer)  
**Dataset:** 100% easy samples  
**Duration:** 400 epochs  
**Batch Size:** 256  
**Learning Rate:** 1e-4  
**Output:** `checkpoints_dit_easy/`  
**Use Case:** Train class-conditional diffusion transformer on easy samples

#### `train_vtongan.sh`
**Trains:** VTON-GAN (GAN-based Virtual Try-On)  
**Dataset:** 100% easy samples  
**Duration:** 50 epochs  
**Batch Size:** 4  
**Learning Rate:** 2e-4  
**Output:** `checkpoints_vtongan_easy/`  
**Use Case:** Train adversarial VTON model on simple cases

#### `train_all.sh`
**Trains:** All 3 models sequentially (CATVTON → DiT → VTON-GAN)  
**Dataset:** 100% easy samples for all  
**Use Case:** Complete training pipeline on easy difficulty

---

## 2. Medium Directory (`bash_scripts/medium/`)

### Dataset Composition
- **Easy:** 30%
- **Medium:** 70%
- **Hard:** 0%

### Scripts

#### `train_catvton.sh`
**Trains:** CATVTON  
**Dataset:** 30% easy + 70% medium  
**Duration:** 50 epochs  
**Output:** `checkpoints_catvton_medium/`  
**Use Case:** Train CATVTON on moderately challenging try-on scenarios

#### `train_dit.sh`
**Trains:** DiT  
**Dataset:** 30% easy + 70% medium  
**Duration:** 400 epochs  
**Output:** `checkpoints_dit_medium/`  
**Use Case:** Train DiT with balanced difficulty

#### `train_vtongan.sh`
**Trains:** VTON-GAN  
**Dataset:** 30% easy + 70% medium  
**Duration:** 50 epochs  
**Output:** `checkpoints_vtongan_medium/`  
**Use Case:** Train GAN on moderately difficult cases

#### `train_all.sh`
**Trains:** All 3 models sequentially  
**Dataset:** 30% easy + 70% medium for all  
**Use Case:** Intermediate difficulty training for all models

---

## 3. Hard Directory (`bash_scripts/hard/`)

### Dataset Composition
- **Easy:** 25%
- **Medium:** 25%
- **Hard:** 50%

### Scripts

#### `train_catvton.sh`
**Trains:** CATVTON  
**Dataset:** 25% easy + 25% medium + 50% hard  
**Duration:** 50 epochs  
**Output:** `checkpoints_catvton_hard/`  
**Use Case:** Train CATVTON on challenging try-on scenarios (complex poses, occlusions)

#### `train_dit.sh`
**Trains:** DiT  
**Dataset:** 25% easy + 25% medium + 50% hard  
**Duration:** 400 epochs  
**Output:** `checkpoints_dit_hard/`  
**Use Case:** Train DiT on difficult generation tasks

#### `train_vtongan.sh`
**Trains:** VTON-GAN  
**Dataset:** 25% easy + 25% medium + 50% hard  
**Duration:** 50 epochs  
**Output:** `checkpoints_vtongan_hard/`  
**Use Case:** Train GAN on hardest cases

#### `train_all.sh`
**Trains:** All 3 models sequentially  
**Dataset:** 25% easy + 25% medium + 50% hard for all  
**Use Case:** Final refinement training on challenging samples

---

## 4. Mix Directory (`bash_scripts/mix/`)

### Dataset Composition
**Configurable** - Pass difficulty as argument (easy, medium, or hard)

### Scripts

#### `train_catvton.sh [difficulty]`
**Trains:** CATVTON  
**Dataset:** Based on difficulty argument  
**Example:** `bash train_catvton.sh medium`  
**Output:** `checkpoints_catvton_${difficulty}/`  
**Use Case:** Flexible CATVTON training with any difficulty

#### `train_idmvton.sh [difficulty]`
**Trains:** IDM-VTON (Improving Diffusion Models for VTON)  
**Base Model:** SD2 Base  
**Dataset:** Based on difficulty argument  
**Batch Size:** 4  
**Learning Rate:** 1e-5  
**Output:** `checkpoints_idmvton_${difficulty}/`  
**Use Case:** Train SD2-based diffusion VTON with CLIP garment encoder

#### `train_cpvton.sh [difficulty]`
**Trains:** CP-VTON (Characteristic-Preserving VTON)  
**Architecture:** Two-stage (GMM + TOM)  
**Dataset:** Based on difficulty argument  
**Batch Size:** 4  
**Learning Rate:** 1e-4  
**Output:** `checkpoints_cpvton_${difficulty}/`  
**Use Case:** Train two-stage VTON with geometric matching

#### `train_vtongan.sh [difficulty]`
**Trains:** VTON-GAN  
**Dataset:** Based on difficulty argument  
**Output:** `checkpoints_vtongan_${difficulty}/`  
**Use Case:** Flexible GAN training

#### `train_ootdiffusion.sh [difficulty]`
**Trains:** OOTDiffusion (Outfitting Fusion based Latent Diffusion)  
**Base Model:** SD2 Base  
**Resolution:** 768×1024 (high-res!)  
**Dataset:** Based on difficulty argument  
**Batch Size:** 2 (due to high resolution)  
**Learning Rate:** 1e-5  
**Output:** `checkpoints_ootdiffusion_${difficulty}/`  
**Use Case:** Train high-resolution VTON with fusion blocks

#### `train_dit.sh [difficulty]`
**Trains:** DiT  
**Dataset:** Based on difficulty argument  
**Output:** `checkpoints_dit_${difficulty}/`  
**Use Case:** Flexible DiT training

#### `train_all.sh [difficulty]`
**Trains:** All 6 models sequentially:
1. CATVTON
2. IDM-VTON
3. CP-VTON
4. VTON-GAN
5. OOTDiffusion
6. DiT

**Dataset:** Based on difficulty argument (default: medium)  
**Example:** `bash train_all.sh hard`  
**Use Case:** Complete training of all VTON models with chosen difficulty

---

## 5. Curriculum Directory (`bash_scripts/curriculum/`) ⭐

### Training Strategy
**Progressive Difficulty:** Easy → Medium → Hard  
**Automatic dataloader switching** at predefined steps/epochs

### Scripts

#### `train_catvton.sh`
**Trains:** CATVTON with curriculum learning  

**Stage 1 (Steps 0-10,000):**
- Dataset: 100% easy
- Learning Rate: 1e-5
- Focus: Build foundation

**Stage 2 (Steps 10,000-25,000):**
- Dataset: 30% easy + 70% medium
- Learning Rate: 5e-6 (reduced)
- Focus: Increase difficulty

**Stage 3 (Steps 25,000-40,000):**
- Dataset: 25% easy + 25% medium + 50% hard
- Learning Rate: 2e-6 (further reduced)
- Focus: Final refinement

**Total:** 40,000 steps  
**Output:** `checkpoints_catvton_curriculum/`  
**Use Case:** Optimal CATVTON training with progressive difficulty

---

#### `train_dit.sh`
**Trains:** DiT with curriculum learning  

**Stage 1 (Epochs 0-100):**
- Dataset: 100% easy
- Learning Rate: 1e-4
- Focus: Quick initial learning

**Stage 2 (Epochs 100-250):**
- Dataset: 30% easy + 70% medium
- Learning Rate: 5e-5
- Focus: Balanced training

**Stage 3 (Epochs 250-400):**
- Dataset: 25% easy + 25% medium + 50% hard
- Learning Rate: 2e-5
- Focus: Hard sample refinement

**Total:** 400 epochs  
**Output:** `checkpoints_dit_curriculum/`  
**Use Case:** Optimal DiT training with progressive difficulty

---

#### `train_vtongan.sh`
**Trains:** VTON-GAN with curriculum learning  

**Stage 1 (Steps 0-8,000):**
- Dataset: 100% easy
- Learning Rate: 2e-4
- Focus: Stable GAN initialization

**Stage 2 (Steps 8,000-20,000):**
- Dataset: 30% easy + 70% medium
- Learning Rate: 1e-4
- Focus: Adversarial refinement

**Stage 3 (Steps 20,000-35,000):**
- Dataset: 25% easy + 25% medium + 50% hard
- Learning Rate: 5e-5
- Focus: Final quality improvement

**Total:** 35,000 steps  
**Output:** `checkpoints_vtongan_curriculum/`  
**Use Case:** Optimal GAN training with progressive difficulty

---

#### `train_all.sh`
**Trains:** All 3 models with curriculum learning  
**Sequence:**
1. CATVTON (40k steps, 3 stages)
2. DiT (400 epochs, 3 stages)
3. VTON-GAN (35k steps, 3 stages)

**Use Case:** Complete curriculum learning pipeline for all models

---

## Model Comparison

| Model | Architecture | Base Model | Resolution | Specialty |
|-------|-------------|-----------|------------|-----------|
| **CATVTON** | Diffusion + Concatenation | SD v1.5 | 512×512 | Multi-modal fusion |
| **IDM-VTON** | Diffusion + CLIP | SD2 Base | 512×512 | Garment understanding |
| **CP-VTON** | Two-stage (GMM+TOM) | Custom | 256×256 | Geometric matching |
| **VTON-GAN** | GAN | Custom | 256×256 | Adversarial training |
| **OOTDiffusion** | Diffusion + Fusion | SD2 Base | 768×1024 | High-resolution |
| **DiT** | Transformer | Custom | 256×256 | Class-conditional |

---

## Training Time Estimates (Single A100 GPU)

### Easy/Medium/Hard Directories
| Model | Time per Run |
|-------|-------------|
| CATVTON | ~12 hours |
| DiT | ~40 hours |
| VTON-GAN | ~10 hours |
| **Total (train_all.sh)** | **~62 hours** |

### Mix Directory (All 6 Models)
| Model | Time per Run |
|-------|-------------|
| CATVTON | ~12 hours |
| IDM-VTON | ~14 hours |
| CP-VTON | ~8 hours |
| VTON-GAN | ~10 hours |
| OOTDiffusion | ~18 hours (high-res) |
| DiT | ~40 hours |
| **Total (train_all.sh)** | **~102 hours** |

### Curriculum Directory
| Model | Total Time (3 Stages) |
|-------|---------------------|
| CATVTON | ~12 hours |
| DiT | ~40 hours |
| VTON-GAN | ~10 hours |
| **Total (train_all.sh)** | **~62 hours** |

---

## Use Case Recommendations

### For Quick Prototyping
```bash
cd bash_scripts/easy
bash train_catvton.sh  # Single model, easy samples
```

### For Best Quality (Recommended ⭐)
```bash
cd bash_scripts/curriculum
bash train_all.sh  # Progressive difficulty, all models
```

### For Specific Difficulty
```bash
cd bash_scripts/medium
bash train_all.sh  # All models on medium difficulty
```

### For All Models, Flexible Difficulty
```bash
cd bash_scripts/mix
bash train_all.sh hard  # All 6 models on hard difficulty
```

### For Single Model, Custom Difficulty
```bash
cd bash_scripts/mix
bash train_ootdiffusion.sh medium  # Just OOTDiffusion on medium
```

---

## Summary Table

| Directory | Models | Difficulty | Dataloader Switching | Best For |
|-----------|--------|------------|---------------------|----------|
| **easy/** | 3 | Fixed (100% easy) | No | Quick prototyping |
| **medium/** | 3 | Fixed (30%+70%) | No | Balanced training |
| **hard/** | 3 | Fixed (25%+25%+50%) | No | Challenge testing |
| **mix/** | 6 | Configurable | No | Flexibility |
| **curriculum/** | 3 | Progressive | **Yes** ✅ | **Best quality** ⭐ |

---

## Quick Reference

### Train Everything (Best Quality)
```bash
cd bash_scripts/curriculum && bash train_all.sh
```

### Train All Models on Medium Difficulty
```bash
cd bash_scripts/medium && bash train_all.sh
```

### Train Specific Model with Curriculum
```bash
cd bash_scripts/curriculum && bash train_catvton.sh
```

### Train All 6 Models on Hard Difficulty
```bash
cd bash_scripts/mix && bash train_all.sh hard
```

---

**Total Scripts:** 28 bash files  
**Total Models:** 6 unique VTON/diffusion models  
**Total Training Strategies:** 5 (easy, medium, hard, mix, curriculum)

---

**Last Updated:** 2026-01-22
