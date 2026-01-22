# Benchmark Evaluation Scripts

This directory contains scripts to evaluate all VTON models on standard benchmark datasets.

## 📊 Available Benchmarks

### 1. VITON-HD
- **Script**: `metrics_vitonhd.sh`
- **Dataset Location**: `baselines/viton-hd/test/`
- **Description**: High-resolution (1024×768) virtual try-on benchmark
- **Paper**: VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization

### 2. DressCode
- **Script**: `metrics_dresscode.sh`
- **Dataset Location**: `baselines/dresscode/test/`
- **Description**: Multi-category garment transfer benchmark (upper-body, lower-body, dresses)
- **Paper**: DressCode: High-Resolution Multi-Category Virtual Try-On

### 3. DeepFashion
- **Script**: `metrics_deepfashion.sh`
- **Dataset Location**: `baselines/deepfashion/test/`
- **Description**: Large-scale fashion dataset for in-shop clothes retrieval
- **Paper**: DeepFashion: Powering Robust Clothes Recognition and Retrieval

### 4. Our Custom Test Set
- **Script**: `metrics_ours_test.sh`
- **Dataset Location**: `dataset_ultimate/test/`
- **Description**: Our curated test set with diverse difficulty levels

## 🚀 Quick Start

### Run Single Benchmark
```bash
cd bash_scripts

# VITON-HD
bash metrics_vitonhd.sh

# DressCode
bash metrics_dresscode.sh

# DeepFashion
bash metrics_deepfashion.sh

# Our test set
bash metrics_ours_test.sh
```

### Run All Benchmarks
```bash
cd bash_scripts
bash metrics_all_benchmarks.sh
```

This will:
1. Evaluate all models on all 4 benchmarks
2. Generate individual results for each benchmark
3. Create a combined summary comparing all benchmarks

## 📁 Expected Dataset Structure

Each benchmark dataset should follow this structure:

```
baselines/
├── viton-hd/
│   └── test/
│       ├── initial_image/
│       ├── cloth_image/
│       └── try_on_image/
├── dresscode/
│   └── test/
│       ├── initial_image/
│       ├── cloth_image/
│       └── try_on_image/
└── deepfashion/
    └── test/
        ├── initial_image/
        ├── cloth_image/
        └── try_on_image/

dataset_ultimate/
└── test/
    ├── initial_image/
    ├── cloth_image/
    └── try_on_image/
```

## 📈 Metrics Computed

For each benchmark, the following metrics are computed:

### Image Quality
- **LPIPS** (Learned Perceptual Image Patch Similarity) ↓ lower is better
- **SSIM** (Structural Similarity Index) ↑ higher is better
- **PSNR** (Peak Signal-to-Noise Ratio) ↑ higher is better

### Region-Specific (if mask available)
- **Masked LPIPS** - LPIPS in garment region only
- **Masked SSIM** - SSIM in garment region only

### Semantic (auto-extracted)
- **mIOU** (mean Intersection over Union) - Segmentation accuracy
- **PCK** (Percentage of Correct Keypoints) - Pose alignment

## 📊 Output Format

### Individual Benchmark Results
Each script generates:
- `evaluation_results/{benchmark}/*_metrics.json` - Detailed metrics per model
- Console summary table

Example output:
```
========================================
VITON-HD BENCHMARK RESULTS SUMMARY
========================================
Model           LPIPS      SSIM       PSNR      
------------------------------------------------------------
catvton         0.1234     0.8765     25.43     
idmvton         0.1156     0.8821     26.12     
ootdiffusion    0.1289     0.8701     24.98     
dit             0.1401     0.8623     24.21     
========================================
```

### Combined Summary
`metrics_all_benchmarks.sh` generates:
- `evaluation_results/all_benchmarks_summary.json` - Combined results
- Comparison table across all benchmarks

Example:
```
CATVTON:
Benchmark       LPIPS      SSIM       PSNR      
--------------------------------------------------
VITON-HD        0.1234     0.8765     25.43     
DressCode       0.1189     0.8812     26.01     
DeepFashion     0.1301     0.8689     24.87     
Ours            0.1098     0.8901     27.34     
```

## 🔧 Customization

### Modify Models to Evaluate
Edit the `MODELS` array in any script:
```bash
MODELS=("catvton" "idmvton" "ootdiffusion" "dit")
```

### Change Device
Set `DEVICE` variable:
```bash
DEVICE="cuda"  # or "cpu"
```

### Custom Checkpoint Location
The scripts automatically find the latest checkpoint in `checkpoints_{model}/`.

To use a specific checkpoint, modify the script:
```bash
CHECKPOINT="path/to/specific/checkpoint.pt"
```

## 📝 Notes

### Automatic Feature Extraction
The evaluation framework automatically extracts:
- **Segmentation** using DeepLabV3 (for mIOU)
- **Keypoints** using Keypoint R-CNN (for PCK)

No manual preprocessing required!

### Missing Datasets
If a dataset is not found, the script will:
1. Print a warning
2. Skip that benchmark
3. Continue with other benchmarks

### Checkpoint Discovery
Scripts automatically find the latest checkpoint by:
1. Looking in `checkpoints_{model}/`
2. Sorting by modification time
3. Using the most recent `.pt` or `.pth` file

## 🐛 Troubleshooting

### "Dataset directory not found"
Download the benchmark dataset and place it in the correct location:
- VITON-HD: `baselines/viton-hd/test/`
- DressCode: `baselines/dresscode/test/`
- DeepFashion: `baselines/deepfashion/test/`

### "No checkpoint found"
Ensure you have trained the model and checkpoints exist in:
- `checkpoints_catvton/`
- `checkpoints_idmvton/`
- etc.

### CUDA out of memory
Reduce batch size in the metrics code or use CPU:
```bash
DEVICE="cpu"
```

## 📚 References

- **VITON-HD**: [Paper](https://arxiv.org/abs/2103.16874)
- **DressCode**: [Paper](https://arxiv.org/abs/2204.08532)
- **DeepFashion**: [Paper](https://arxiv.org/abs/1511.02793)

## 🎯 Recommended Workflow

1. **Train models** on your dataset
2. **Download benchmark datasets** to `baselines/`
3. **Run evaluation**:
   ```bash
   bash metrics_all_benchmarks.sh
   ```
4. **Analyze results** in `evaluation_results/`
5. **Compare** with published baselines

---

**Last Updated**: 2026-01-22
