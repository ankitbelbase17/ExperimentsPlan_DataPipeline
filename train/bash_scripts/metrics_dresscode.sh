#!/bin/bash
# Evaluate all models on DressCode benchmark dataset
# Dataset location: baselines/dresscode/test

set -e

DATASET_DIR="baselines/dresscode/test"
OUTPUT_DIR="evaluation_results/dresscode"
DEVICE="cuda"

echo "=========================================="
echo "DressCode Benchmark Evaluation"
echo "=========================================="
echo "Dataset: $DATASET_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Check if dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory not found: $DATASET_DIR"
    echo "Please ensure DressCode test set is downloaded to baselines/dresscode/test/"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# List of models to evaluate
MODELS=("catvton" "idmvton" "ootdiffusion" "dit")

echo "Models to evaluate: ${MODELS[@]}"
echo ""

# Evaluate each model
for MODEL in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Evaluating: $MODEL"
    echo "=========================================="
    
    # Find latest checkpoint
    CHECKPOINT_DIR="checkpoints_${MODEL}"
    
    if [ -d "$CHECKPOINT_DIR" ]; then
        # Find latest .pt or .pth file
        CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/*.pt "$CHECKPOINT_DIR"/*.pth 2>/dev/null | head -n 1)
        
        if [ -z "$CHECKPOINT" ]; then
            echo "Warning: No checkpoint found for $MODEL in $CHECKPOINT_DIR"
            echo "Skipping $MODEL..."
            echo ""
            continue
        fi
        
        echo "Using checkpoint: $CHECKPOINT"
    else
        echo "Warning: Checkpoint directory not found: $CHECKPOINT_DIR"
        echo "Skipping $MODEL..."
        echo ""
        continue
    fi
    
    # Run evaluation
    python -m common.metrics \
        --model "$MODEL" \
        --checkpoint "$CHECKPOINT" \
        --test_dir "$DATASET_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --device "$DEVICE"
    
    echo ""
done

echo "=========================================="
echo "DressCode Evaluation Complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Generate summary
python -c "
import json
import os
from glob import glob

results_dir = '$OUTPUT_DIR'
results_files = glob(os.path.join(results_dir, '*_metrics.json'))

print('\n' + '='*60)
print('DRESSCODE BENCHMARK RESULTS SUMMARY')
print('='*60)
print(f'{'Model':<15} {'LPIPS':<10} {'SSIM':<10} {'PSNR':<10}')
print('-'*60)

for file in sorted(results_files):
    model = os.path.basename(file).replace('_metrics.json', '')
    with open(file) as f:
        data = json.load(f)
    
    lpips = data.get('lpips', {}).get('mean', 0)
    ssim = data.get('ssim', {}).get('mean', 0)
    psnr = data.get('psnr', {}).get('mean', 0)
    
    print(f'{model:<15} {lpips:<10.4f} {ssim:<10.4f} {psnr:<10.2f}')

print('='*60)
"

echo "Done!"
