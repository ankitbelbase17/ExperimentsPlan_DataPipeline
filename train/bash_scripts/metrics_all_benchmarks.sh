#!/bin/bash
# Run evaluation on ALL benchmark datasets
# This script evaluates all models on VITON-HD, DressCode, DeepFashion, and our custom test set

set -e

echo "=========================================="
echo "COMPREHENSIVE BENCHMARK EVALUATION"
echo "=========================================="
echo "This will evaluate all models on:"
echo "  1. VITON-HD"
echo "  2. DressCode"
echo "  3. DeepFashion"
echo "  4. Our Custom Test Set"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run each benchmark
echo ""
echo "=========================================="
echo "1/4: VITON-HD Benchmark"
echo "=========================================="
bash "$SCRIPT_DIR/metrics_vitonhd.sh"

echo ""
echo "=========================================="
echo "2/4: DressCode Benchmark"
echo "=========================================="
bash "$SCRIPT_DIR/metrics_dresscode.sh"

echo ""
echo "=========================================="
echo "3/4: DeepFashion Benchmark"
echo "=========================================="
bash "$SCRIPT_DIR/metrics_deepfashion.sh"

echo ""
echo "=========================================="
echo "4/4: Our Custom Test Set"
echo "=========================================="
bash "$SCRIPT_DIR/metrics_ours_test.sh"

# Generate combined summary
echo ""
echo "=========================================="
echo "GENERATING COMBINED SUMMARY"
echo "=========================================="

python -c "
import json
import os
from glob import glob
from collections import defaultdict

# Collect all results
benchmarks = {
    'VITON-HD': 'evaluation_results/viton-hd',
    'DressCode': 'evaluation_results/dresscode',
    'DeepFashion': 'evaluation_results/deepfashion',
    'Ours': 'evaluation_results/ours_test'
}

all_results = defaultdict(dict)

for bench_name, results_dir in benchmarks.items():
    if not os.path.exists(results_dir):
        continue
    
    results_files = glob(os.path.join(results_dir, '*_metrics.json'))
    
    for file in results_files:
        model = os.path.basename(file).replace('_metrics.json', '')
        
        with open(file) as f:
            data = json.load(f)
        
        all_results[model][bench_name] = {
            'lpips': data.get('lpips', {}).get('mean', 0),
            'ssim': data.get('ssim', {}).get('mean', 0),
            'psnr': data.get('psnr', {}).get('mean', 0)
        }

# Print combined table
print('\n' + '='*100)
print('COMPREHENSIVE BENCHMARK RESULTS - ALL DATASETS')
print('='*100)

for model in sorted(all_results.keys()):
    print(f'\n{model.upper()}:')
    print(f'{'Benchmark':<15} {'LPIPS':<10} {'SSIM':<10} {'PSNR':<10}')
    print('-'*50)
    
    for bench_name in ['VITON-HD', 'DressCode', 'DeepFashion', 'Ours']:
        if bench_name in all_results[model]:
            metrics = all_results[model][bench_name]
            print(f'{bench_name:<15} {metrics[\"lpips\"]:<10.4f} {metrics[\"ssim\"]:<10.4f} {metrics[\"psnr\"]:<10.2f}')
        else:
            print(f'{bench_name:<15} {'N/A':<10} {'N/A':<10} {'N/A':<10}')

print('\n' + '='*100)

# Save combined results
combined_file = 'evaluation_results/all_benchmarks_summary.json'
os.makedirs('evaluation_results', exist_ok=True)
with open(combined_file, 'w') as f:
    json.dump(dict(all_results), f, indent=2)

print(f'\nCombined results saved to: {combined_file}')
"

echo ""
echo "=========================================="
echo "ALL BENCHMARKS COMPLETE!"
echo "=========================================="
echo "Results saved to evaluation_results/"
echo ""
