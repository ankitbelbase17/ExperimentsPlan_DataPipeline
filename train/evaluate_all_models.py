#!/usr/bin/env python3
"""
Master Evaluation Script
Evaluates all VTON models using their latest checkpoints
"""
import os
import sys
import argparse
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.checkpoint_utils import get_latest_checkpoint
from common.metrics import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate All VTON Models")
    parser.add_argument("--test_dir", type=str, required=True, 
                       help="Test dataset directory (same structure as dataset_ultimate)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory for all results")
    parser.add_argument("--models", nargs="+", 
                       default=['catvton', 'idmvton', 'cpvton', 'vtongan', 'ootdiffusion', 'dit'],
                       help="Models to evaluate")
    parser.add_argument("--s3_bucket", type=str, default="p1-to-ep1",
                       help="S3 bucket for checkpoints")
    parser.add_argument("--use_s3", action="store_true", default=True,
                       help="Use S3 checkpoints")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*80)
    print("VTON MODELS EVALUATION")
    print("="*80)
    print(f"Test Directory: {args.test_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Device: {args.device}")
    print("="*80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Store all results
    all_results = {}
    
    # Evaluate each model
    for model_name in args.models:
        print(f"\n\n{'#'*80}")
        print(f"# EVALUATING: {model_name.upper()}")
        print(f"{'#'*80}\n")
        
        # Find latest checkpoint
        print(f"Finding latest checkpoint for {model_name}...")
        
        s3_prefix = f"checkpoints/{model_name}"
        local_dir = f"checkpoints_{model_name}"
        
        checkpoint = get_latest_checkpoint(
            model_name=model_name,
            use_s3=args.use_s3,
            local_dir=local_dir,
            s3_bucket=args.s3_bucket,
            s3_prefix=s3_prefix
        )
        
        if checkpoint is None:
            print(f"⚠️  No checkpoint found for {model_name}, skipping...")
            all_results[model_name] = {"error": "No checkpoint found"}
            continue
        
        print(f"✓ Using checkpoint: {checkpoint}\n")
        
        # Run evaluation
        try:
            results = evaluate_model(
                model_name=model_name,
                checkpoint=checkpoint,
                test_dir=args.test_dir,
                output_dir=args.output_dir,
                device=args.device
            )
            all_results[model_name] = results
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            all_results[model_name] = {"error": str(e)}
    
    # Save combined results
    print(f"\n\n{'='*80}")
    print("SUMMARY OF ALL MODELS")
    print(f"{'='*80}\n")
    
    summary_file = os.path.join(args.output_dir, "all_models_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print comparison table
    print(f"{'Model':<15} {'LPIPS':<10} {'SSIM':<10} {'PSNR':<10}")
    print("-" * 50)
    
    for model_name, results in all_results.items():
        if 'error' in results:
            print(f"{model_name:<15} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10}")
        else:
            lpips = results.get('lpips', {}).get('mean', 0)
            ssim = results.get('ssim', {}).get('mean', 0)
            psnr = results.get('psnr', {}).get('mean', 0)
            print(f"{model_name:<15} {lpips:<10.4f} {ssim:<10.4f} {psnr:<10.2f}")
    
    print(f"\n{'='*80}")
    print(f"✓ Evaluation complete!")
    print(f"✓ Results saved to: {args.output_dir}")
    print(f"✓ Summary saved to: {summary_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
