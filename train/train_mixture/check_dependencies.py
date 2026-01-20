#!/usr/bin/env python3
"""
Simple import test to verify all dependencies are available
Run this first to ensure you have all required packages
"""

import sys

def test_imports():
    """Test all required imports"""
    
    print("Testing imports...\n")
    
    imports_to_test = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("PIL", "Pillow"),
        ("boto3", "AWS SDK (boto3)"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("wandb", "Weights & Biases"),
    ]
    
    failed = []
    
    for module_name, display_name in imports_to_test:
        try:
            __import__(module_name)
            print(f"✓ {display_name:40} OK")
        except ImportError as e:
            print(f"✗ {display_name:40} MISSING")
            failed.append((module_name, display_name))
    
    print("\n" + "="*60)
    
    if failed:
        print(f"❌ {len(failed)} package(s) missing:")
        print("="*60)
        for module_name, display_name in failed:
            print(f"\n  {display_name}:")
            print(f"    pip install {module_name}")
        
        print("\n" + "="*60)
        print("Install missing packages and try again")
        return False
    else:
        print("✅ All dependencies are installed!")
        print("="*60)
        print("\nYou can now run:")
        print("  1. python verify_dresscode_setup.py")
        print("  2. python test_dresscode_gradients.py")
        print("  3. python train_with_dresscode.py --datasource dresscode")
        return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
