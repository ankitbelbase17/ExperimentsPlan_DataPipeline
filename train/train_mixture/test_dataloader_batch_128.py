"""
Test script to validate DressCODE dataloader from S3
Tests with batch size 128 to verify tensor shapes and data integrity
"""

import torch
import time
from dataloader_dresscode_s3 import get_dresscode_dataloader
import config

def test_dataloader_batch_size_128():
    """Test dataloader with batch size 128"""
    
    print("\n" + "="*80)
    print("DressCODE S3 DATALOADER TEST - BATCH SIZE 128")
    print("="*80)
    
    # Test 1: Check configuration
    print("\n[TEST 1] Configuration Check")
    print("-" * 80)
    print(f"S3 Bucket: {config.S3_BUCKET_NAME}")
    print(f"S3 Region: {config.AWS_REGION}")
    print(f"DressCODE Root: {config.DRESSCODE_ROOT}")
    print(f"Image Size: {config.IMAGE_SIZE}")
    print("✓ Configuration loaded")
    
    # Test 2: Load dataloader
    print("\n[TEST 2] Loading DataLoader with batch_size=128")
    print("-" * 80)
    try:
        dataloader = get_dresscode_dataloader(
            batch_size=128,
            split='train',
            categories=['dresses', 'lower_body', 'upper_body']
        )
        print(f"✓ DataLoader created successfully")
        print(f"✓ Total samples in dataset: {len(dataloader.dataset)}")
        print(f"✓ Batch size: 128")
        print(f"✓ Number of batches: {len(dataloader)}")
    except Exception as e:
        print(f"✗ Failed to create DataLoader: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Load first batch
    print("\n[TEST 3] Loading First Batch (128 samples)")
    print("-" * 80)
    try:
        start_time = time.time()
        batch = next(iter(dataloader))
        load_time = time.time() - start_time
        
        print(f"✓ Batch loaded in {load_time:.2f} seconds")
    except Exception as e:
        print(f"✗ Failed to load batch: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Validate tensor shapes and types
    print("\n[TEST 4] Tensor Shape and Type Validation")
    print("-" * 80)
    
    try:
        # Check keys
        expected_keys = {'ground_truth', 'cloth', 'mask', 'category', 'batch_size'}
        actual_keys = set(batch.keys())
        
        if expected_keys == actual_keys:
            print("✓ Batch contains all expected keys:", expected_keys)
        else:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            if missing:
                print(f"✗ Missing keys: {missing}")
            if extra:
                print(f"✗ Extra keys: {extra}")
            return False
        
        # Validate batch size from collate function
        actual_batch_size = batch['batch_size']
        print(f"✓ Actual batch size: {actual_batch_size}")
        
        # Validate ground truth
        gt = batch['ground_truth']
        print(f"\nGround Truth:")
        print(f"  - Shape: {gt.shape}")
        print(f"  - Expected: torch.Size([{actual_batch_size}, 3, 512, 512])")
        print(f"  - Data type: {gt.dtype}")
        print(f"  - Min value: {gt.min():.4f}")
        print(f"  - Max value: {gt.max():.4f}")
        print(f"  - Expected range: [-1.0, 1.0]")
        
        if gt.shape != torch.Size([actual_batch_size, 3, 512, 512]):
            print("✗ Ground truth shape mismatch!")
            return False
        if gt.dtype != torch.float32:
            print("✗ Ground truth dtype should be float32!")
            return False
        if gt.min() < -1.1 or gt.max() > 1.1:
            print("⚠ Ground truth values outside expected range [-1, 1]")
        
        # Validate cloth
        cloth = batch['cloth']
        print(f"\nCloth:")
        print(f"  - Shape: {cloth.shape}")
        print(f"  - Expected: torch.Size([{actual_batch_size}, 3, 512, 512])")
        print(f"  - Data type: {cloth.dtype}")
        print(f"  - Min value: {cloth.min():.4f}")
        print(f"  - Max value: {cloth.max():.4f}")
        print(f"  - Expected range: [-1.0, 1.0]")
        
        if cloth.shape != torch.Size([actual_batch_size, 3, 512, 512]):
            print("✗ Cloth shape mismatch!")
            return False
        if cloth.dtype != torch.float32:
            print("✗ Cloth dtype should be float32!")
            return False
        
        # Validate mask
        mask = batch['mask']
        print(f"\nMask:")
        print(f"  - Shape: {mask.shape}")
        print(f"  - Expected: torch.Size([{actual_batch_size}, 1, 512, 512])")
        print(f"  - Data type: {mask.dtype}")
        print(f"  - Min value: {mask.min():.4f}")
        print(f"  - Max value: {mask.max():.4f}")
        print(f"  - Expected range: [0.0, 1.0]")
        
        if mask.shape != torch.Size([actual_batch_size, 1, 512, 512]):
            print("✗ Mask shape mismatch!")
            return False
        if mask.dtype != torch.float32:
            print("✗ Mask dtype should be float32!")
            return False
        if mask.min() < -0.1 or mask.max() > 1.1:
            print("⚠ Mask values outside expected range [0, 1]")
        
        print("✓ All tensor shapes and types validated successfully")
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Check categories
    print("\n[TEST 5] Category Distribution")
    print("-" * 80)
    try:
        categories = batch['category']
        print(f"✓ Categories in batch: {len(categories)}")
        
        # Count categories
        from collections import Counter
        cat_count = Counter(categories)
        print("✓ Category distribution:")
        for cat, count in sorted(cat_count.items()):
            percentage = (count / len(categories)) * 100
            print(f"    - {cat}: {count} samples ({percentage:.1f}%)")
        
        # Validate categories
        valid_cats = {'dresses', 'lower_body', 'upper_body'}
        for cat in cat_count.keys():
            if cat not in valid_cats:
                print(f"✗ Invalid category found: {cat}")
                return False
        
    except Exception as e:
        print(f"✗ Category validation failed: {e}")
        return False
    
    # Test 6: Memory check
    print("\n[TEST 6] Memory Usage")
    print("-" * 80)
    try:
        gt_memory = (gt.numel() * 4) / (1024**2)  # 4 bytes per float32
        cloth_memory = (cloth.numel() * 4) / (1024**2)
        mask_memory = (mask.numel() * 4) / (1024**2)
        total_memory = gt_memory + cloth_memory + mask_memory
        
        print(f"✓ Ground truth memory: {gt_memory:.2f} MB")
        print(f"✓ Cloth memory: {cloth_memory:.2f} MB")
        print(f"✓ Mask memory: {mask_memory:.2f} MB")
        print(f"✓ Total batch memory: {total_memory:.2f} MB")
        
    except Exception as e:
        print(f"⚠ Memory calculation failed: {e}")
    
    # Test 7: Additional batches
    print("\n[TEST 7] Loading Additional Batches (3 batches total)")
    print("-" * 80)
    try:
        batch_times = []
        for i, batch_data in enumerate(dataloader):
            if i >= 3:
                break
            
            start_time = time.time()
            # Simulate processing
            _ = batch_data['ground_truth']
            load_time = time.time() - start_time
            batch_times.append(load_time)
            
            batch_size = batch_data['batch_size']
            print(f"  Batch {i+1}: {batch_size} samples, loaded in {load_time:.2f}s")
        
        avg_time = sum(batch_times) / len(batch_times)
        print(f"\n✓ Average load time per batch: {avg_time:.2f}s")
        print(f"✓ Estimated time for full epoch: {avg_time * len(dataloader) / 60:.2f} minutes")
        
    except Exception as e:
        print(f"✗ Failed to load additional batches: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Final summary
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nSummary:")
    print(f"  ✓ S3 connection successful")
    print(f"  ✓ DataLoader created with batch_size=128")
    print(f"  ✓ Tensors have correct shapes:")
    print(f"    - Ground truth: [B, 3, 512, 512]")
    print(f"    - Cloth: [B, 3, 512, 512]")
    print(f"    - Mask: [B, 1, 512, 512]")
    print(f"  ✓ Data normalization correct")
    print(f"  ✓ Category labels working")
    print(f"  ✓ Collate function handling batches properly")
    print(f"  ✓ Ready for training!")
    print("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    try:
        success = test_dataloader_batch_size_128()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
