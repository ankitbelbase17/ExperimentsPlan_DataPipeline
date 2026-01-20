"""
Utility script to verify S3 DressCODE dataset structure and test dataloader
Run this to diagnose any setup issues before training
"""

import boto3
import sys
from pathlib import Path
import config


def check_aws_credentials():
    """Verify AWS credentials are configured"""
    print("\n" + "="*60)
    print("1. CHECKING AWS CREDENTIALS")
    print("="*60)
    
    try:
        if not config.AWS_ACCESS_KEY_ID or config.AWS_ACCESS_KEY_ID == "YOUR_AWS_ACCESS_KEY":
            print("✗ AWS_ACCESS_KEY_ID not configured in config.py")
            return False
        
        if not config.AWS_SECRET_ACCESS_KEY or config.AWS_SECRET_ACCESS_KEY == "YOUR_AWS_SECRET_KEY":
            print("✗ AWS_SECRET_ACCESS_KEY not configured in config.py")
            return False
        
        print(f"✓ AWS_ACCESS_KEY_ID configured")
        print(f"✓ AWS_SECRET_ACCESS_KEY configured")
        print(f"✓ AWS_REGION: {config.AWS_REGION}")
        print(f"✓ S3_BUCKET_NAME: {config.S3_BUCKET_NAME}")
        
        return True
    except Exception as e:
        print(f"✗ Error checking credentials: {e}")
        return False


def check_s3_access():
    """Test S3 bucket access"""
    print("\n" + "="*60)
    print("2. CHECKING S3 BUCKET ACCESS")
    print("="*60)
    
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION
        )
        
        # Try to list objects
        response = s3_client.head_bucket(Bucket=config.S3_BUCKET_NAME)
        print(f"✓ Successfully connected to S3 bucket: {config.S3_BUCKET_NAME}")
        return True, s3_client
        
    except Exception as e:
        print(f"✗ Cannot access S3 bucket: {e}")
        print("  - Check AWS credentials")
        print("  - Check bucket name")
        print("  - Check bucket permissions")
        return False, None


def check_s3_structure(s3_client):
    """Verify DressCODE dataset structure in S3"""
    print("\n" + "="*60)
    print("3. CHECKING S3 DRESSCODE STRUCTURE")
    print("="*60)
    
    categories = ['dresses', 'lower_body', 'upper_body']
    subdirs = ['image', 'cloth', 'mask']
    all_good = True
    
    for category in categories:
        print(f"\n  📁 Category: {category}")
        category_path = f"{config.DRESSCODE_ROOT}/{category}"
        
        # Check subdirectories
        for subdir in subdirs:
            subdir_path = f"{category_path}/{subdir}/"
            try:
                paginator = s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=config.S3_BUCKET_NAME, Prefix=subdir_path, MaxKeys=1)
                
                found = False
                for page in pages:
                    if 'Contents' in page:
                        found = True
                        break
                
                if found:
                    print(f"    ✓ {subdir}/ exists")
                else:
                    print(f"    ✗ {subdir}/ not found")
                    all_good = False
            except Exception as e:
                print(f"    ✗ Error checking {subdir}/: {e}")
                all_good = False
        
        # Check train_pairs.txt
        pairs_file = f"{category_path}/train_pairs.txt"
        try:
            response = s3_client.get_object(Bucket=config.S3_BUCKET_NAME, Key=pairs_file)
            content = response['Body'].read().decode('utf-8')
            num_pairs = len([line for line in content.strip().split('\n') if line.strip()])
            print(f"    ✓ train_pairs.txt exists ({num_pairs} pairs)")
        except Exception as e:
            print(f"    ✗ train_pairs.txt not found: {e}")
            all_good = False
    
    return all_good


def count_images_in_s3(s3_client):
    """Count actual images in S3"""
    print("\n" + "="*60)
    print("4. COUNTING IMAGES IN S3")
    print("="*60)
    
    categories = ['dresses', 'lower_body', 'upper_body']
    
    for category in categories:
        image_prefix = f"{config.DRESSCODE_ROOT}/{category}/image/"
        
        try:
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=config.S3_BUCKET_NAME, Prefix=image_prefix)
            
            count = 0
            for page in pages:
                if 'Contents' in page:
                    count += len(page['Contents'])
            
            print(f"  {category}: {count} images")
        except Exception as e:
            print(f"  {category}: Error counting - {e}")


def test_dataloader():
    """Test the dataloader with a single batch"""
    print("\n" + "="*60)
    print("5. TESTING DATALOADER")
    print("="*60)
    
    try:
        from dataloader_dresscode_s3 import get_dresscode_dataloader
        
        print("  Loading DressCODE dataloader...")
        dataloader = get_dresscode_dataloader(batch_size=1, split='train', categories=['dresses'])
        print(f"  ✓ DataLoader created successfully")
        print(f"  ✓ Dataset size: {len(dataloader.dataset)}")
        
        print("  Loading first batch...")
        batch = next(iter(dataloader))
        
        print(f"  ✓ Batch loaded successfully")
        print(f"    - ground_truth shape: {batch['ground_truth'].shape}")
        print(f"    - cloth shape: {batch['cloth'].shape}")
        print(f"    - mask shape: {batch['mask'].shape}")
        print(f"    - category: {batch['category']}")
        
        # Check data ranges
        gt_min, gt_max = batch['ground_truth'].min().item(), batch['ground_truth'].max().item()
        mask_min, mask_max = batch['mask'].min().item(), batch['mask'].max().item()
        
        print(f"\n  Data validation:")
        print(f"    - ground_truth range: [{gt_min:.3f}, {gt_max:.3f}] (expected: [-1, 1])")
        print(f"    - mask range: [{mask_min:.3f}, {mask_max:.3f}] (expected: [0, 1])")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error testing dataloader: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks"""
    print("\n" + "="*70)
    print("DressCODE S3 DATASET VERIFICATION TOOL")
    print("="*70)
    
    # Check 1: Credentials
    if not check_aws_credentials():
        print("\n" + "="*70)
        print("❌ SETUP INCOMPLETE: Configure AWS credentials in config.py")
        print("="*70)
        return
    
    # Check 2: S3 Access
    success, s3_client = check_s3_access()
    if not success:
        print("\n" + "="*70)
        print("❌ S3 ACCESS FAILED: Check AWS credentials and bucket name")
        print("="*70)
        return
    
    # Check 3: S3 Structure
    if not check_s3_structure(s3_client):
        print("\n" + "="*70)
        print("⚠️  WARNING: Some directories are missing from S3")
        print("   Make sure your S3 structure matches the expected format")
        print("="*70)
    
    # Check 4: Count images
    count_images_in_s3(s3_client)
    
    # Check 5: Test dataloader
    if test_dataloader():
        print("\n" + "="*70)
        print("✅ ALL CHECKS PASSED!")
        print("="*70)
        print("\nYou're ready to run the gradient backpropagation test:")
        print("  python test_dresscode_gradients.py")
        print("\nThen start training:")
        print("  python train_with_dresscode.py --datasource dresscode")
    else:
        print("\n" + "="*70)
        print("❌ DATALOADER TEST FAILED: Check error messages above")
        print("="*70)


if __name__ == "__main__":
    main()
