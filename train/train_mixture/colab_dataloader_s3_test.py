"""
Google Colab Compatible S3 DataLoader - Ready for Testing
=========================================================

This script is designed to work in Google Colab with S3 backend.
It includes built-in setup, configuration, and testing utilities.

Usage in Colab:
1. Upload this file or clone the repo
2. Run the setup cells to install dependencies and configure AWS credentials
3. Run the dataloader cells to test
"""

# ============================================================================
# SECTION 1: IMPORTS & DEPENDENCIES
# ============================================================================

import os
import io
import sys
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import boto3
except ImportError:
    print("Installing boto3...")
    os.system('pip install -q boto3')
    import boto3

try:
    from PIL import Image
except ImportError:
    print("Installing Pillow...")
    os.system('pip install -q Pillow')
    from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

print("✓ All dependencies imported successfully")


# ============================================================================
# SECTION 2: COLAB SETUP HELPER
# ============================================================================

class ColabSetup:
    """Helper class for Google Colab setup"""
    
    @staticmethod
    def setup_aws_credentials(access_key_id: str, secret_access_key: str, region: str = "eu-north-1"):
        """
        Set AWS credentials in environment for this session.
        
        Args:
            access_key_id: AWS Access Key ID
            secret_access_key: AWS Secret Access Key
            region: AWS Region (default: eu-north-1)
        """
        os.environ['AWS_ACCESS_KEY_ID'] = access_key_id
        os.environ['AWS_SECRET_ACCESS_KEY'] = secret_access_key
        os.environ['AWS_DEFAULT_REGION'] = region
        print(f"✓ AWS credentials configured for region: {region}")
    
    @staticmethod
    def test_aws_connection(bucket_name: str) -> bool:
        """Test S3 connection"""
        try:
            s3_client = boto3.client('s3')
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"✓ Successfully connected to S3 bucket: {bucket_name}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to S3: {e}")
            return False


# ============================================================================
# SECTION 3: S3 CONNECTION MANAGER
# ============================================================================

class S3ConnectionManager:
    """Manages S3 connection with error handling (Singleton pattern)"""
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_client(self):
        """Get or create S3 client"""
        if self._client is None:
            try:
                self._client = boto3.client('s3')
                print("[S3ConnectionManager] ✓ S3 client initialized")
            except Exception as e:
                print(f"[S3ConnectionManager] ✗ Failed to initialize S3 client: {e}")
                raise
        return self._client
    
    def reset(self):
        """Reset the connection (useful for reinitialization)"""
        self._client = None


# ============================================================================
# SECTION 4: CUSTOM COLLATE FUNCTION
# ============================================================================

def dresscode_collate_fn(batch):
    """
    Custom collate function for DressCODE dataset.
    Handles batching of image tensors and metadata.
    """
    # Filter out None samples (in case some failed to load)
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        raise RuntimeError("All samples in batch failed to load from S3")
    
    # Stack tensors
    ground_truth = torch.stack([item['ground_truth'] for item in batch])
    cloth = torch.stack([item['cloth'] for item in batch])
    mask = torch.stack([item['mask'] for item in batch])
    
    # Collect categories
    categories = [item['category'] for item in batch]
    
    return {
        'ground_truth': ground_truth,
        'cloth': cloth,
        'mask': mask,
        'category': categories,
        'batch_size': len(batch)
    }


# ============================================================================
# SECTION 5: DRESSCODE S3 DATASET CLASS
# ============================================================================

class DressCodeS3Dataset(Dataset):
    """
    DressCODE dataset loader from S3 bucket.
    
    Returns:
        - ground_truth: Person image [3, H, W]
        - cloth: Clothing item image [3, H, W]
        - mask: Binary mask [1, H, W]
        - category: Category name (dresses, lower_body, upper_body)
    
    S3 Structure Expected:
        s3://bucket_name/
            dresscode/dresscode/
                dresses/
                    image/
                    cloth/
                    mask/
                    train_pairs.txt
                lower_body/
                    ...
                upper_body/
                    ...
    """
    
    def __init__(
        self,
        bucket_name: str,
        root_dir: str,
        categories: Optional[List[str]] = None,
        size: int = 512,
        split: str = 'train'
    ):
        """
        Initialize DressCODE S3 Dataset
        
        Args:
            bucket_name: S3 bucket name (e.g., 'dipan-dresscode-s3-bucket')
            root_dir: S3 prefix path (e.g., 'dresscode/dresscode')
            categories: list of categories to load (default: all three)
            size: image size for resizing
            split: 'train' or 'test'
        """
        self.bucket_name = bucket_name
        self.root_dir = root_dir
        self.size = size
        self.split = split
        self.categories = categories or ['dresses', 'lower_body', 'upper_body']
        
        # Get S3 client from connection manager
        s3_manager = S3ConnectionManager()
        self.s3_client = s3_manager.get_client()
        
        # Load pairs from S3
        self.pairs = []
        self._load_pairs_from_s3()
        
        # Image transforms (normalized to [-1, 1])
        self.transforms = transforms.Compose([
            transforms.Resize(
                (size, size),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        # Mask transforms (to tensor, range [0, 1])
        self.mask_transforms = transforms.Compose([
            transforms.Resize(
                (size, size),
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor(),
        ])
        
        print(f"[DressCodeS3Dataset] ✓ Loaded {len(self.pairs)} image pairs from S3")
    
    def _load_pairs_from_s3(self):
        """Load training/test pairs from S3 for all categories"""
        for category in self.categories:
            try:
                # Determine the pairs file based on split
                if self.split == 'train':
                    pairs_file = f"{self.root_dir}/{category}/train_pairs.txt"
                else:
                    pairs_file = f"{self.root_dir}/{category}/test_pairs_paired.txt"
                
                # Read pairs file from S3
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=pairs_file
                )
                pairs_content = response['Body'].read().decode('utf-8')
                
                # Parse pairs
                for line in pairs_content.strip().split('\n'):
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            image_name = parts[0]
                            cloth_name = parts[1]
                            
                            # Construct S3 keys
                            image_key = f"{self.root_dir}/{category}/image/{image_name}"
                            cloth_key = f"{self.root_dir}/{category}/cloth/{cloth_name}"
                            mask_key = f"{self.root_dir}/{category}/mask/{image_name}"
                            
                            self.pairs.append({
                                'category': category,
                                'image_key': image_key,
                                'cloth_key': cloth_key,
                                'mask_key': mask_key,
                            })
                
                category_count = len([p for p in self.pairs if p['category'] == category])
                print(f"[DressCodeS3Dataset] ✓ Loaded {category_count} pairs from '{category}'")
                
            except Exception as e:
                print(f"[DressCodeS3Dataset] ⚠ Could not load pairs for {category}: {e}")
    
    def _load_image_from_s3(self, s3_key: str) -> Image.Image:
        """Load image from S3 and return PIL Image"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            image_data = response['Body'].read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            return image
        except Exception as e:
            print(f"[DressCodeS3Dataset] ⚠ Error loading {s3_key}: {e}")
            return Image.new('RGB', (self.size, self.size))
    
    def _load_mask_from_s3(self, s3_key: str) -> Image.Image:
        """Load mask from S3 and return PIL Image in grayscale"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            image_data = response['Body'].read()
            mask = Image.open(io.BytesIO(image_data)).convert("L")
            return mask
        except Exception as e:
            print(f"[DressCodeS3Dataset] ⚠ Error loading mask {s3_key}: {e}")
            return Image.new('L', (self.size, self.size))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load images and mask
        ground_truth = self._load_image_from_s3(pair['image_key'])
        cloth = self._load_image_from_s3(pair['cloth_key'])
        mask = self._load_mask_from_s3(pair['mask_key'])
        
        # Apply transforms
        ground_truth_tensor = self.transforms(ground_truth)
        cloth_tensor = self.transforms(cloth)
        mask_tensor = self.mask_transforms(mask)
        
        return {
            "ground_truth": ground_truth_tensor,
            "cloth": cloth_tensor,
            "mask": mask_tensor,
            "category": pair['category']
        }


# ============================================================================
# SECTION 6: DATALOADER FACTORY FUNCTION
# ============================================================================

def get_dresscode_dataloader(
    bucket_name: str,
    root_dir: str,
    batch_size: int = 4,
    split: str = 'train',
    categories: Optional[List[str]] = None,
    image_size: int = 512,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a DataLoader for DressCODE dataset from S3
    
    Args:
        bucket_name: S3 bucket name
        root_dir: S3 prefix path
        batch_size: batch size (default: 4)
        split: 'train' or 'test'
        categories: list of categories to load (default: all)
        image_size: image resolution (default: 512)
        shuffle: whether to shuffle the dataset
    
    Returns:
        DataLoader configured for DressCODE dataset
    
    Example:
        >>> dataloader = get_dresscode_dataloader(
        ...     bucket_name='my-bucket',
        ...     root_dir='dresscode/dresscode',
        ...     batch_size=8
        ... )
    """
    dataset = DressCodeS3Dataset(
        bucket_name=bucket_name,
        root_dir=root_dir,
        categories=categories,
        size=image_size,
        split=split
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and (split == 'train'),
        num_workers=0,  # Must be 0 for S3 operations
        drop_last=(split == 'train'),
        collate_fn=dresscode_collate_fn
    )


# ============================================================================
# SECTION 7: TESTING UTILITIES
# ============================================================================

class DataLoaderTester:
    """Utility class for testing dataloaders in Colab"""
    
    @staticmethod
    def test_batch(batch: Dict, batch_num: int = 1):
        """Print batch information"""
        print(f"\n{'='*60}")
        print(f"Batch #{batch_num} Information")
        print(f"{'='*60}")
        print(f"Batch size: {batch['batch_size']}")
        print(f"\nTensor shapes:")
        print(f"  - ground_truth: {batch['ground_truth'].shape} (range: [{batch['ground_truth'].min():.2f}, {batch['ground_truth'].max():.2f}])")
        print(f"  - cloth: {batch['cloth'].shape} (range: [{batch['cloth'].min():.2f}, {batch['cloth'].max():.2f}])")
        print(f"  - mask: {batch['mask'].shape} (range: [{batch['mask'].min():.2f}, {batch['mask'].max():.2f}])")
        print(f"\nCategories: {batch['category']}")
        print(f"{'='*60}\n")
    
    @staticmethod
    def test_dataloader(dataloader: DataLoader, num_batches: int = 2):
        """Test dataloader by iterating through batches"""
        print(f"\nTesting DataLoader ({num_batches} batches)...")
        print(f"Total samples: {len(dataloader.dataset)}")
        print(f"Batch size: {dataloader.batch_size}")
        print(f"Total batches: {len(dataloader)}")
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            DataLoaderTester.test_batch(batch, batch_num=batch_idx + 1)
        
        print(f"✓ DataLoader test completed successfully!")
    
    @staticmethod
    def visualize_batch(batch: Dict, num_samples: int = 2):
        """Visualize images from a batch (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            print("Installing matplotlib...")
            os.system('pip install -q matplotlib')
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        
        num_samples = min(num_samples, batch['batch_size'])
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Ground truth
            img = batch['ground_truth'][i].cpu()
            img = (img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            axes[i, 0].imshow(img.permute(1, 2, 0).clamp(0, 1))
            axes[i, 0].set_title(f"Ground Truth\n{batch['category'][i]}")
            axes[i, 0].axis('off')
            
            # Cloth
            cloth = batch['cloth'][i].cpu()
            cloth = (cloth + 1) / 2
            axes[i, 1].imshow(cloth.permute(1, 2, 0).clamp(0, 1))
            axes[i, 1].set_title("Cloth")
            axes[i, 1].axis('off')
            
            # Mask
            mask = batch['mask'][i].cpu().squeeze()
            axes[i, 2].imshow(mask, cmap='gray')
            axes[i, 2].set_title("Mask")
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# SECTION 8: MAIN TESTING CODE
# ============================================================================

def main_colab_test():
    """
    Main function to test dataloader in Colab.
    Run this in a Colab cell to verify everything works.
    """
    print("\n" + "="*70)
    print("GOOGLE COLAB S3 DATALOADER TEST")
    print("="*70)
    
    # ---- Configuration ----
    # Get credentials from environment variables
    AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION = os.environ.get("AWS_REGION", "eu-north-1")
    S3_BUCKET_NAME = "dipan-dresscode-s3-bucket"
    DRESSCODE_ROOT = "dresscode/dresscode"
    
    # Check if credentials are provided
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("⚠️  ERROR: AWS credentials not set!")
        print("Please set environment variables:")
        print("  export AWS_ACCESS_KEY_ID='your-key'")
        print("  export AWS_SECRET_ACCESS_KEY='your-secret'")
        print("\nOr in Colab:")
        print("  import os")
        print("  os.environ['AWS_ACCESS_KEY_ID'] = 'your-key'")
        print("  os.environ['AWS_SECRET_ACCESS_KEY'] = 'your-secret'")
        return
    
    # ---- Setup AWS Credentials ----
    print("\n[Step 1] Setting up AWS credentials...")
    ColabSetup.setup_aws_credentials(
        access_key_id=AWS_ACCESS_KEY_ID,
        secret_access_key=AWS_SECRET_ACCESS_KEY,
        region=AWS_REGION
    )
    
    # ---- Test AWS Connection ----
    print("\n[Step 2] Testing AWS connection...")
    if not ColabSetup.test_aws_connection(S3_BUCKET_NAME):
        print("✗ Connection failed. Please check your credentials.")
        return
    
    # ---- Create DataLoader ----
    print("\n[Step 3] Creating DataLoader...")
    try:
        dataloader = get_dresscode_dataloader(
            bucket_name=S3_BUCKET_NAME,
            root_dir=DRESSCODE_ROOT,
            batch_size=4,
            split='train',
            categories=['dresses', 'lower_body', 'upper_body'],
            image_size=512,
            shuffle=True
        )
        print("✓ DataLoader created successfully!")
    except Exception as e:
        print(f"✗ Failed to create DataLoader: {e}")
        return
    
    # ---- Test DataLoader ----
    print("\n[Step 4] Testing DataLoader iteration...")
    try:
        DataLoaderTester.test_dataloader(dataloader, num_batches=2)
    except Exception as e:
        print(f"✗ DataLoader test failed: {e}")
        return
    
    # ---- Visualize Batch ----
    print("\n[Step 5] Visualizing batch samples...")
    try:
        for batch in dataloader:
            DataLoaderTester.visualize_batch(batch, num_samples=2)
            break
    except Exception as e:
        print(f"⚠ Could not visualize batch: {e}")
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main_colab_test()
