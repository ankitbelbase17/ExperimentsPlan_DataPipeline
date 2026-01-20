import os
import io
import boto3
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config

class S3ConnectionManager:
    """Manages S3 connection with error handling"""
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_client(self):
        """Get or create S3 client (singleton pattern)"""
        if self._client is None:
            try:
                self._client = boto3.client(
                    's3',
                    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
                    region_name=config.AWS_REGION
                )
                # Test connection
                self._client.head_bucket(Bucket=config.S3_BUCKET_NAME)
                print(f"[S3ConnectionManager] ✓ Successfully connected to S3 bucket: {config.S3_BUCKET_NAME}")
                print(f"[S3ConnectionManager] ✓ Region: {config.AWS_REGION}")
            except Exception as e:
                print(f"[S3ConnectionManager] ✗ Failed to connect to S3: {e}")
                raise
        return self._client


def dresscode_collate_fn(batch):
    """
    Custom collate function to handle batches properly.
    Handles cases where some samples might fail to load.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Dictionary with stacked tensors
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

class DressCodeS3Dataset(Dataset):
    """
    DressCODE dataset loader from S3 bucket.
    Returns: ground_truth (image), cloth (image), mask (binary mask)
    
    S3 Structure:
    dipan-dresscode-s3-bucket/
        dresscode/
            dresscode/
                dresses/
                    image/
                    cloth/
                    mask/
                    train_pairs.txt
                lower_body/
                    image/
                    cloth/
                    mask/
                    train_pairs.txt
                upper_body/
                    image/
                    cloth/
                    mask/
                    train_pairs.txt
    """
    
    def __init__(self, bucket_name, root_dir, categories=None, size=512, split='train'):
        """
        Args:
            bucket_name: S3 bucket name (e.g., 'dipan-dresscode-s3-bucket')
            root_dir: S3 prefix path (e.g., 'dresscode/dresscode')
            categories: list of categories to load (default: ['dresses', 'lower_body', 'upper_body'])
            size: image size for resizing
            split: 'train' or 'test'
        """
        self.bucket_name = bucket_name
        self.root_dir = root_dir
        self.size = size
        self.split = split
        self.categories = categories or ['dresses', 'lower_body', 'upper_body']
        
        # Get S3 client from connection manager (singleton)
        s3_manager = S3ConnectionManager()
        self.s3_client = s3_manager.get_client()
        
        # Load pairs from S3
        self.pairs = []
        self._load_pairs_from_s3()
        
        # Image transforms
        self.transforms = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
        ])
        
        # Mask transforms (no normalization, just to tensor and resize)
        self.mask_transforms = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),  # This will be [0, 1] range
        ])
        
        print(f"[DressCodeS3Dataset] Loaded {len(self.pairs)} image pairs from S3")
    
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
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=pairs_file)
                pairs_content = response['Body'].read().decode('utf-8')
                
                # Parse pairs
                for line in pairs_content.strip().split('\n'):
                    if line.strip():
                        # Expected format: "image.jpg cloth.jpg"
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            image_name = parts[0]
                            cloth_name = parts[1]
                            
                            # Construct S3 keys
                            image_key = f"{self.root_dir}/{category}/image/{image_name}"
                            cloth_key = f"{self.root_dir}/{category}/cloth/{cloth_name}"
                            mask_key = f"{self.root_dir}/{category}/mask/{image_name}"  # Same name as image
                            
                            self.pairs.append({
                                'category': category,
                                'image_key': image_key,
                                'cloth_key': cloth_key,
                                'mask_key': mask_key,
                            })
                
                print(f"[DressCodeS3Dataset] Loaded {len([p for p in self.pairs if p['category'] == category])} pairs from {category}")
                
            except Exception as e:
                print(f"[DressCodeS3Dataset] Warning: Could not load pairs for {category}: {e}")
    
    def _load_image_from_s3(self, s3_key):
        """Load image from S3 and return PIL Image"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            image_data = response['Body'].read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            return image
        except Exception as e:
            print(f"[DressCodeS3Dataset] Error loading {s3_key}: {e}")
            return Image.new('RGB', (self.size, self.size))
    
    def _load_mask_from_s3(self, s3_key):
        """Load mask from S3 and return PIL Image in grayscale"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            image_data = response['Body'].read()
            mask = Image.open(io.BytesIO(image_data)).convert("L")  # Grayscale
            return mask
        except Exception as e:
            print(f"[DressCodeS3Dataset] Error loading mask {s3_key}: {e}")
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
        ground_truth_tensor = self.transforms(ground_truth)  # [3, H, W], range [-1, 1]
        cloth_tensor = self.transforms(cloth)                # [3, H, W], range [-1, 1]
        mask_tensor = self.mask_transforms(mask)             # [1, H, W], range [0, 1]
        
        example = {
            "ground_truth": ground_truth_tensor,
            "cloth": cloth_tensor,
            "mask": mask_tensor,
            "category": pair['category']
        }
        
        return example


def get_dresscode_dataloader(batch_size=None, split='train', categories=None):
    """
    Create a DataLoader for DressCODE dataset from S3
    
    Args:
        batch_size: batch size (default from config)
        split: 'train' or 'test'
        categories: list of categories to load
    
    Returns:
        DataLoader with custom collate function
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    dataset = DressCodeS3Dataset(
        bucket_name=config.S3_BUCKET_NAME,
        root_dir=config.DRESSCODE_ROOT,
        categories=categories,
        size=config.IMAGE_SIZE,
        split=split
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=0,  # Keep 0 for S3 operations (boto3 not multiprocessing-safe)
        drop_last=(split == 'train'),
        collate_fn=dresscode_collate_fn  # Use custom collate function
    )


if __name__ == "__main__":
    # Test the dataloader
    dataloader = get_dresscode_dataloader(batch_size=2, split='train')
    print(f"DataLoader created with {len(dataloader.dataset)} samples")
    
    # Get one batch
    for batch in dataloader:
        print("\nBatch keys:", batch.keys())
        print("Ground truth shape:", batch['ground_truth'].shape)
        print("Cloth shape:", batch['cloth'].shape)
        print("Mask shape:", batch['mask'].shape)
        print("Categories:", batch['category'])
        break
