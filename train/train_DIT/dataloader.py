import os
import io
import boto3
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import config

class ImageNetDataset(Dataset):
    """
    ImageNet-style dataset for DiT training
    
    Expected structure:
    - images/{class_id}/{image_id}.jpg
    - labels.txt with format: image_path class_id
    """
    def __init__(self, root_dir, split='train', size=256):
        self.root_dir = root_dir
        self.size = size
        
        # S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION
        )
        self.bucket_name = config.S3_BUCKET_NAME
        
        # Load image paths and labels
        self.samples = []
        labels_file = f"{root_dir}/{split}_labels.txt"
        
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=labels_file)
            labels_content = response['Body'].read().decode('utf-8')
            for line in labels_content.strip().split('\n'):
                if line.strip():
                    image_path, label = line.strip().split()
                    self.samples.append((image_path, int(label)))
        except Exception as e:
            print(f"Error loading labels: {e}")
            # Create dummy samples
            for i in range(1000):
                self.samples.append((f"dummy_{i}.jpg", i % 1000))
        
        print(f"Loaded {len(self.samples)} samples for {split}")
        
        # Transforms
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize(size),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
    
    def load_image_from_s3(self, key):
        try:
            full_key = f"{self.root_dir}/{key}"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=full_key)
            image_data = response['Body'].read()
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            print(f"Error loading {key}: {e}")
            return Image.new('RGB', (self.size, self.size))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        image = self.load_image_from_s3(image_path)
        image_tensor = self.transforms(image)
        
        return {
            "image": image_tensor,      # [3, H, W]
            "label": torch.tensor(label, dtype=torch.long)  # scalar
        }



def get_dataloader(batch_size=None, split='train', difficulty='medium', s3_prefixes=None):
    """
    Get dataloader with difficulty-based sampling support
    
    Args:
        batch_size: Batch size (default: config.BATCH_SIZE)
        split: 'train' or 'val'
        difficulty: 'easy', 'medium', or 'hard' for curriculum learning
        s3_prefixes: List of S3 prefixes to load from (optional)
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    # Use S3VTONDataset if available, otherwise fallback to ImageNet (original DiT dataset)
    try:
        from train.common.dataset import get_vton_dataset
        from torchvision import transforms
        
        # Define transforms for VTON dataset -> DiT format
        # DiT expects 'image' and 'label'
        # We need to adapt the VTON triplet to this format
        # For simple DiT training, we can use the ground truth (try_on_image) as the target image
        transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        # Default S3 prefixes if not provided
        if s3_prefixes is None:
            if difficulty == 'easy':
                s3_prefixes = [
                    'dataset_ultimate/easy/female/',
                    'dataset_ultimate/easy/male/',
                ]
            elif difficulty == 'medium':
                s3_prefixes = [
                    'dataset_ultimate/easy/female/',
                    'dataset_ultimate/easy/male/',
                    'dataset_ultimate/medium/female/',
                    'dataset_ultimate/medium/male/',
                ]
            else:  # hard
                s3_prefixes = [
                    'dataset_ultimate/easy/female/',
                    'dataset_ultimate/easy/male/',
                    'dataset_ultimate/medium/female/',
                    'dataset_ultimate/medium/male/',
                    'dataset_ultimate/hard/female/',
                    'dataset_ultimate/hard/male/',
                ]
        
        # Get dataset with difficulty-based sampling
        dataset = get_vton_dataset(
            difficulty=difficulty,
            s3_prefixes=s3_prefixes,
            transform=transform,
            s3_bucket=config.S3_BUCKET_NAME
        )
        
        print(f"Using S3VTONDataset with difficulty: {difficulty}")
        print(f"Dataset size: {len(dataset)}")
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=32,
            pin_memory=True,
            collate_fn=collate_fn_ignore_none
        )

def collate_fn_ignore_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)

    except ImportError:
        # Fallback to original ImageNet logic if common dataset not available
        print("S3VTONDataset not available, falling back to local ImageNet")
        
        if split == 'train':
            transform = transforms.Compose([
                transforms.Resize(config.IMAGE_SIZE),
                transforms.RandomCrop(config.IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(config.IMAGE_SIZE),
                transforms.CenterCrop(config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            
        dataset = ImageNetDataset(root_dir=config.DATASET_ROOT, split=split, size=config.IMAGE_SIZE)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=4,
            pin_memory=True
        )
