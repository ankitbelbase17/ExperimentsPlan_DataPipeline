import os
import io
import boto3
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import config

class CATVTONDataset(Dataset):
    """
    CATVTON Dataset for Virtual Try-On
    
    Expected data structure:
    - person images: person/{id}.jpg
    - garment images: garment/{id}.jpg
    - pose maps: pose/{id}.jpg (18-channel pose heatmaps visualized as RGB)
    - segmentation: segmentation/{id}.png (body part masks)
    - pairs file: train_pairs.txt with format: person_id garment_id
    """
    def __init__(self, root_dir, pairs_file, tokenizer=None, size=512):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.size = size
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION
        )
        self.bucket_name = config.S3_BUCKET_NAME
        
        # Load pairs
        self.pairs = []
        pairs_path = os.path.join(root_dir, pairs_file)
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=pairs_path)
            pairs_content = response['Body'].read().decode('utf-8')
            for line in pairs_content.strip().split('\n'):
                if line.strip():
                    person_id, garment_id = line.strip().split()
                    self.pairs.append((person_id, garment_id))
        except Exception as e:
            print(f"Error loading pairs file: {e}")
            # Fallback: create dummy pairs
            self.pairs = [("person_001", "garment_001")]
            
        print(f"Loaded {len(self.pairs)} training pairs")
        
        # Transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        self.augment_transforms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip() if config.USE_HORIZONTAL_FLIP else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
    def load_image_from_s3(self, key):
        """Load image from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            image_data = response['Body'].read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            return image
        except Exception as e:
            # Raise exception 
            raise Exception(f"Error loading {key}: {e}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        try:
            person_id, garment_id = self.pairs[idx]
            
            # Load only person and garment images
            person_img = self.load_image_from_s3(f"{self.root_dir}/person/{person_id}.jpg")
            garment_img = self.load_image_from_s3(f"{self.root_dir}/garment/{garment_id}.jpg")
            
            # Apply transforms
            person_tensor = self.augment_transforms(person_img)
            garment_tensor = self.image_transforms(garment_img)
            
            # Text prompt
            caption = f"a person wearing {garment_id}"
            
            example = {
                "person_img": person_tensor,        # [3, H, W]
                "garment_img": garment_tensor,      # [3, H, W]
                "input_ids": None,
                "person_id": person_id,
                "garment_id": garment_id
            }
            
            if self.tokenizer:
                inputs = self.tokenizer(
                    caption,
                    max_length=self.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                example["input_ids"] = inputs.input_ids.squeeze(0)
            
            return example
            
        except Exception as e:
            # Return None to be filtered by collate_fn
            print(f"⚠️ Failed to load index {idx}: {e}")
            return None


def collate_fn_ignore_none(batch):
    """Filter out None samples from batch"""
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)


def get_dataloader(tokenizer, batch_size=None, split='train', difficulty='medium', s3_prefixes=None):
    """
    Get dataloader with difficulty-based sampling support
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    # Use S3VTONDataset if available
    try:
        from train.common.dataset import get_vton_dataset
        from torchvision import transforms
        
        # Define transforms
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
            collate_fn=collate_fn_ignore_none
        )
        
    except ImportError:
        # Fallback to original pairs-based dataset
        print("S3VTONDataset not available, using pairs-based dataset")
        pairs_file = config.TRAIN_PAIRS_FILE if split == 'train' else config.VAL_PAIRS_FILE
        
        dataset = CATVTONDataset(
            root_dir=config.DATASET_ROOT,
            pairs_file=pairs_file,
            tokenizer=tokenizer,
            size=config.IMAGE_SIZE
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=32,
            collate_fn=collate_fn_ignore_none
        )

