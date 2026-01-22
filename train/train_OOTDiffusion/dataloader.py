import os
import io
import boto3
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np
import config

class OOTDiffusionDataset(Dataset):
    """
    OOTDiffusion Dataset with high-resolution images and pose information
    
    Expected structure:
    - person/{id}.jpg (768x1024)
    - garment/{id}.jpg
    - pose/{id}.npy (18-channel pose map: OpenPose + DensePose)
    - mask/{id}.png (inpainting mask)
    - pairs file with: person_id garment_id
    """
    def __init__(self, root_dir, pairs_file, tokenizer=None, height=1024, width=768):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.height = height
        self.width = width
        
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
            self.pairs = [("person_001", "garment_001")]
            
        print(f"Loaded {len(self.pairs)} OOTDiffusion training pairs")
        
        # Image transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        self.mask_transforms = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
        ])
        
    def load_image_from_s3(self, key):
        """Load image from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            image_data = response['Body'].read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            return image
        except Exception as e:
            raise Exception(f"Error loading {key}: {e}")
    
    def load_pose_from_s3(self, key):
        """Load pose numpy array from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            pose_data = response['Body'].read()
            pose_array = np.load(io.BytesIO(pose_data))
            return pose_array
        except Exception as e:
            raise Exception(f"Error loading pose {key}: {e}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        try:
            person_id, garment_id = self.pairs[idx]
            
            # Load all required data
            person_img = self.load_image_from_s3(f"{self.root_dir}/person/{person_id}.jpg")
            garment_img = self.load_image_from_s3(f"{self.root_dir}/garment/{garment_id}.jpg")
            mask_img = self.load_image_from_s3(f"{self.root_dir}/mask/{person_id}.png")
            
            # Load pose map (18 channels: OpenPose keypoints + DensePose)
            pose_array = self.load_pose_from_s3(f"{self.root_dir}/pose/{person_id}.npy")
            
            # Apply transforms
            person_tensor = self.image_transforms(person_img)      # [3, H, W]
            garment_tensor = self.image_transforms(garment_img)    # [3, H, W]
            mask_tensor = self.mask_transforms(mask_img)[:1]       # [1, H, W]
            
            # Resize pose map if needed
            if pose_array.shape[1:] != (self.height, self.width):
                pose_tensor = torch.from_numpy(pose_array).float()
                pose_tensor = torch.nn.functional.interpolate(
                    pose_tensor.unsqueeze(0),
                    size=(self.height, self.width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            else:
                pose_tensor = torch.from_numpy(pose_array).float()  # [18, H, W]
            
            # Text prompt
            caption = f"a person wearing a {garment_id} garment"
            
            example = {
                "person_img": person_tensor,        # [3, H, W]
                "garment_img": garment_tensor,      # [3, H, W]
                "pose_map": pose_tensor,            # [18, H, W]
                "mask": mask_tensor,                # [1, H, W]
                "input_ids": None,
                "person_id": person_id,
                "garment_id": garment_id,
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
            print(f"⚠️ Error loading index {idx}: {e}")
            return None


def collate_fn_ignore_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)


def get_dataloader(tokenizer, batch_size=None, split='train', difficulty='medium', s3_prefixes=None):
    """
    Get dataloader with difficulty-based sampling support
    
    Args:
        tokenizer: Tokenizer for text processing
        batch_size: Batch size (default: config.BATCH_SIZE)
        split: 'train' or 'val'
        difficulty: 'easy', 'medium', or 'hard' for curriculum learning
        s3_prefixes: List of S3 prefixes to load from (optional)
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    # Use S3VTONDataset if available
    try:
        from train.common.dataset import get_vton_dataset
        from torchvision import transforms
        
        # Define transforms - OOTDiffusion uses high-res (1024x768)
        transform = transforms.Compose([
            transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
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
        # Fallback
        print("S3VTONDataset not available, using simple paired dataset")
        pairs_file = config.TRAIN_PAIRS_FILE if split == 'train' else config.VAL_PAIRS_FILE
        
        dataset = OOTDiffusionDataset(
            root_dir=config.DATASET_ROOT,
            pairs_file=pairs_file,
            tokenizer=tokenizer,
            height=config.IMAGE_HEIGHT,
            width=config.IMAGE_WIDTH
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=32,
            collate_fn=collate_fn_ignore_none
        )
