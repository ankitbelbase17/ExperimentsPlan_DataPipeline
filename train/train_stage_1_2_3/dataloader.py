import os
import io
import boto3
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from torchvision import transforms
import config

class S3ImageDataset(Dataset):
    """
    Generic Dataset class for an S3 prefix
    """
    def __init__(self, s3_prefix, tokenizer=None, size=512):
        self.s3_prefix = s3_prefix
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
        
        # List objects
        print(f"Listing objects in s3://{self.bucket_name}/{self.s3_prefix}...")
        self.image_keys = []
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=self.s3_prefix)
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            self.image_keys.append(key)
        except Exception as e:
            print(f"Error listing S3 objects for {s3_prefix}: {e}")
            
        print(f"Found {len(self.image_keys)} images in {self.s3_prefix}.")
        
        self.transforms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        s3_key = self.image_keys[idx]
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            image_data = response['Body'].read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            print(f"Error loading image {s3_key}: {e}")
            image = Image.new('RGB', (self.size, self.size))

        pixel_values = self.transforms(image)
        caption = "a photo of a synthetic object" # Placeholder
        
        example = {
            "pixel_values": pixel_values,
            "input_ids": None
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

def get_dataloader(tokenizer, batch_size=None):
    if batch_size is None:
        batch_size = config.BATCH_SIZE
        
    # Create datasets
    ds1 = S3ImageDataset(config.DATASET_STAGE_1_ROOT, tokenizer, config.IMAGE_SIZE)
    ds2 = S3ImageDataset(config.DATASET_STAGE_2_ROOT, tokenizer, config.IMAGE_SIZE)
    ds3 = S3ImageDataset(config.DATASET_STAGE_3_ROOT, tokenizer, config.IMAGE_SIZE)
    
    # Handle empty datasets
    if len(ds1) == 0 and len(ds2) == 0 and len(ds3) == 0:
        print("Warning: All datasets are empty.")
        return DataLoader(ds1, batch_size=batch_size) 
        
    combined_dataset = ConcatDataset([ds1, ds2, ds3])
    
    # Calculate Weights for WeightedRandomSampler
    # Target: 25% DS1, 25% DS2, 50% DS3
    
    n1 = len(ds1)
    n2 = len(ds2)
    n3 = len(ds3)
    
    w1 = config.DATASET_WEIGHTS[0] / n1 if n1 > 0 else 0
    w2 = config.DATASET_WEIGHTS[1] / n2 if n2 > 0 else 0
    w3 = config.DATASET_WEIGHTS[2] / n3 if n3 > 0 else 0
    
    # weights array
    samples_weights = [w1] * n1 + [w2] * n2 + [w3] * n3
    
    if sum(samples_weights) == 0:
         return DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(combined_dataset),
        replacement=True
    )
    
    return DataLoader(
        combined_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=True
    )
