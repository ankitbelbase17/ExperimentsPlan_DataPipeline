import os
import io
import boto3
import torch
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config

class S3TripleDataset(Dataset):
    """
    Expects triplets of files in S3.
    Since we don't have the exact structure, we assume:
    - filename.jpg (Person)
    - filename_cloth.jpg (Cloth)
    - filename_mask.jpg (Agnostic Mask)
    Or we simulate it loosely for this template.
    """
    def __init__(self, root_dir, size=512):
        self.root_dir = root_dir
        self.size = size
        
        # Initialize S3 client (assuming S3 logic is same as before)
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION
        )
        self.bucket_name = config.S3_BUCKET_NAME
        
        # List objects
        print(f"Listing objects in s3://{self.bucket_name}/{self.root_dir}...")
        self.keys = []
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=self.root_dir)
            
            all_keys = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        all_keys.append(obj['Key'])
            
            # Simple heuristic: Filter for base images (not ending in _cloth or _mask)
            # This is an assumption. In production, provide a manifest CSV.
            images = [k for k in all_keys if k.lower().endswith(('.jpg', '.png')) 
                     and not k.lower().endswith(('_cloth.jpg', '_cloth.png', '_mask.jpg', '_mask.png'))]
            
            self.keys = images
            
        except Exception as e:
            print(f"Error listing S3 objects: {e}")
            
        print(f"Found {len(self.keys)} potential triplet samples.")
        
        self.transform_image = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.transform_mask = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor() # 0 to 1
        ])
        
        # CLIP Expects specific normalization, but we usually pass raw pixels to processor
        # We'll just resize here
        self.transform_clip = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(), # Processor handles norm
        ])

    def fetch_image(self, key):
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            data = response['Body'].read()
            return Image.open(io.BytesIO(data)).convert("RGB")
        except:
             # Fallback
            return Image.new("RGB", (self.size, self.size), (128, 128, 128))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        person_key = self.keys[idx]
        
        # Infer other keys
        # Assumption: ext is 4 chars (.jpg)
        base = person_key[:-4] 
        ext = person_key[-4:]
        
        cloth_key = f"{base}_cloth{ext}"
        mask_key = f"{base}_mask{ext}"
        
        # Fetch (In reality, would verify existence or rely on manifest)
        person_img = self.fetch_image(person_key)
        cloth_img = self.fetch_image(cloth_key)
        
        # For mask, fetch or generate dummy if missing (for robustness of this template)
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=mask_key)
            mask_data = response['Body'].read()
            mask_img = Image.open(io.BytesIO(mask_data)).convert("L")
        except:
            # Dummy agnostic mask (center hole)
            mask_img = Image.new("L", (self.size, self.size), 0)
            # In a real agnositic mask, the body is usually masked? Or the cloth area?
            # "masking the cloth region" -> cloth region is 1 (white), others 0 (black)?
            # or inpainting unet expects 1 for "mask out/inpainting area"?
            # Standard SD Inpainting: 1 = mask (re-generate), 0 = keep.
            
            # Let's simple mask center
            # In real data, mask is provided.
            mask_img = Image.new("L", (self.size, self.size), 0) # Keep all
        
        # Transforms
        person_tensor = self.transform_image(person_img)
        cloth_tensor = self.transform_clip(cloth_img) # For CLIP Encoder
        mask_tensor = self.transform_mask(mask_img)
        
        # Create "masked_image"
        # masked_image = person * (1 - mask)
        # Note: Person is [-1, 1], mask is [0, 1]
        # In latent space, we mask latents, but some pipelines mask pixels too.
        # SD Inpainting Pipeline usually expects:
        # - image (original)
        # - mask_image
        # We will return original and mask, masking happens in training loop ideally for latents.
        
        return {
            "person_pixel_values": person_tensor,
            "cloth_pixel_values": cloth_tensor, # Ready for CLIP processor? No, tensor is fine if we skip processor norm or do it manually
            # Actually, huggingface CLIPImageProcessor expects numpy or list or tensor 0-1.
            # We already converted to tensor 0-1 in transform_clip.
            "mask": mask_tensor
        }

def get_dataloader(batch_size=None):
    if batch_size is None:
        batch_size = config.BATCH_SIZE
        
    dataset = S3TripleDataset(root_dir=config.DATASET_ROOT, size=config.IMAGE_SIZE)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
