import os
import io
import boto3
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config

class Stage1Dataset(Dataset):
    def __init__(self, root_dir, tokenizer=None, size=512):
        self.root_dir = root_dir # This corresponds to the S3 Prefix now
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
        
        # List objects in S3 bucket with the prefix
        print(f"Listing objects in s3://{self.bucket_name}/{self.root_dir}...")
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=self.root_dir)
            
            self.image_keys = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            self.image_keys.append(key)
                            
            print(f"Found {len(self.image_keys)} images in S3.")
            
        except Exception as e:
            print(f"Error listing S3 objects: {e}")
            self.image_keys = []
            
        self.transforms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        # file path here essentially acts as the S3 Key
        s3_key = self.image_keys[idx]
        
        try:
            # Fetch image from S3
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            image_data = response['Body'].read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            print(f"Error loading image {s3_key}: {e}")
            # Return a blank image or handle error appropriately
            image = Image.new('RGB', (self.size, self.size))

        pixel_values = self.transforms(image)
        
        # Placeholder for caption
        caption = "a photo of a synthetic object" 
        
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
        
    dataset = Stage1Dataset(
        root_dir=config.DATASET_ROOT,
        tokenizer=tokenizer,
        size=config.IMAGE_SIZE
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, # Shuffle is good for training
        num_workers=0 # S3/boto3 often has issues with multiprocessing. Start with 0.
    )
