import os
import io
import boto3
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config

class CPVTONDataset(Dataset):
    """CP-VTON Dataset - person, garment, person representation, target"""
    def __init__(self, root_dir, pairs_file, size=256):
        self.root_dir = root_dir
        self.size = size
        
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
        except:
            self.pairs = [("person_001", "garment_001")]
        
        self.transforms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def load_image_from_s3(self, key):
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            image_data = response['Body'].read()
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            # Raise exception for retry
            raise Exception(f"Failed to load {key}: {e}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        try:
            person_id, garment_id = self.pairs[idx]
            
            person = self.load_image_from_s3(f"{self.root_dir}/person/{person_id}.jpg")
            garment = self.load_image_from_s3(f"{self.root_dir}/garment/{garment_id}.jpg")
            person_repr = self.load_image_from_s3(f"{self.root_dir}/person_repr/{person_id}.jpg")
            target = self.load_image_from_s3(f"{self.root_dir}/target/{person_id}_{garment_id}.jpg")
            
            return {
                "person": self.transforms(person),
                "garment": self.transforms(garment),
                "person_repr": self.transforms(person_repr),
                "target": self.transforms(target),
            }
        except Exception as e:
            # Return None to filter
            print(f"⚠️ Error loading index {idx}: {e}")
            return None

def collate_fn_ignore_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)

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
    
    # Use S3VTONDataset if available
    try:
        from train.common.dataset import get_vton_dataset
        from torchvision import transforms
        
        # Define transforms for CP-VTON (usually 256x192, but using config.IMAGE_SIZE)
        transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
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

        dataset = CPVTONDataset(
            root_dir=config.DATASET_ROOT,
            pairs_file=pairs_file,
            size=config.IMAGE_SIZE
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=32,
            collate_fn=collate_fn_ignore_none
        )
