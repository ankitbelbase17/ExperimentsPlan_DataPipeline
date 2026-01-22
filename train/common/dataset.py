"""
S3 VTON Dataset with Difficulty-based Sampling

This module provides three dataset variants:
- Easy: 100% easy samples
- Medium: 30% easy + 70% medium samples
- Hard: 25% easy + 25% medium + 50% hard samples
"""

import torch
from torch.utils.data import Dataset
import boto3
import logging
from PIL import Image
from io import BytesIO
import os
from collections import defaultdict
import random

try:
    from src.config import S3_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME
except ImportError:
    # Fallback if running outside of package context
    S3_REGION = os.getenv("S3_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

logger = logging.getLogger(__name__)


class S3VTONDataset(Dataset):
    """
    Base S3 VTON Dataset that discovers image triplets from S3 directories.
    
    Expected S3 structure:
    - {prefix}/initial_image/{stem}_person.png
    - {prefix}/cloth_image/{stem}_cloth_*.png
    - {prefix}/try_on_image/{stem}_vton.png
    """
    
    def __init__(self, s3_prefixes, transform=None, s3_bucket=None, difficulty_weights=None):
        """
        Args:
            s3_prefixes (list): List of S3 directory prefixes to scan for images.
                               Example: ["dataset_ultimate/easy/female/", "dataset_ultimate/medium/male/"]
                               Or can be s3:// URIs: ["s3://bucket/dataset_ultimate/easy/female/"]
            transform (callable, optional): PyTorch transforms for images.
            s3_bucket (str, optional): Default bucket name if s3:// path doesn't specify.
            difficulty_weights (dict, optional): Weights for sampling from different difficulties.
                                                Format: {"easy": 0.3, "medium": 0.7, "hard": 0.0}
        """
        self.transform = transform
        self.data = []
        self.s3_client = None  # Lazy init per worker
        self.s3_bucket = s3_bucket or S3_BUCKET_NAME
        self.difficulty_weights = difficulty_weights or {}
        
        # Store data by difficulty level
        self.data_by_difficulty = {
            "easy": [],
            "medium": [],
            "hard": []
        }
        
        # 1. Scan S3 directories and build metadata
        self._load_metadata_from_s3(s3_prefixes)
        
        # 2. Apply difficulty-based sampling if weights provided
        if self.difficulty_weights:
            self._apply_difficulty_sampling()
        
    def _init_s3(self):
        """Initialize boto3 client per-worker to avoid fork safety issues"""
        if self.s3_client is None:
            self.s3_client = boto3.client(
                's3',
                region_name=S3_REGION,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY
            )

    def _load_metadata_from_s3(self, s3_prefixes):
        """
        Scans S3 directories to discover image triplets.
        Expected structure:
        - {prefix}/initial_image/{stem}_person.png
        - {prefix}/cloth_image/{stem}_cloth_*.png
        - {prefix}/try_on_image/{stem}_vton.png
        """
        # Initialize a temporary client just for metadata loading
        temp_client = boto3.client(
            's3',
            region_name=S3_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        
        for prefix in s3_prefixes:
            print(f"Scanning S3 directory: {prefix}...")
            
            # Parse bucket and prefix
            bucket, base_prefix = self._parse_s3_path(prefix)
            
            # Ensure prefix ends with /
            if not base_prefix.endswith('/'):
                base_prefix += '/'
            
            # Detect difficulty level from path
            difficulty = self._detect_difficulty(base_prefix)
            
            # Dictionary to group files by stem
            # stem -> {"initial": key, "cloth": key, "try_on": key}
            triplets = defaultdict(dict)
            
            # Scan all three subdirectories
            for image_type in ['initial_image', 'cloth_image', 'try_on_image']:
                scan_prefix = f"{base_prefix}{image_type}/"
                
                try:
                    paginator = temp_client.get_paginator('list_objects_v2')
                    for page in paginator.paginate(Bucket=bucket, Prefix=scan_prefix):
                        if 'Contents' not in page:
                            continue
                            
                        for obj in page['Contents']:
                            key = obj['Key']
                            
                            # Skip if not an image
                            if not key.lower().endswith(('.png', '.jpg', '.jpeg')):
                                continue
                            
                            # Extract filename and stem
                            filename = os.path.basename(key)
                            stem = self._extract_stem(filename, image_type)
                            
                            if stem:
                                triplets[stem][image_type] = f"s3://{bucket}/{key}"
                                
                except Exception as e:
                    logger.error(f"Failed to scan {scan_prefix}: {e}")
            
            # Filter complete triplets (must have all three images)
            for stem, images in triplets.items():
                if len(images) == 3:  # Has all three image types
                    sample = {
                        "initial_image": images['initial_image'],
                        "cloth_image": images['cloth_image'],
                        "try_on_image": images['try_on_image'],
                        "stem": stem,
                        "difficulty": difficulty,
                        "prefix": base_prefix
                    }
                    
                    # Add to both main data and difficulty-specific data
                    self.data.append(sample)
                    self.data_by_difficulty[difficulty].append(sample)
                else:
                    logger.warning(f"Incomplete triplet for stem '{stem}': {images.keys()}")
        
        print(f"Loaded {len(self.data)} complete training samples from S3.")
        print(f"  - Easy: {len(self.data_by_difficulty['easy'])} samples")
        print(f"  - Medium: {len(self.data_by_difficulty['medium'])} samples")
        print(f"  - Hard: {len(self.data_by_difficulty['hard'])} samples")

    def _detect_difficulty(self, prefix):
        """Detect difficulty level from S3 prefix path"""
        prefix_lower = prefix.lower()
        if '/easy/' in prefix_lower or prefix_lower.endswith('easy/'):
            return 'easy'
        elif '/medium/' in prefix_lower or prefix_lower.endswith('medium/'):
            return 'medium'
        elif '/hard/' in prefix_lower or prefix_lower.endswith('hard/'):
            return 'hard'
        else:
            # Default to easy if not specified
            logger.warning(f"Could not detect difficulty from prefix: {prefix}, defaulting to 'easy'")
            return 'easy'

    def _apply_difficulty_sampling(self):
        """
        Apply difficulty-based sampling according to weights.
        Reconstructs self.data based on difficulty_weights.
        """
        if not self.difficulty_weights:
            return
        
        print(f"\nApplying difficulty-based sampling with weights: {self.difficulty_weights}")
        
        # Calculate target counts
        total_samples = len(self.data)
        target_counts = {}
        
        for difficulty, weight in self.difficulty_weights.items():
            target_counts[difficulty] = int(total_samples * weight)
        
        # Sample from each difficulty level
        sampled_data = []
        
        for difficulty, target_count in target_counts.items():
            available = self.data_by_difficulty[difficulty]
            
            if len(available) == 0:
                logger.warning(f"No samples available for difficulty '{difficulty}'")
                continue
            
            if target_count > len(available):
                # If we need more samples than available, sample with replacement
                logger.warning(
                    f"Target count ({target_count}) exceeds available samples ({len(available)}) "
                    f"for difficulty '{difficulty}'. Sampling with replacement."
                )
                sampled = random.choices(available, k=target_count)
            else:
                # Sample without replacement
                sampled = random.sample(available, target_count)
            
            sampled_data.extend(sampled)
            print(f"  - Sampled {len(sampled)} from {difficulty} (target: {target_count}, available: {len(available)})")
        
        # Shuffle the combined samples
        random.shuffle(sampled_data)
        
        # Replace data with sampled data
        self.data = sampled_data
        
        print(f"Final dataset size after sampling: {len(self.data)} samples\n")

    def _extract_stem(self, filename, image_type):
        """
        Extract the stem (base identifier) from filename based on image type.
        - initial_image: {stem}_person.png -> stem
        - cloth_image: {stem}_cloth_*.png -> stem
        - try_on_image: {stem}_vton.png -> stem
        """
        # Remove extension
        name_no_ext = os.path.splitext(filename)[0]
        
        if image_type == 'initial_image':
            # Format: {stem}_person
            if name_no_ext.endswith('_person'):
                return name_no_ext[:-7]  # Remove '_person'
        elif image_type == 'cloth_image':
            # Format: {stem}_cloth_{original_name}
            if '_cloth_' in name_no_ext:
                parts = name_no_ext.split('_cloth_', 1)
                return parts[0]
        elif image_type == 'try_on_image':
            # Format: {stem}_vton
            if name_no_ext.endswith('_vton'):
                return name_no_ext[:-5]  # Remove '_vton'
        
        return None

    def _parse_s3_path(self, path):
        """
        Parses s3://bucket/key or just key if default bucket provided.
        Returns (bucket, key)
        """
        if path.startswith("s3://"):
            # s3://bucket/key/path...
            path_no_scheme = path[5:]
            parts = path_no_scheme.split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            return bucket, key
        else:
            # Use default bucket
            return self.s3_bucket, path

    def _download_image(self, s3_uri):
        """Download image from S3"""
        bucket, key = self._parse_s3_path(s3_uri)
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            image_data = response['Body'].read()
            image = Image.open(BytesIO(image_data)).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error downloading {s3_uri}: {e}")
            return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self._init_s3()  # Ensure client exists
        
        item = self.data[idx]
        
        # Format: { "initial_image": ..., "cloth_image": ..., "try_on_image": ... }
        person_uri = item['initial_image']
        cloth_uri = item['cloth_image']
        target_uri = item['try_on_image']
        
        person_img = self._download_image(person_uri)
        cloth_img = self._download_image(cloth_uri)
        target_img = self._download_image(target_uri)
        
        # Handle failures
        if person_img is None or cloth_img is None or target_img is None:
            # Return None to be filtered by collate_fn
            logger.warning(f"Failed to load images for index {idx}, stem: {item.get('stem', 'unknown')}")
            return None

        if self.transform:
            person_img = self.transform(person_img)
            cloth_img = self.transform(cloth_img)
            target_img = self.transform(target_img)
            
        return {
            "initial_image": person_img,
            "cloth_image": cloth_img,
            "try_on_image": target_img,
            "difficulty": item['difficulty'],
            "stem": item['stem']
        }


class S3VTONDatasetEasy(S3VTONDataset):
    """
    Easy variant: 100% easy samples
    """
    def __init__(self, s3_prefixes, transform=None, s3_bucket=None):
        difficulty_weights = {
            "easy": 1.0,
            "medium": 0.0,
            "hard": 0.0
        }
        super().__init__(
            s3_prefixes=s3_prefixes,
            transform=transform,
            s3_bucket=s3_bucket,
            difficulty_weights=difficulty_weights
        )
        print("✅ Initialized Easy Dataset: 100% easy samples")


class S3VTONDatasetMedium(S3VTONDataset):
    """
    Medium variant: 30% easy + 70% medium samples
    """
    def __init__(self, s3_prefixes, transform=None, s3_bucket=None):
        difficulty_weights = {
            "easy": 0.30,
            "medium": 0.70,
            "hard": 0.0
        }
        super().__init__(
            s3_prefixes=s3_prefixes,
            transform=transform,
            s3_bucket=s3_bucket,
            difficulty_weights=difficulty_weights
        )
        print("✅ Initialized Medium Dataset: 30% easy + 70% medium samples")


class S3VTONDatasetHard(S3VTONDataset):
    """
    Hard variant: 25% easy + 25% medium + 50% hard samples
    """
    def __init__(self, s3_prefixes, transform=None, s3_bucket=None):
        difficulty_weights = {
            "easy": 0.25,
            "medium": 0.25,
            "hard": 0.50
        }
        super().__init__(
            s3_prefixes=s3_prefixes,
            transform=transform,
            s3_bucket=s3_bucket,
            difficulty_weights=difficulty_weights
        )
        print("✅ Initialized Hard Dataset: 25% easy + 25% medium + 50% hard samples")


# Convenience factory function
def get_vton_dataset(difficulty='easy', s3_prefixes=None, transform=None, s3_bucket=None):
    """
    Factory function to create VTON dataset with specified difficulty.
    
    Args:
        difficulty (str): One of 'easy', 'medium', 'hard'
        s3_prefixes (list): List of S3 prefixes to scan
        transform: PyTorch transforms
        s3_bucket (str): S3 bucket name
    
    Returns:
        S3VTONDataset: Dataset instance with appropriate difficulty sampling
    
    Example:
        >>> dataset = get_vton_dataset(
        ...     difficulty='medium',
        ...     s3_prefixes=[
        ...         'dataset_ultimate/easy/female/',
        ...         'dataset_ultimate/medium/female/',
        ...         'dataset_ultimate/hard/female/'
        ...     ],
        ...     transform=my_transforms
        ... )
    """
    if difficulty == 'easy':
        return S3VTONDatasetEasy(s3_prefixes, transform, s3_bucket)
    elif difficulty == 'medium':
        return S3VTONDatasetMedium(s3_prefixes, transform, s3_bucket)
    elif difficulty == 'hard':
        return S3VTONDatasetHard(s3_prefixes, transform, s3_bucket)
    else:
        raise ValueError(f"Invalid difficulty: {difficulty}. Must be 'easy', 'medium', or 'hard'")
