"""
Example usage of S3 VTON Dataset with different difficulty levels
"""

from train.common.dataset import get_vton_dataset, S3VTONDatasetEasy, S3VTONDatasetMedium, S3VTONDatasetHard
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

# Define transforms
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# S3 prefixes to scan
s3_prefixes = [
    'dataset_ultimate/easy/female/',
    'dataset_ultimate/easy/male/',
    'dataset_ultimate/medium/female/',
    'dataset_ultimate/medium/male/',
    'dataset_ultimate/hard/female/',
    'dataset_ultimate/hard/male/',
]


def example_1_factory_function():
    """Example 1: Using factory function"""
    print("=" * 60)
    print("Example 1: Using factory function")
    print("=" * 60)
    
    # Easy dataset
    dataset_easy = get_vton_dataset(
        difficulty='easy',
        s3_prefixes=s3_prefixes,
        transform=transform
    )
    print(f"Easy dataset size: {len(dataset_easy)}")
    
    # Medium dataset
    dataset_medium = get_vton_dataset(
        difficulty='medium',
        s3_prefixes=s3_prefixes,
        transform=transform
    )
    print(f"Medium dataset size: {len(dataset_medium)}")
    
    # Hard dataset
    dataset_hard = get_vton_dataset(
        difficulty='hard',
        s3_prefixes=s3_prefixes,
        transform=transform
    )
    print(f"Hard dataset size: {len(dataset_hard)}\n")


def example_2_direct_classes():
    """Example 2: Using classes directly"""
    print("=" * 60)
    print("Example 2: Using classes directly")
    print("=" * 60)
    
    # Easy variant
    dataset_easy = S3VTONDatasetEasy(
        s3_prefixes=s3_prefixes,
        transform=transform
    )
    
    # Medium variant
    dataset_medium = S3VTONDatasetMedium(
        s3_prefixes=s3_prefixes,
        transform=transform
    )
    
    # Hard variant
    dataset_hard = S3VTONDatasetHard(
        s3_prefixes=s3_prefixes,
        transform=transform
    )
    
    print(f"Created 3 dataset variants\n")


def example_3_with_dataloader():
    """Example 3: Using with DataLoader"""
    print("=" * 60)
    print("Example 3: Using with DataLoader")
    print("=" * 60)
    
    # Create medium difficulty dataset
    dataset = get_vton_dataset(
        difficulty='medium',
        s3_prefixes=s3_prefixes,
        transform=transform
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"DataLoader created with {len(dataloader)} batches")
    
    # Get one batch
    batch = next(iter(dataloader))
    
    print(f"\nBatch contents:")
    print(f"  - initial_image: {batch['initial_image'].shape}")
    print(f"  - cloth_image: {batch['cloth_image'].shape}")
    print(f"  - try_on_image: {batch['try_on_image'].shape}")
    print(f"  - difficulty: {batch['difficulty']}")
    print(f"  - stem: {batch['stem']}\n")


def example_4_gender_specific():
    """Example 4: Gender-specific datasets"""
    print("=" * 60)
    print("Example 4: Gender-specific datasets")
    print("=" * 60)
    
    # Female only - Easy
    dataset_female_easy = get_vton_dataset(
        difficulty='easy',
        s3_prefixes=[
            'dataset_ultimate/easy/female/',
            'dataset_ultimate/medium/female/',
            'dataset_ultimate/hard/female/',
        ],
        transform=transform
    )
    print(f"Female Easy dataset: {len(dataset_female_easy)} samples")
    
    # Male only - Hard
    dataset_male_hard = get_vton_dataset(
        difficulty='hard',
        s3_prefixes=[
            'dataset_ultimate/easy/male/',
            'dataset_ultimate/medium/male/',
            'dataset_ultimate/hard/male/',
        ],
        transform=transform
    )
    print(f"Male Hard dataset: {len(dataset_male_hard)} samples\n")


def example_5_training_loop():
    """Example 5: Simulated training loop"""
    print("=" * 60)
    print("Example 5: Simulated training loop")
    print("=" * 60)
    
    # Create dataset
    dataset = get_vton_dataset(
        difficulty='medium',
        s3_prefixes=s3_prefixes[:2],  # Use fewer prefixes for demo
        transform=transform
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )
    
    print(f"Training on {len(dataset)} samples")
    print(f"Batch size: 4")
    print(f"Number of batches: {len(dataloader)}\n")
    
    # Simulate training for 3 batches
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        
        person_img = batch['initial_image']
        cloth_img = batch['cloth_image']
        target_img = batch['try_on_image']
        
        print(f"Batch {i+1}:")
        print(f"  - Person images: {person_img.shape}")
        print(f"  - Cloth images: {cloth_img.shape}")
        print(f"  - Target images: {target_img.shape}")
        print(f"  - Difficulties: {batch['difficulty']}")
        
        # Simulate forward pass
        # loss = model(person_img, cloth_img, target_img)
        # loss.backward()
        # optimizer.step()
    
    print("\nTraining simulation complete\n")


def example_6_difficulty_statistics():
    """Example 6: Analyze difficulty distribution"""
    print("=" * 60)
    print("Example 6: Difficulty distribution analysis")
    print("=" * 60)
    
    for difficulty in ['easy', 'medium', 'hard']:
        dataset = get_vton_dataset(
            difficulty=difficulty,
            s3_prefixes=s3_prefixes,
            transform=transform
        )
        
        # Count samples by difficulty
        difficulty_counts = {'easy': 0, 'medium': 0, 'hard': 0}
        
        for i in range(min(len(dataset), 1000)):  # Sample first 1000
            sample = dataset[i]
            difficulty_counts[sample['difficulty']] += 1
        
        print(f"\n{difficulty.upper()} Dataset Distribution (first 1000 samples):")
        total = sum(difficulty_counts.values())
        for diff, count in difficulty_counts.items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  - {diff.capitalize()}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("S3 VTON Dataset Examples")
    print("=" * 60 + "\n")
    
    # Run examples
    try:
        example_1_factory_function()
        example_2_direct_classes()
        example_3_with_dataloader()
        example_4_gender_specific()
        example_5_training_loop()
        example_6_difficulty_statistics()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure AWS credentials are configured and S3 data exists.")
