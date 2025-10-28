"""
Test script to demonstrate and verify the GroupedBatchSampler functionality.
"""

import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path
# Add parent directory to path to import from sibling modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import AgeDB
from samplers.grouped_sampler import GroupedBatchSampler, GroupedRandomSampler
from utils import get_transforms
from collections import Counter


def test_grouped_batch_sampler():
    """Test the GroupedBatchSampler with the AgeDB dataset."""
    
    print("=" * 80)
    print("Testing GroupedBatchSampler")
    print("=" * 80)
    
    # Load dataset
    transform = get_transforms(split='train', aug='')
    dataset = AgeDB(
        data_folder='/home/cwatzenboeck/data/public/agedb/',
        transform=transform,
        split='train'
    )
    
    batch_size = 300  # 8
    
    # Create the grouped batch sampler
    grouped_sampler = GroupedBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=True
    )
    
    # Create DataLoader with the custom batch sampler
    # NOTE: When using a batch_sampler, don't specify batch_size, shuffle, or drop_last
    dataloader = DataLoader(
        dataset,
        batch_sampler=grouped_sampler,
        num_workers=0,  # Use 0 for testing
        pin_memory=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Number of unique names: {len(grouped_sampler.name_to_indices)}")
    print()
    
    # Analyze first few batches
    print("Analyzing first 5 batches:")
    print("-" * 80)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 5:
            break
        
        names = batch['name']
        ages = batch['y_true']
        
        # Count unique names in this batch
        name_counts = Counter(names)
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Batch size: {len(names)}")
        print(f"  Unique names: {len(name_counts)}")
        print(f"  Name distribution: {dict(name_counts)}")
        print(f"  Age range: [{ages.min():.1f}, {ages.max():.1f}]")
    
    # Calculate statistics across all batches
    print("\n" + "=" * 80)
    print("Statistics across all batches:")
    print("-" * 80)
    
    total_names_per_batch = []
    unique_names_per_batch = []
    
    for batch in dataloader:
        names = batch['name']
        name_counts = Counter(names)
        total_names_per_batch.append(len(names))
        unique_names_per_batch.append(len(name_counts))
    
    import numpy as np
    print(f"Average batch size: {np.mean(total_names_per_batch):.2f}")
    print(f"Average unique names per batch: {np.mean(unique_names_per_batch):.2f}")
    print(f"Min unique names per batch: {np.min(unique_names_per_batch)}")
    print(f"Max unique names per batch: {np.max(unique_names_per_batch)}")
    print()


def test_grouped_random_sampler():
    """Test the GroupedRandomSampler with the AgeDB dataset."""
    
    print("=" * 80)
    print("Testing GroupedRandomSampler (used with regular DataLoader)")
    print("=" * 80)
    
    # Load dataset
    transform = get_transforms(split='train', aug='')
    dataset = AgeDB(
        data_folder='/home/cwatzenboeck/data/public/agedb/',
        transform=transform,
        split='train'
    )
    
    batch_size = 300
    
    # Create the grouped random sampler
    grouped_sampler = GroupedRandomSampler(
        dataset=dataset,
        shuffle=True
    )
    
    # Create DataLoader with the custom sampler
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=grouped_sampler,  # Use sampler instead of batch_sampler
        num_workers=0,
        pin_memory=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {len(dataloader)}")
    print()
    
    # Analyze first few batches
    print("Analyzing first 5 batches:")
    print("-" * 80)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 5:
            break
        
        names = batch['name']
        ages = batch['y_true']
        
        # Count unique names in this batch
        name_counts = Counter(names)
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Batch size: {len(names)}")
        print(f"  Unique names: {len(name_counts)}")
        print(f"  Name distribution: {dict(name_counts)}")
    
    print("\n" + "=" * 80)


def compare_samplers():
    """Compare standard random sampler vs grouped sampler."""
    
    print("=" * 80)
    print("Comparing Standard vs Grouped Sampler")
    print("=" * 80)
    
    transform = get_transforms(split='train', aug='')
    dataset = AgeDB(
        data_folder='/home/cwatzenboeck/data/public/agedb/',
        transform=transform,
        split='train'
    )
    
    batch_size = 300 # 8
    
    # Standard sampler
    print("\n1. STANDARD RANDOM SAMPLER:")
    print("-" * 80)
    
    standard_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    unique_names_standard = []
    for batch_idx, batch in enumerate(standard_loader):
        if batch_idx >= 20:
            break
        names = batch['name']
        name_counts = Counter(names)
        unique_names_standard.append(len(name_counts))
    
    import numpy as np
    print(f"Average unique names per batch: {np.mean(unique_names_standard):.2f}")
    print(f"Std dev: {np.std(unique_names_standard):.2f}")
    
    # Grouped sampler
    print("\n2. GROUPED BATCH SAMPLER:")
    print("-" * 80)
    
    grouped_sampler = GroupedBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=True
    )
    
    grouped_loader = DataLoader(
        dataset,
        batch_sampler=grouped_sampler,
        num_workers=0
    )
    
    unique_names_grouped = []
    for batch_idx, batch in enumerate(grouped_loader):
        if batch_idx >= 20:
            break
        names = batch['name']
        name_counts = Counter(names)
        unique_names_grouped.append(len(name_counts))
    
    print(f"Average unique names per batch: {np.mean(unique_names_grouped):.2f}")
    print(f"Std dev: {np.std(unique_names_grouped):.2f}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print(f"The grouped sampler reduces unique names per batch from "
          f"{np.mean(unique_names_standard):.2f} to {np.mean(unique_names_grouped):.2f}")
    print(f"This means samples with the same name are {np.mean(unique_names_standard) / np.mean(unique_names_grouped):.2f}x "
          f"more likely to be in the same batch!")
    print("=" * 80)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == 'batch':
            test_grouped_batch_sampler()
        elif test_type == 'random':
            test_grouped_random_sampler()
        elif test_type == 'compare':
            compare_samplers()
        else:
            print("Unknown test type. Use 'batch', 'random', or 'compare'")
    else:
        # Run all tests
        test_grouped_batch_sampler()
        print("\n\n")
        test_grouped_random_sampler()
        print("\n\n")
        compare_samplers()


