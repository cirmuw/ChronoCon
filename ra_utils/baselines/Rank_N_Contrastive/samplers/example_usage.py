"""
Quick Start Example: Using Grouped Sampler

This is a minimal working example showing how to use the grouped sampler.
Run this script to see it in action!
"""

import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path
# Add parent directory to path to import from sibling modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import AgeDB
from samplers.grouped_sampler import GroupedBatchSampler
from utils import get_transforms


def main():
    print("\n" + "="*80)
    print("GROUPED SAMPLER - QUICK START EXAMPLE")
    print("="*80 + "\n")
    
    # Step 1: Create your dataset (same as before)
    print("Step 1: Creating dataset...")
    transform = get_transforms(split='train', aug='crop,flip')
    dataset = AgeDB(
        data_folder='/home/cwatzenboeck/data/public/agedb/',
        transform=transform,
        split='train'
    )
    print(f"✓ Dataset created with {len(dataset)} samples\n")
    
    # Step 2: Create the grouped sampler
    print("Step 2: Creating GroupedBatchSampler...")
    batch_size = 16
    sampler = GroupedBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=True
    )
    print(f"✓ Sampler created")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Number of unique names: {len(sampler.name_to_indices)}")
    print(f"  - Shuffle: True\n")
    
    # Step 3: Create DataLoader with the sampler
    print("Step 3: Creating DataLoader with grouped sampler...")
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,  # <-- Key difference!
        num_workers=2,
        pin_memory=True
    )
    print(f"✓ DataLoader created with {len(dataloader)} batches\n")
    
    # Step 4: Iterate and show results
    print("Step 4: Iterating through first 3 batches...")
    print("-"*80)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 3:
            break
        
        images = batch['image']
        ages = batch['y_true']
        names = batch['name']
        
        # Count unique names in this batch
        unique_names = list(set(names))
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Ages range: [{ages.min():.1f}, {ages.max():.1f}]")
        print(f"  Unique names in batch: {len(unique_names)} out of {len(names)} samples")
        print(f"  Grouping efficiency: {(1 - len(unique_names)/len(names))*100:.1f}%")
        
        # Show which names appear and how many times
        from collections import Counter
        name_counts = Counter(names)
        print(f"  Name distribution: {dict(list(name_counts.items())[:3])}...")
    
    print("\n" + "="*80)
    print("COMPARISON: Standard vs Grouped Sampling")
    print("="*80)
    
    # Standard dataloader for comparison
    print("\nStandard DataLoader (shuffle=True):")
    standard_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    # Count unique names in first 10 batches
    unique_counts = []
    for batch_idx, batch in enumerate(standard_loader):
        if batch_idx >= 10:
            break
        names = batch['name']
        unique_counts.append(len(set(names)))
    
    import numpy as np
    print(f"  Average unique names per batch: {np.mean(unique_counts):.2f}")
    
    # Grouped dataloader
    print("\nGrouped DataLoader (GroupedBatchSampler):")
    unique_counts_grouped = []
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 10:
            break
        names = batch['name']
        unique_counts_grouped.append(len(set(names)))
    
    print(f"  Average unique names per batch: {np.mean(unique_counts_grouped):.2f}")
    
    print("\n" + "="*80)
    improvement = np.mean(unique_counts) / np.mean(unique_counts_grouped)
    print(f"✓ Grouped sampler is {improvement:.1f}x better at keeping same names together!")
    print("="*80 + "\n")
    
    # Training loop example
    print("="*80)
    print("INTEGRATION IN TRAINING LOOP")
    print("="*80)
    print("""
Your training loop remains exactly the same:

for epoch in range(num_epochs):
    for batch in train_loader:
        images = batch['image']
        labels = batch['y_true']
        
        # Your training code here
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
The only change is HOW you create train_loader (see above)!
    """)
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

