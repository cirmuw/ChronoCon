import torch
import numpy as np
from torch.utils.data import Sampler
from collections import defaultdict
from typing import Iterator, List


class GroupedBatchSampler(Sampler):
    """
    A sampler that groups samples by their 'name' attribute and tries to keep
    samples with the same name in the same batch.
    
    Args:
        dataset: The dataset to sample from. Must return a dict with 'name' key.
        batch_size: The desired batch size.
        drop_last: Whether to drop the last incomplete batch.
        shuffle: Whether to shuffle the groups and samples.
    """
    
    def __init__(self, dataset, batch_size: int, drop_last: bool = False, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # Build a mapping from name to list of indices
        self.name_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            sample = dataset[idx]
            name = sample.get('name', f'unknown_{idx}')
            self.name_to_indices[name].append(idx)
        
        # Convert to regular dict for easier manipulation
        self.name_to_indices = dict(self.name_to_indices)
        self.names = list(self.name_to_indices.keys())
        
    def __iter__(self) -> Iterator[List[int]]:
        """
        Generate batches where samples with the same name are mostly grouped together.
        """
        # Shuffle the order of groups if requested
        if self.shuffle:
            np.random.shuffle(self.names)
        
        # For each group, shuffle the indices within the group
        indices_by_group = []
        for name in self.names:
            group_indices = self.name_to_indices[name].copy()
            if self.shuffle:
                np.random.shuffle(group_indices)
            indices_by_group.append(group_indices)
        
        # Now create batches trying to keep groups together
        batch = []
        
        for group_indices in indices_by_group:
            # If the group is larger than remaining space in current batch,
            # we need to handle it carefully
            group_idx = 0
            
            while group_idx < len(group_indices):
                remaining_in_batch = self.batch_size - len(batch)
                remaining_in_group = len(group_indices) - group_idx
                
                # Add as many from this group as possible to current batch
                take = min(remaining_in_batch, remaining_in_group)
                batch.extend(group_indices[group_idx:group_idx + take])
                group_idx += take
                
                # If batch is full, yield it
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        
        # Handle the last batch
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self) -> int:
        """
        Return the number of batches.
        """
        total_samples = len(self.dataset)
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size


class GroupedRandomSampler(Sampler):
    """
    A simpler sampler that randomly samples indices but with a bias towards
    keeping samples with the same name close together in the ordering.
    
    This version provides more randomness while still having a tendency to
    group samples by name.
    
    Args:
        dataset: The dataset to sample from. Must return a dict with 'name' key.
        shuffle: Whether to shuffle the groups and samples.
    """
    
    def __init__(self, dataset, shuffle: bool = True):
        self.dataset = dataset
        self.shuffle = shuffle
        
        # Build a mapping from name to list of indices
        self.name_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            sample = dataset[idx]
            name = sample.get('name', f'unknown_{idx}')
            self.name_to_indices[name].append(idx)
        
        # Convert to regular dict
        self.name_to_indices = dict(self.name_to_indices)
        self.names = list(self.name_to_indices.keys())
        
    def __iter__(self) -> Iterator[int]:
        """
        Generate indices where samples with the same name tend to be close together.
        """
        # Shuffle the order of groups if requested
        if self.shuffle:
            np.random.shuffle(self.names)
        
        # Collect all indices, grouped by name
        all_indices = []
        for name in self.names:
            group_indices = self.name_to_indices[name].copy()
            if self.shuffle:
                np.random.shuffle(group_indices)
            all_indices.extend(group_indices)
        
        return iter(all_indices)
    
    def __len__(self) -> int:
        return len(self.dataset)

