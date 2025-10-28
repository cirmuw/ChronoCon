"""
Training utilities for Rank_N_Contrastive baselines.

This module provides generic training and validation loop functions that handle
boilerplate code (timing, logging, loss computation) while allowing flexible
loss functions that can use sample metadata (like 'name').
"""

import sys
import time
import torch
from utils import AverageMeter
from samplers import GroupedBatchSampler, GroupedRandomSampler


def create_dataloader_with_sampler(dataset, opt, split='train', drop_last=False, 
                                   use_grouped_override=None):
    """
    Create a DataLoader with optional grouped sampling support.
    
    Args:
        dataset: PyTorch Dataset
        opt: Options object with sampler configuration
        split: 'train', 'val', or 'test'
        drop_last: Whether to drop the last incomplete batch
        use_grouped_override: If provided, overrides the grouped sampler setting
                             (useful for test set which may have different behavior)
        
    Returns:
        DataLoader with appropriate sampler
    """
    # Determine if we should use grouped sampler
    if use_grouped_override is not None:
        use_grouped = use_grouped_override
    else:
        use_grouped = opt.use_grouped_sampler if split == 'train' else getattr(opt, 'use_grouped_sampler_val', False)
    
    if use_grouped:
        if split == 'train':
            print(f"\nUsing Grouped Sampler for {split} (type: {opt.sampler_type})")
        else:
            print(f"\nUsing Grouped Sampler for {split} (deterministic, no shuffling)")
        
        if opt.sampler_type == 'batch':
            sampler = GroupedBatchSampler(
                dataset=dataset,
                batch_size=opt.batch_size,
                drop_last=drop_last,
                shuffle=(split == 'train')
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=opt.num_workers,
                pin_memory=True
            )
        else:  # 'random'
            sampler = GroupedRandomSampler(
                dataset=dataset,
                shuffle=(split == 'train')
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=opt.batch_size,
                sampler=sampler,
                num_workers=opt.num_workers,
                pin_memory=True,
                drop_last=drop_last
            )
        
        # Print sampler statistics
        if hasattr(sampler, 'name_to_indices'):
            num_groups = len(sampler.name_to_indices)
            group_sizes = [len(indices) for indices in sampler.name_to_indices.values()]
            print(f"{split.capitalize()} - Number of name groups: {num_groups}")
            print(f"{split.capitalize()} - Average samples per name: {sum(group_sizes) / num_groups:.2f}")
            print(f"{split.capitalize()} - Min/Max samples per name: {min(group_sizes)}/{max(group_sizes)}")
    else:
        if split == 'train':
            print(f"\nUsing Standard Random Sampler for {split}")
        else:
            print(f"\nUsing Standard Sampler for {split}")
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=(split == 'train'),
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=drop_last
        )
    
    return loader


def train_epoch(train_loader, compute_loss_fn, optimizer, epoch, opt, print_fn=print):
    """
    Generic training loop for one epoch.
    
    Args:
        train_loader: DataLoader for training data
        compute_loss_fn: Function that computes loss. Should have signature:
                        loss, batch_size = compute_loss_fn(batch)
                        This function is responsible for:
                        - Extracting data from batch
                        - Forward pass
                        - Computing loss (can use batch['name'] if needed)
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        opt: Options object with print_freq
        print_fn: Function to use for printing (defaults to print)
        
    Returns:
        Average loss for the epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    end = time.time()
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        # Compute loss using the provided function
        loss, bsz = compute_loss_fn(batch)
        losses.update(loss.item(), bsz)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update timing
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Print progress
        if (idx + 1) % opt.print_freq == 0:
            to_print = 'Train: [{0}][{1}/{2}]\t'\
                       'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                       'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'\
                       'loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses
            )
            print_fn(to_print)
            sys.stdout.flush()
    
    return losses.avg


def validate_epoch(val_loader, compute_loss_fn, print_fn=print):
    """
    Generic validation loop.
    
    Args:
        val_loader: DataLoader for validation data
        compute_loss_fn: Function that computes validation loss. Should have signature:
                        loss, batch_size = compute_loss_fn(batch)
                        This function is responsible for:
                        - Extracting data from batch
                        - Forward pass (with torch.no_grad() already set)
                        - Computing loss
        print_fn: Function to use for printing (defaults to print)
        
    Returns:
        Average validation loss
    """
    losses = AverageMeter()
    
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            # Compute loss using the provided function
            loss, bsz = compute_loss_fn(batch)
            losses.update(loss.item(), bsz)
    
    return losses.avg

