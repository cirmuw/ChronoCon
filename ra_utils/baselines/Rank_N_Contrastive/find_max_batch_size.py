#!/usr/bin/env python
"""
Script to find the maximum batch size that fits in GPU memory for the model.
"""
import argparse
import torch
import torch.nn as nn
from model import SupResNet
from utils import get_label_dim


def parse_args():
    parser = argparse.ArgumentParser('Find maximum batch size')
    parser.add_argument('--lx', type=int, default=224, help='image width')
    parser.add_argument('--ly', type=int, default=224, help='image height')
    parser.add_argument('--model', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet50'], help='model architecture')
    parser.add_argument('--dataset', type=str, default='AgeDB', 
                        choices=['AgeDB'], help='dataset')
    parser.add_argument('--start_batch', type=int, default=1, 
                        help='starting batch size')
    parser.add_argument('--max_batch', type=int, default=1024, 
                        help='maximum batch size to test')
    parser.add_argument('--increment', type=int, default=1, 
                        help='batch size increment for initial search')
    parser.add_argument('--runs_per_batch', type=int, default=3,
                        help='number of forward+backward passes per batch size test')
    return parser.parse_args()


def test_batch_size(model, criterion, batch_size, lx, ly, runs=3):
    """
    Test if a given batch size fits in memory.
    
    Args:
        model: The model to test
        criterion: Loss function
        batch_size: Batch size to test
        lx: Image width
        ly: Image height
        runs: Number of forward+backward passes to ensure stability
    
    Returns:
        bool: True if batch size fits, False otherwise
    """
    try:
        torch.cuda.empty_cache()
        model.train()
        
        for _ in range(runs):
            # Create dummy input
            dummy_input = torch.randn(batch_size, 3, ly, lx).cuda()
            dummy_labels = torch.randn(batch_size, 1).cuda()
            
            # Forward pass
            output = model(dummy_input)
            loss = criterion(output, dummy_labels)
            
            # Backward pass
            loss.backward()
            
            # Clear gradients
            model.zero_grad()
            
            del dummy_input, dummy_labels, output, loss
            torch.cuda.empty_cache()
        
        return True
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return False
        else:
            raise e


def binary_search_batch_size(model, criterion, low, high, lx, ly, runs=3):
    """
    Use binary search to find maximum batch size.
    
    Args:
        model: The model to test
        criterion: Loss function
        low: Minimum batch size that works
        high: Maximum batch size to test
        lx: Image width
        ly: Image height
        runs: Number of forward+backward passes per test
    
    Returns:
        int: Maximum working batch size
    """
    max_working = low
    
    while low <= high:
        mid = (low + high) // 2
        print(f"Testing batch size: {mid}...", end=" ", flush=True)
        
        if test_batch_size(model, criterion, mid, lx, ly, runs):
            print("✓ Success")
            max_working = mid
            low = mid + 1
        else:
            print("✗ Out of memory")
            high = mid - 1
    
    return max_working


def main():
    args = parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a GPU.")
        return
    
    print("="*80)
    print("Finding Maximum Batch Size")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Image size: {args.lx}x{args.ly}")
    print(f"Dataset: {args.dataset}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("="*80)
    
    # Setup model
    model = SupResNet(name=args.model, num_classes=get_label_dim(args.dataset))
    criterion = nn.L1Loss()
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model.encoder = nn.DataParallel(model.encoder)
    
    model = model.cuda()
    criterion = criterion.cuda()
    
    # Phase 1: Quick incremental search to find upper bound
    print("\nPhase 1: Quick search to find upper bound...")
    current_batch = args.start_batch
    last_working = args.start_batch
    
    while current_batch <= args.max_batch:
        print(f"Testing batch size: {current_batch}...", end=" ", flush=True)
        
        if test_batch_size(model, criterion, current_batch, args.lx, args.ly, args.runs_per_batch):
            print("✓ Success")
            last_working = current_batch
            current_batch *= 2  # Double the batch size for quick search
        else:
            print("✗ Out of memory")
            break
    
    # Phase 2: Binary search between last working and first failing
    print(f"\nPhase 2: Binary search between {last_working} and {current_batch}...")
    max_batch_size = binary_search_batch_size(
        model, criterion, last_working, current_batch - 1, 
        args.lx, args.ly, args.runs_per_batch
    )
    
    print("\n" + "="*80)
    print(f"Maximum batch size: {max_batch_size}")
    print("="*80)
    print(f"\nRecommendation: Use batch_size={max_batch_size} or slightly smaller")
    print(f"                (e.g., {int(max_batch_size * 0.9)}) to leave headroom")
    print("="*80)


if __name__ == '__main__':
    main()

