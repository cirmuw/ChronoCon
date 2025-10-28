# Grouped Sampler for PyTorch DataLoader

This module provides custom samplers that group samples by their "name" attribute, keeping samples with the same name mostly in the same batch while maintaining randomness.

## Overview

The module provides two sampler types:

1. **GroupedBatchSampler**: A batch sampler that handles batching internally and tries to keep all samples with the same name in the same batch.

2. **GroupedRandomSampler**: A simpler sampler that randomizes the order but groups samples by name, letting DataLoader handle the batching.

## Why Use Grouped Sampling?

Grouped sampling is useful when:
- You have multiple samples (images) per subject/person/entity
- You want to reduce data leakage between batches during training
- You want batch statistics (like BatchNorm) to be computed on more diverse samples
- You want samples from the same entity to be processed together for certain training techniques

## Installation

No installation needed - just copy `grouped_sampler.py` to your project directory.

## Usage

### Option 1: GroupedBatchSampler (Recommended)

This sampler provides the best grouping behavior as it controls the entire batching process:

```python
from torch.utils.data import DataLoader
from dataset import AgeDB
from grouped_sampler import GroupedBatchSampler

# Create your dataset
dataset = AgeDB(data_folder='...', transform=transform, split='train')

# Create the grouped batch sampler
sampler = GroupedBatchSampler(
    dataset=dataset,
    batch_size=32,
    drop_last=False,
    shuffle=True
)

# Create DataLoader with batch_sampler (don't specify batch_size again!)
train_loader = DataLoader(
    dataset,
    batch_sampler=sampler,  # Use batch_sampler, not sampler
    num_workers=4,
    pin_memory=True
)
```

**Important**: When using `batch_sampler`, do NOT specify `batch_size`, `shuffle`, or `drop_last` in the DataLoader constructor.

### Option 2: GroupedRandomSampler

This sampler only controls the order of samples, letting DataLoader handle batching:

```python
from torch.utils.data import DataLoader
from dataset import AgeDB
from grouped_sampler import GroupedRandomSampler

# Create your dataset
dataset = AgeDB(data_folder='...', transform=transform, split='train')

# Create the grouped random sampler
sampler = GroupedRandomSampler(
    dataset=dataset,
    shuffle=True
)

# Create DataLoader with sampler (specify batch_size as normal)
train_loader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler,  # Use sampler, not batch_sampler
    num_workers=4,
    pin_memory=True
)
```

**Note**: Don't specify `shuffle=True` in DataLoader when using a custom sampler.

## Integration with Existing Code

### Minimal Change Example

To integrate into your existing training script, modify the loader creation:

```python
# Before (standard approach):
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=opt.batch_size, 
    shuffle=True, 
    num_workers=opt.num_workers, 
    pin_memory=True
)

# After (with grouped sampler):
from grouped_sampler import GroupedBatchSampler

train_sampler = GroupedBatchSampler(
    dataset=train_dataset,
    batch_size=opt.batch_size,
    drop_last=False,
    shuffle=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_sampler=train_sampler,
    num_workers=opt.num_workers, 
    pin_memory=True
)
```

### With Command Line Argument

See `main_l1_with_grouped_sampler.py` for a complete example with argparse integration:

```bash
# Train with standard sampler
python main_l1_with_grouped_sampler.py --batch_size 32

# Train with grouped sampler
python main_l1_with_grouped_sampler.py --batch_size 32 --use_grouped_sampler --sampler_type batch
```

## Testing the Samplers

Run the test script to see how the samplers work:

```bash
# Run all tests
python test_grouped_sampler.py

# Run specific test
python test_grouped_sampler.py batch      # Test GroupedBatchSampler
python test_grouped_sampler.py random     # Test GroupedRandomSampler
python test_grouped_sampler.py compare    # Compare standard vs grouped
```

The test script will show:
- How many unique names appear in each batch
- Statistics on grouping effectiveness
- Comparison between standard and grouped sampling

## How It Works

### GroupedBatchSampler

1. **Initialization**: 
   - Scans the entire dataset and builds a mapping from "name" to list of indices
   - Stores all unique names

2. **Each Epoch**:
   - Shuffles the order of name groups (if shuffle=True)
   - Shuffles indices within each group (if shuffle=True)
   - Yields batches by taking samples from groups sequentially
   - Tries to keep all samples from the same group in the same batch
   - If a group is larger than batch_size, it may span multiple batches

3. **Example**:
   ```
   Groups: 
   - "person_A": [0, 1, 2, 3, 4]
   - "person_B": [5, 6]
   - "person_C": [7, 8, 9]
   
   With batch_size=4:
   Batch 1: [0, 1, 2, 3]        # All from person_A
   Batch 2: [4, 5, 6, 7]        # person_A + person_B + person_C
   Batch 3: [8, 9, ...]         # Rest of person_C + next groups
   ```

### GroupedRandomSampler

1. Shuffles the order of name groups
2. Shuffles indices within each group
3. Concatenates all indices in group order
4. DataLoader then batches these indices sequentially

This approach is simpler but provides slightly less control over exact batch composition.

## Performance Considerations

- **Initialization**: Both samplers scan the entire dataset once during initialization. For large datasets, this adds a small one-time cost at the start of training.

- **Memory**: The samplers store a mapping from names to indices, which requires O(N) memory where N is the dataset size.

- **Speed**: No impact on training speed after initialization. The samplers are deterministic and efficient.

## Requirements

Your dataset must:
- Return a dictionary or object with a 'name' attribute when indexed
- Have a `__len__` method

Example dataset structure:
```python
class MyDataset(data.Dataset):
    def __getitem__(self, index):
        return {
            'image': ...,
            'y_true': ...,
            'name': ...    # Required for grouped sampler
        }
```

## Troubleshooting

### "KeyError: 'name'"
Your dataset doesn't return a 'name' field. Modify your dataset's `__getitem__` method to include the name.

### Batches are still not grouped well
- Make sure you're using `batch_sampler` (not `sampler`) with GroupedBatchSampler
- Check that your dataset actually has multiple samples per name
- Try using GroupedBatchSampler instead of GroupedRandomSampler for better grouping

### Different results each epoch
This is expected! The samplers shuffle the group order and samples within groups each epoch to maintain randomness while keeping groups together.

## Comparison: Standard vs Grouped Sampling

```
Standard Random Sampling:
├─ Batch 1: [person_A, person_B, person_C, person_D, person_E, person_F, person_G, person_H]
├─ Batch 2: [person_B, person_A, person_E, person_I, person_J, person_C, person_K, person_L]
└─ Batch 3: [person_M, person_D, person_N, person_O, person_F, person_P, person_Q, person_R]
Average unique names per batch: ~7.8

Grouped Sampling:
├─ Batch 1: [person_A, person_A, person_A, person_B, person_B, person_B, person_B, person_B]
├─ Batch 2: [person_C, person_C, person_C, person_C, person_D, person_D, person_D, person_D]
└─ Batch 3: [person_E, person_E, person_E, person_E, person_F, person_F, person_F, person_F]
Average unique names per batch: ~2.0
```

The grouped sampling reduces the number of unique names per batch by ~4x, meaning samples with the same name are much more likely to appear together!

## License

Same as the parent project.

## Questions?

For questions or issues, please open an issue in the project repository.

