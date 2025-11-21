########################################################################################
# Code is based on the LDS and FDS (https://arxiv.org/pdf/2102.09554.pdf) implementation
# from https://github.com/YyzHarry/imbalanced-regression/tree/main/imdb-wiki-dir 
# by Yuzhe Yang et al.
########################################################################################
import os
import shutil
import torch
import logging
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def query_yes_no(question):
    """ Ask a yes/no question via input() and return their answer. """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = " [Y/n] "

    while True:
        print(question + prompt, end=':')
        choice = input().lower()
        if choice == '':
            return valid['y']
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def prepare_folders(args):
    folders_util = [args.store_root, os.path.join(args.store_root, args.store_name)]
    if os.path.exists(folders_util[-1]) and not args.resume and not args.pretrained and not args.evaluate:
        if query_yes_no('overwrite previous folder: {} ?'.format(folders_util[-1])):
            shutil.rmtree(folders_util[-1])
            print(folders_util[-1] + ' removed.')
        else:
            raise RuntimeError('Output folder {} already exists'.format(folders_util[-1]))
    for folder in folders_util:
        if not os.path.exists(folder):
            print(f"===> Creating folder: {folder}")
            os.mkdir(folder)


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(args, state, is_best, prefix=''):
    filename = f"{args.store_root}/{args.store_name}/{prefix}ckpt.pth.tar"
    torch.save(state, filename)
    if is_best:
        logging.info("===> Saving current best checkpoint...")
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.1, clip_max=10):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 == 0.).any():
        valid = (v1 != 0.)
        factor = torch.clamp(v2[valid] / v1[valid], clip_min, clip_max)
        matrix[:, valid] = (matrix[:, valid] - m1[valid]) * torch.sqrt(factor) + m2[valid]
        return matrix

    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


def create_dataloader_with_sampler(dataset, args, split='train', drop_last=False, use_grouped_override=None):
    """
    Create a DataLoader with optional grouped sampling support.
    
    Args:
        dataset: PyTorch Dataset
        args: Arguments object with sampler configuration
        split: 'train', 'val', or 'test'
        drop_last: Whether to drop the last incomplete batch
        use_grouped_override: If provided, overrides the grouped sampler setting
                             (useful for test set which may have different behavior)
        
    Returns:
        DataLoader with appropriate sampler
    """
    from torch.utils.data import DataLoader
    from samplers import GroupedBatchSampler, GroupedRandomSampler
    
    # Determine if we should use grouped sampler
    if use_grouped_override is not None:
        use_grouped = use_grouped_override
    else:
        use_grouped = args.use_grouped_sampler if split == 'train' else getattr(args, 'use_grouped_sampler_val', False)
    
    if use_grouped:
        if split == 'train':
            logging.info(f"\nUsing Grouped Sampler for {split} (type: {args.sampler_type})")
        else:
            logging.info(f"\nUsing Grouped Sampler for {split} (deterministic, no shuffling)")
        
        if args.sampler_type == 'batch':
            sampler = GroupedBatchSampler(
                dataset=dataset,
                batch_size=args.batch_size,
                drop_last=drop_last,
                shuffle=(split == 'train')
            )
            loader = DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=args.workers,
                pin_memory=True
            )
        else:  # 'random'
            sampler = GroupedRandomSampler(
                dataset=dataset,
                shuffle=(split == 'train')
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=drop_last
            )
        
        # Print sampler statistics
        if hasattr(sampler, 'name_to_indices'):
            num_groups = len(sampler.name_to_indices)
            group_sizes = [len(indices) for indices in sampler.name_to_indices.values()]
            logging.info(f"{split.capitalize()} - Number of name groups: {num_groups}")
            logging.info(f"{split.capitalize()} - Average samples per name: {sum(group_sizes) / num_groups:.2f}")
            logging.info(f"{split.capitalize()} - Min/Max samples per name: {min(group_sizes)}/{max(group_sizes)}")
    else:
        if split == 'train':
            logging.info(f"\nUsing Standard Random Sampler for {split}")
        else:
            logging.info(f"\nUsing Standard Sampler for {split}")
        
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == 'train'),
            num_workers=args.workers,
            pin_memory=True,
            drop_last=drop_last
        )
    
    return loader
