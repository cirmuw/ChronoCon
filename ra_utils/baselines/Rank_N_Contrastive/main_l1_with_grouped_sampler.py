"""
main_l1_with_grouped_sampler.py


Modified version of main_l1.py that uses the GroupedBatchSampler.

"""

import argparse
import os
import sys
import logging
import torch
import time
from model import SupResNet
from dataset import *
from utils import *
from grouped_sampler import GroupedBatchSampler, GroupedRandomSampler

print = logging.info


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--save_curr_freq', type=int, default=1, help='save curr last frequency')

    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=6, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=2, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.2, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')

    parser.add_argument('--base_data_dir', type=str, default='/home/cwatzenboeck/data/mlflow_cirpc_tmp/age_db/basic/', help='base directory for saving models and logs')
    parser.add_argument('--data_folder', type=str, default='/home/cwatzenboeck/data/public/agedb/', help='path to custom dataset')
    
    parser.add_argument('--dataset', type=str, default='AgeDB', choices=['AgeDB'], help='dataset')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50'])
    parser.add_argument('--resume', type=str, default='', help='resume ckpt path')
    parser.add_argument('--aug', type=str, default='crop,flip,color,grayscale', help='augmentations')
    parser.add_argument('--path_to_data_table', type=str, default='/home/cwatzenboeck/data/public/agedb/tabular/03_agedb_splits_stratified_new.csv', help='path to data table')
    
    # NEW: Add option to use grouped sampler
    parser.add_argument('--use_grouped_sampler', action='store_true', 
                        help='Use GroupedBatchSampler to keep samples with same name in same batch')
    parser.add_argument('--sampler_type', type=str, default='batch', choices=['batch', 'random'],
                        help='Type of grouped sampler: "batch" uses GroupedBatchSampler, "random" uses GroupedRandomSampler')
    parser.add_argument('--use_grouped_sampler_val', action='store_true',
                        help='Use GroupedBatchSampler for validation/test (deterministic, no shuffling)')

    opt = parser.parse_args()

    opt.model_path = f'{opt.base_data_dir}/save/{opt.dataset}_models'
    sampler_suffix = '_grouped' if opt.use_grouped_sampler else ''
    val_sampler_suffix = '_val_grouped' if opt.use_grouped_sampler_val else ''
    opt.model_name = f"L1_{opt.dataset}_{opt.model}_ep_{opt.epochs}_lr_{opt.learning_rate}_d_{opt.lr_decay_rate}_wd_{opt.weight_decay}_mmt_{opt.momentum}_bsz_{opt.batch_size}_aug_{opt.aug}_trial_{opt.trial}{sampler_suffix}{val_sampler_suffix}"
    
    if len(opt.resume):
        opt.model_name = opt.resume.split('/')[-2]

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.log_folder = os.path.join(f'{opt.base_data_dir}/save/{opt.dataset}_logs/', opt.model_name)
    if not os.path.isdir(opt.log_folder):
        os.makedirs(opt.log_folder)
    
    
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(opt.save_folder, 'training.log')),
            logging.StreamHandler()
        ]
    )
    print(f"Model name: {opt.model_name}")
    print(f"Options: {opt}")
    print(f"Logging to: {os.path.join(opt.save_folder, 'training.log')}")

    return opt


def set_loader(opt):
    """
    Modified set_loader function that supports grouped sampling.
    """
    train_transform = get_transforms(split='train', aug=opt.aug)
    val_transform = get_transforms(split='val', aug=opt.aug)

    print(f"Train Transforms: {train_transform}")
    print(f"Val Transforms: {val_transform}")

    train_dataset = globals()[opt.dataset](data_folder=opt.data_folder, transform=train_transform, split='train', path_to_data_table=opt.path_to_data_table)
    val_dataset = globals()[opt.dataset](data_folder=opt.data_folder, transform=val_transform, split='val', path_to_data_table=opt.path_to_data_table)
    test_dataset = globals()[opt.dataset](data_folder=opt.data_folder, transform=val_transform, split='test', path_to_data_table=opt.path_to_data_table)

    print(f'Train set size: {train_dataset.__len__()}\t'
          f'Val set size: {val_dataset.__len__()}\t'
          f'Test set size: {test_dataset.__len__()}')

    # Create train loader with optional grouped sampler
    if opt.use_grouped_sampler:
        print(f"\nUsing Grouped Sampler (type: {opt.sampler_type})")
        
        if opt.sampler_type == 'batch':
            # Use GroupedBatchSampler - handles batching internally
            train_sampler = GroupedBatchSampler(
                dataset=train_dataset,
                batch_size=opt.batch_size,
                drop_last=False,
                shuffle=True
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_sampler=train_sampler,  # Use batch_sampler instead of batch_size
                num_workers=opt.num_workers, 
                pin_memory=True
            )
        else:  # 'random'
            # Use GroupedRandomSampler - only controls ordering, not batching
            train_sampler = GroupedRandomSampler(
                dataset=train_dataset,
                shuffle=True
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=opt.batch_size,
                sampler=train_sampler,  # Use sampler
                num_workers=opt.num_workers,
                pin_memory=True
            )
        
        # Analyze the sampler to show grouping statistics
        if hasattr(train_sampler, 'name_to_indices'):
            num_groups = len(train_sampler.name_to_indices)
            group_sizes = [len(indices) for indices in train_sampler.name_to_indices.values()]
            print(f"Number of name groups: {num_groups}")
            print(f"Average samples per name: {sum(group_sizes) / num_groups:.2f}")
            print(f"Min/Max samples per name: {min(group_sizes)}/{max(group_sizes)}")
    else:
        print("\nUsing Standard Random Sampler")
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=opt.batch_size, 
            shuffle=True, 
            num_workers=opt.num_workers, 
            pin_memory=True
        )

    # Create val and test loaders with optional grouped sampler
    if opt.use_grouped_sampler_val:
        print("\nUsing Grouped Sampler for Validation/Test (deterministic, no shuffling)")
        
        # Validation loader with GroupedBatchSampler
        val_sampler = GroupedBatchSampler(
            dataset=val_dataset,
            batch_size=opt.batch_size,
            drop_last=False,
            shuffle=False  # Deterministic for validation
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=opt.num_workers,
            pin_memory=True
        )
        
        # # Test loader with GroupedBatchSampler
        # test_sampler = GroupedBatchSampler(
        #     dataset=test_dataset,
        #     batch_size=opt.batch_size,
        #     drop_last=False,
        #     shuffle=False  # Deterministic for test
        # )
        # test_loader = torch.utils.data.DataLoader(
        #     test_dataset,
        #     batch_sampler=test_sampler,
        #     num_workers=opt.num_workers,
        #     pin_memory=True
        # )
        
        # Analyze the samplers
        if hasattr(val_sampler, 'name_to_indices'):
            num_groups = len(val_sampler.name_to_indices)
            group_sizes = [len(indices) for indices in val_sampler.name_to_indices.values()]
            print(f"Val - Number of name groups: {num_groups}")
            print(f"Val - Average samples per name: {sum(group_sizes) / num_groups:.2f}")
            print(f"Val - Min/Max samples per name: {min(group_sizes)}/{max(group_sizes)}")
        
        # if hasattr(test_sampler, 'name_to_indices'):
        #     num_groups = len(test_sampler.name_to_indices)
        #     group_sizes = [len(indices) for indices in test_sampler.name_to_indices.values()]
        #     print(f"Test - Number of name groups: {num_groups}")
        #     print(f"Test - Average samples per name: {sum(group_sizes) / num_groups:.2f}")
        #     print(f"Test - Min/Max samples per name: {min(group_sizes)}/{max(group_sizes)}")
    else:
        print("\nUsing Standard Sampler for Validation/Test")
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True
        )
        
    # Test set is always using the standard sampler
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def set_model(opt):
    model = SupResNet(name=opt.model, num_classes=get_label_dim(opt.dataset))
    criterion = torch.nn.L1Loss()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        torch.backends.cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, batch in enumerate(train_loader):
        images = batch['image']
        labels = batch['y_true']
        data_time.update(time.time() - end)
        bsz = labels.shape[0]

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        output = model(images)

        loss = criterion(output, labels)
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0:
            to_print = 'Train: [{0}][{1}/{2}]\t'\
                       'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                       'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'\
                       'loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses
            )
            print(to_print)
            sys.stdout.flush()


def validate(val_loader, model):
    model.eval()

    losses = AverageMeter()
    criterion_l1 = torch.nn.L1Loss()

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            images = batch['image']
            labels = batch['y_true']
            images = images.cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            output = model(images)

            loss_l1 = criterion_l1(output, labels)
            losses.update(loss_l1.item(), bsz)

    return losses.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader, val_loader, test_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    start_epoch = 1
    if len(opt.resume):
        ckpt_state = torch.load(opt.resume)
        model.load_state_dict(ckpt_state['model'])
        optimizer.load_state_dict(ckpt_state['optimizer'])
        start_epoch = ckpt_state['epoch'] + 1
        print(f"<=== Epoch [{ckpt_state['epoch']}] Resumed from {opt.resume}!")

    best_error = 1e5
    save_file_best = os.path.join(opt.save_folder, 'best.pth')

    # training routine
    for epoch in range(start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, opt)

        valid_error = validate(val_loader, model)
        print('Val L1 error: {:.3f}'.format(valid_error))

        is_best = valid_error < best_error
        best_error = min(valid_error, best_error)
        print(f"Best Error: {best_error:.3f}")

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

        if epoch % opt.save_curr_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'curr_last.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

        if is_best:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'best_error': best_error
            }, save_file_best)

    print("=" * 120)
    print("Test best model on test set...")
    checkpoint = torch.load(save_file_best)
    model.load_state_dict(checkpoint['model'])
    print(f"Loaded best model, epoch {checkpoint['epoch']}, best val error {checkpoint['best_error']:.3f}")
    test_loss = validate(test_loader, model)
    to_print = 'Test L1 error: {:.3f}'.format(test_loss)
    print(to_print)


if __name__ == '__main__':
    main()
