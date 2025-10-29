import argparse
import os
import sys
import logging
import torch
import time
from model import Encoder, model_dict
from dataset import *
from utils import *
from training_utils import train_epoch, validate_epoch, create_dataloader_with_sampler

print = logging.info


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    #parser.add_argument('--save_freq', type=int, default=50, help='save frequency')

    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=6, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')

    # parser.add_argument('--base_data_dir', type=str, default='/home/cwatzenboeck/data/mlflow_cirpc_tmp/age_db/basic/', help='base directory for saving models and logs')
    parser.add_argument('--data_folder', type=str, default='/home/cwatzenboeck/data/public/agedb/', help='path to custom dataset')
    parser.add_argument('--dataset', type=str, default='AgeDB', choices=['AgeDB'], help='dataset')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50'])
    parser.add_argument('--resume', type=str, default='', help='resume ckpt path')
    parser.add_argument('--aug', type=str, default='crop,flip,color,grayscale', help='augmentations')
    parser.add_argument('--path_to_data_table', type=str, default='/home/cwatzenboeck/data/public/agedb/tabular/03_agedb_splits_stratified_new.csv', help='path to data table')

    parser.add_argument('--ckpt', type=str, default='', help='path to the trained encoder')

    # Grouped sampler options
    parser.add_argument('--use_grouped_sampler', action='store_true', 
                        help='Use GroupedBatchSampler to keep samples with same name in same batch')
    parser.add_argument('--sampler_type', type=str, default='batch', choices=['batch', 'random'],
                        help='Type of grouped sampler: "batch" uses GroupedBatchSampler, "random" uses GroupedRandomSampler')
    parser.add_argument('--use_grouped_sampler_val', action='store_true',
                        help='Use GroupedBatchSampler for validation/test (deterministic, no shuffling)')

    opt = parser.parse_args()

    sampler_suffix = '_grouped' if opt.use_grouped_sampler else ''
    val_sampler_suffix = '_val_grouped' if opt.use_grouped_sampler_val else ''
    opt.model_name = f"Regressor_{opt.dataset}_ep_{opt.epochs}_lr_{opt.learning_rate}_d_{opt.lr_decay_rate}_wd_{opt.weight_decay}_mmt_{opt.momentum}_bsz_{opt.batch_size}_trial_{opt.trial}{sampler_suffix}{val_sampler_suffix}"
    if len(opt.resume):
        opt.model_name = opt.resume.split('/')[-1][:-len('_last.pth')]
    opt.save_folder = '/'.join(opt.ckpt.split('/')[:-1])

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(opt.save_folder, f'{opt.model_name}.log')),
            logging.StreamHandler()
        ]
    )

    print(f"Model name: {opt.model_name}")
    print(f"Options: {opt}")

    return opt


def set_loader(opt):
    """
    Create data loaders with optional grouped sampling support.
    Uses create_dataloader_with_sampler from training_utils.
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

    # Create loaders using the helper function
    train_loader = create_dataloader_with_sampler(train_dataset, opt, split='train', drop_last=False)
    val_loader = create_dataloader_with_sampler(val_dataset, opt, split='val', drop_last=False)
    test_loader = create_dataloader_with_sampler(test_dataset, opt, split='test', drop_last=False)

    return train_loader, val_loader, test_loader


def set_model(opt):
    model = Encoder(name=opt.model)
    criterion = torch.nn.L1Loss()

    dim_in = model_dict[opt.model][1]
    dim_out = get_label_dim(opt.dataset)
    regressor = torch.nn.Linear(dim_in, dim_out)
    ckpt = torch.load(opt.ckpt, map_location='cpu', weights_only=False)
    state_dict = ckpt['model']

    if torch.cuda.device_count() > 1:
        model.encoder = torch.nn.DataParallel(model.encoder)
    else:
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
    model = model.cuda()
    regressor = regressor.cuda()
    criterion = criterion.cuda()
    torch.backends.cudnn.benchmark = True

    model.load_state_dict(state_dict)
    print(f"<=== Epoch [{ckpt['epoch']}] checkpoint Loaded from {opt.ckpt}!")

    return model, regressor, criterion


def train(train_loader, model, regressor, criterion, optimizer, epoch, opt):
    """
    Training loop for linear probe.
    Uses generic train_epoch from training_utils.
    """
    model.eval()
    regressor.train()
    
    def compute_loss_fn(batch):
        """Compute loss for linear probe."""
        images = batch['image']
        labels = batch['y_true']
        # names = batch['name']  # Available if loss needs sample IDs
        
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        
        with torch.no_grad():
            features = model(images)
        
        output = regressor(features.detach())
        loss = criterion(output, labels)
        
        return loss, bsz
    
    return train_epoch(train_loader, compute_loss_fn, optimizer, epoch, opt, print_fn=print)


def validate(val_loader, model, regressor):
    """
    Validation loop for linear probe.
    Uses generic validate_epoch from training_utils.
    """
    model.eval()
    regressor.eval()
    criterion_l1 = torch.nn.L1Loss()
    
    def compute_val_loss_fn(batch):
        """Compute validation loss for linear probe."""
        images = batch['image']
        labels = batch['y_true']
        images = images.cuda()
        labels = labels.cuda()
        bsz = labels.shape[0]
        
        features = model(images)
        output = regressor(features)
        loss = criterion_l1(output, labels)
        
        return loss, bsz
    
    return validate_epoch(val_loader, compute_val_loss_fn, print_fn=print)


def main():
    opt = parse_option()

    # build data loader
    train_loader, val_loader, test_loader = set_loader(opt)

    # build model and criterion
    model, regressor, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, regressor)

    save_file_best = os.path.join(opt.save_folder, f"{opt.model_name}_best.pth")
    save_file_last = os.path.join(opt.save_folder, f"{opt.model_name}_last.pth")
    best_error = 1e5

    start_epoch = 1
    if len(opt.resume):
        ckpt_state = torch.load(opt.resume, weights_only=False)
        regressor.load_state_dict(ckpt_state['state_dict'])
        start_epoch = ckpt_state['epoch'] + 1
        best_error = ckpt_state['best_error']
        print(f"<=== Epoch [{ckpt_state['epoch']}] Resumed from {opt.resume}!")


    # training routine
    for epoch in range(start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        train(train_loader, model, regressor, criterion, optimizer, epoch, opt)

        valid_error = validate(val_loader, model, regressor)
        print('Val L1 error: {:.3f}'.format(valid_error))

        is_best = valid_error < best_error
        best_error = min(valid_error, best_error)
        print(f"Best Error: {best_error:.3f}")

        if is_best:
            torch.save({
                'epoch': epoch,
                'state_dict': regressor.state_dict(),
                'best_error': best_error
            }, save_file_best)

        torch.save({
            'epoch': epoch,
            'state_dict': regressor.state_dict(),
            'last_error': valid_error
        }, save_file_last)

    print("=" * 120)
    print("Test best model on test set...")
    checkpoint = torch.load(save_file_best, weights_only=False)
    regressor.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded best model, epoch {checkpoint['epoch']}, best val error {checkpoint['best_error']:.3f}")
    test_loss = validate(test_loader, model, regressor)
    to_print = 'Test L1 error: {:.3f}'.format(test_loss)
    print(to_print)


if __name__ == '__main__':
    main()
