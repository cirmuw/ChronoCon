import argparse
import os
import sys
import logging
import torch
import time
from dataset import *
from utils import *
from model import Encoder
from loss import RnCLoss
from training_utils import train_epoch, create_dataloader_with_sampler

print = logging.info


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--save_curr_freq', type=int, default=1, help='save curr last frequency')

    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=6, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=400, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='learning rate')
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

    # RnCLoss Parameters
    parser.add_argument('--temp', type=float, default=2, help='temperature')
    parser.add_argument('--label_diff', type=str, default='l1', choices=['l1'], help='label distance function')
    parser.add_argument('--feature_sim', type=str, default='l2', choices=['l2'], help='feature similarity function')
    parser.add_argument('--path_to_data_table', type=str, default='/home/cwatzenboeck/data/public/agedb/tabular/03_agedb_splits_stratified_new.csv', help='path to data table')

    # Grouped sampler options
    parser.add_argument('--use_grouped_sampler', action='store_true', 
                        help='Use GroupedBatchSampler to keep samples with same name in same batch')
    parser.add_argument('--sampler_type', type=str, default='batch', choices=['batch', 'random'],
                        help='Type of grouped sampler: "batch" uses GroupedBatchSampler, "random" uses GroupedRandomSampler')

    opt = parser.parse_args()

    # opt.model_path = './save/{}_models'.format(opt.dataset)
    opt.model_path = f'{opt.base_data_dir}/save/{opt.dataset}_models'
    sampler_suffix = '_grouped' if opt.use_grouped_sampler else ''
    opt.model_name = (
        f"RnC_{opt.dataset}_{opt.model}_ep_{opt.epochs}_lr_{opt.learning_rate}_d_{opt.lr_decay_rate}"
        f"_wd_{opt.weight_decay}_mmt_{opt.momentum}_bsz_{opt.batch_size}_aug_{opt.aug}"
        f"_temp_{opt.temp}_label_{opt.label_diff}_feature_{opt.feature_sim}_trial_{opt.trial}{sampler_suffix}"
    )
    if len(opt.resume):
        opt.model_name = opt.resume.split('/')[-2]

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    else:
        print('WARNING: folder exist.')

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(opt.save_folder, 'training.log')),
            logging.StreamHandler()
        ])

    print(f"Model name: {opt.model_name}")
    print(f"Options: {opt}")

    return opt


def set_loader(opt):
    """
    Create data loader with optional grouped sampling support.
    Uses create_dataloader_with_sampler from training_utils.
    Note: RnC uses TwoCropTransform for contrastive learning.
    """
    train_transform = get_transforms(split='train', aug=opt.aug)
    print(f"Train Transforms: {train_transform}")

    train_dataset = globals()[opt.dataset](
        data_folder=opt.data_folder,
        transform=TwoCropTransform(train_transform),
        split='train',
        path_to_data_table=opt.path_to_data_table
    )
    print(f'Train set size: {train_dataset.__len__()}')

    # Create train loader using the helper function
    # RnC requires drop_last=True for contrastive learning
    train_loader = create_dataloader_with_sampler(train_dataset, opt, split='train', drop_last=True)

    return train_loader


def set_model(opt):
    model = Encoder(name=opt.model)
    criterion = RnCLoss(temperature=opt.temp, label_diff=opt.label_diff, feature_sim=opt.feature_sim)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        torch.backends.cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """
    Training loop for Rank-N-Contrast.
    Uses generic train_epoch from training_utils.
    """
    model.train()
    
    def compute_loss_fn(batch):
        """Compute loss for RnC contrastive learning."""
        images = batch['image']
        labels = batch['y_true']
        # names = batch['name']  # Available if loss needs sample IDs
        
        bsz = labels.shape[0]
        # Concatenate two crops for contrastive learning
        images = torch.cat([images[0], images[1]], dim=0)
        
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        
        # Get features for both crops
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        
        loss = criterion(features, labels)
        
        return loss, bsz
    
    return train_epoch(train_loader, compute_loss_fn, optimizer, epoch, opt, print_fn=print)


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

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

    # training routine
    for epoch in range(start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        train(train_loader, model, criterion, optimizer, epoch, opt)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

        if epoch % opt.save_curr_freq == 0:
            save_file = os.path.join(opt.save_folder, 'curr_last.pth')
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
