import argparse
import os
import time
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import models
import pandas as pd
import numpy as np
from multiscaleloss import realEPE          # kept for validation monitoring
from unsupervised_loss import unsupervised_loss
import datetime
from tensorboardX import SummaryWriter
from util import AverageMeter, save_checkpoint

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(
    description='StrainNet – Unsupervised Training (Yu et al., ECCV 2016)',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument('--arch', default='StrainNet_f',
                    choices=['StrainNet_f', 'StrainNet_h'],
                    help='network f or h')
parser.add_argument('--solver', default='adam', choices=['adam', 'sgd'],
                    help='solver algorithm')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (0 = full dataset)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float,
                    metavar='B', help='bias decay')
parser.add_argument('--multiscale-weights', '-w',
                    default=[0.005, 0.01, 0.02, 0.08, 0.32],
                    type=float, nargs=5,
                    metavar=('W2', 'W3', 'W4', 'W5', 'W6'),
                    help='per-scale loss weights, finest to coarsest')
parser.add_argument('--sparse', action='store_true',
                    help='handle sparse ground-truth flow during validation')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')
parser.add_argument('--div-flow', default=2, type=float,
                    help='divisor applied to the flow output by the network')
parser.add_argument('--milestones', default=[40, 80, 120, 160, 200, 240],
                    metavar='N', nargs='*',
                    help='epochs at which learning rate is halved')
# Unsupervised loss hyper-parameters
parser.add_argument('--lambda-smooth', default=0.5, type=float,
                    help='weight of the smoothness term (lambda in Yu et al.)')
parser.add_argument('--charbonnier-alpha', default=0.45, type=float,
                    help='exponent of the Charbonnier penalty function')
parser.add_argument('--charbonnier-eps', default=1e-3, type=float,
                    help='epsilon of the Charbonnier penalty function')

best_EPE = -1
n_iter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpecklesDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.Speckles_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.Speckles_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        Ref_name   = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 0])
        Def_name   = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 1])
        Dispx_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 2])
        Dispy_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 3])

        Ref   = np.genfromtxt(Ref_name,   delimiter=',')
        Def   = np.genfromtxt(Def_name,   delimiter=',')
        Dispx = np.genfromtxt(Dispx_name, delimiter=',')
        Dispy = np.genfromtxt(Dispy_name, delimiter=',')

        Ref   = Ref[np.newaxis, ...]
        Def   = Def[np.newaxis, ...]
        Dispx = Dispx[np.newaxis, ...]
        Dispy = Dispy[np.newaxis, ...]

        sample = {'Ref': Ref, 'Def': Def, 'Dispx': Dispx, 'Dispy': Dispy}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Normalization(object):
    """Normalise images to [0, 1] and displacements to [-0.5, 0.5]."""

    def __call__(self, sample):
        Ref, Def, Dispx, Dispy = (
            sample['Ref'], sample['Def'], sample['Dispx'], sample['Dispy']
        )
        img_mean, img_std   = 0.0, 255.0
        disp_mean, disp_std = -1.0, 2.0

        return {
            'Ref':   torch.from_numpy((Ref   - img_mean)  / img_std ).float(),
            'Def':   torch.from_numpy((Def   - img_mean)  / img_std ).float(),
            'Dispx': torch.from_numpy((Dispx - disp_mean) / disp_std).float(),
            'Dispy': torch.from_numpy((Dispy - disp_mean) / disp_std).float(),
        }


def main():
    global args, best_EPE
    args = parser.parse_args()

    save_path = '{},{},{}epochs{},b{},lr{},unsupervised'.format(
        args.arch,
        args.solver,
        args.epochs,
        ',epochSize' + str(args.epoch_size) if args.epoch_size > 0 else '',
        args.batch_size,
        args.lr,
    )

    print('=> saving to {}'.format(save_path))
    os.makedirs(save_path, exist_ok=True)

    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    test_writer  = SummaryWriter(os.path.join(save_path, 'test'))

    transform = transforms.Compose([Normalization()])

    train_set = SpecklesDataset(
        csv_file='~/Train_annotations.csv',
        root_dir='~/Train_Data/',
        transform=transform,
    )
    test_set = SpecklesDataset(
        csv_file='~/Test_annotations.csv',
        root_dir='~/Test_Data/',
        transform=transform,
    )

    print('{} samples found: {} train / {} test'.format(
        len(train_set) + len(test_set), len(train_set), len(test_set),
    ))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=False,
    )

    if args.pretrained:
        network_data = torch.load(args.pretrained)
        print('=> using pre-trained model')
    else:
        network_data = None
        print('=> creating model from scratch')

    model = models.__dict__[args.arch](network_data).cuda()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    assert args.solver in ['adam', 'sgd']
    print('=> {} solver'.format(args.solver))
    param_groups = [
        {'params': model.module.bias_parameters(),   'weight_decay': args.bias_decay},
        {'params': model.module.weight_parameters(), 'weight_decay': args.weight_decay},
    ]
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(
            param_groups, args.lr, betas=(args.momentum, args.beta),
        )
    else:
        optimizer = torch.optim.SGD(
            param_groups, args.lr, momentum=args.momentum,
        )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=0.5,
    )

    for epoch in range(args.start_epoch, args.epochs):

        train_loss, _ = train(
            train_loader, model, optimizer, epoch, train_writer, scheduler,
        )
        train_writer.add_scalar('train_loss', train_loss, epoch)

        with torch.no_grad():
            EPE = validate(val_loader, model, epoch)
        test_writer.add_scalar('mean EPE', EPE, epoch)

        if best_EPE < 0:
            best_EPE = EPE

        is_best = EPE < best_EPE
        best_EPE = min(EPE, best_EPE)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'best_EPE': best_EPE,
                'div_flow': args.div_flow,
            },
            is_best,
            save_path,
        )


def train(train_loader, model, optimizer, epoch, train_writer, scheduler):
    global n_iter, args
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()

    epoch_size = (
        len(train_loader)
        if args.epoch_size == 0
        else min(len(train_loader), args.epoch_size)
    )

    model.train()
    end = time.time()

    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)

        # Raw single-channel images – used by the photometric loss
        ref_img = batch['Ref'].float().to(device)   # (B, 1, H, W)
        def_img = batch['Def'].float().to(device)   # (B, 1, H, W)

        # 3-channel input for the network (grayscale replicated to RGB)
        in_ref = torch.cat([ref_img, ref_img, ref_img], 1)
        in_def = torch.cat([def_img, def_img, def_img], 1)
        net_input = torch.cat([in_ref, in_def], 1)   # (B, 6, H, W)

        # Forward pass  →  [flow2, flow3, flow4, flow5, flow6]
        output = model(net_input)

        # Convert normalised predictions to pixel displacements for the warp
        output_px = [o * args.div_flow for o in output]

        # Unsupervised loss: photometric + smoothness (Yu et al., ECCV 2016)
        loss = unsupervised_loss(
            network_output=output_px,
            ref=ref_img,
            deformed=def_img,
            weights=args.multiscale_weights,
            lambda_smooth=args.lambda_smooth,
            alpha=args.charbonnier_alpha,
            eps=args.charbonnier_eps,
        )

        losses.update(loss.item(), ref_img.size(0))
        train_writer.add_scalar('train_loss', loss.item(), n_iter)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]  Time {3}  Data {4}  Loss {5}'.format(
                    epoch, i, epoch_size, batch_time, data_time, losses,
                )
            )

        n_iter += 1

    return losses.avg, 0.0


def validate(val_loader, model, epoch):
    global args
    batch_time = AverageMeter()
    flow2_EPEs = AverageMeter()

    model.eval()
    end = time.time()

    for i, batch in enumerate(val_loader):
        target_x = batch['Dispx'].to(device)
        target_y = batch['Dispy'].to(device)
        target   = torch.cat([target_x, target_y], 1)

        in_ref = batch['Ref'].float().to(device)
        in_ref = torch.cat([in_ref, in_ref, in_ref], 1)

        in_def = batch['Def'].float().to(device)
        in_def = torch.cat([in_def, in_def, in_def], 1)

        net_input = torch.cat([in_ref, in_def], 1)

        output = model(net_input)
        # eval mode returns only flow2
        flow2_EPE = args.div_flow * realEPE(output, target, sparse=args.sparse)

        flow2_EPEs.update(flow2_EPE.item(), target.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]  Time {2}  EPE {3}'.format(
                i, len(val_loader), batch_time, flow2_EPEs,
            ))

    print(' * EPE {:.3f}'.format(flow2_EPEs.avg))
    return flow2_EPEs.avg


if __name__ == '__main__':
    main()
