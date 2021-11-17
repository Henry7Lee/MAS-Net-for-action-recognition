import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of action recognition models")
parser.add_argument('--dataset', type=str, choices=['somethingv1','somethingv2','kinetics','diving48','somethingv1_mini'],
                   default = 'somethingv1')
parser.add_argument('--dataset_path', type = str, default = '../',
                    help = 'root path to video dataset folders')
parser.add_argument('--store_name', type=str, default="MASNet[default]",
                    help = 'this experiment"s name')

# ========================= Model Configs ==========================
parser.add_argument('--type', type=str, default="MASNet", help = 'type of temporal models')
parser.add_argument('--arch', type=str, default="resnet50",choices=['resnet50','resnet101'],
                    help = 'backbone networks, currently only support resnet')
parser.add_argument('--num_segments', type=int, default=8)
parser.add_argument('--tune_from', type=str, default=None, help='fine-tune from checkpoint')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=70, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_type', default='MAS_lr', type=str,choices=['step_lr','cos_lr','MAS_lr'],
                    metavar='LRtype', help='learning rate type')

parser.add_argument('--lr_steps', default=[30, 40, 50, 60], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--dropout', '--dp', default=0.5, type=float,
                    metavar='dp', help='dropout ratio')
parser.add_argument('--warmup', type=int, default=0,
                    help='number of warmup epochs (default: 0)')

#========================= Optimizer Configs ==========================
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--clip-gradient', '--gd', default=20, type=float,
                    metavar='W', help='gradient norm clipping (default: 20)')

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 1)')
parser.add_argument('--debug',default=False,action='store_true',
                    help='decide if debug')

# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--gpus', nargs='+', type=int, default=[0,1,2,3])

parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_checkpoint', type=str, default='checkpoint')

parser.add_argument('--dense_sample', default=False, action="store_true",
                    help='use dense sample for video dataset')
parser.add_argument('--twice_sample', default=False, action="store_true",
                    help='use dense sample for video dataset')