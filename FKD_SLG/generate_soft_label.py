import argparse
import os
import random
import shutil
import time
import warnings
import timm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from utils import RandomResizedCropWithCoords
from utils import RandomHorizontalFlipWithRes
from utils import ImageFolder_FKD_GSL
from utils import ComposeWithCoords

from torchvision.transforms import InterpolationMode

import torch.multiprocessing


parser = argparse.ArgumentParser(description='FKD Soft Label Generation on ImageNet-1K')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
# FKD soft label generation args
parser.add_argument('--num_crops', default=500, type=int,
                    help='total number of crops in each image')
parser.add_argument('--num_seg', default=50, type=int,
                    help='number of crops on each GPU during generating')
parser.add_argument("--min_scale_crops", type=float, default=0.08,
                    help="argument in RandomResizedCrop")
parser.add_argument("--max_scale_crops", type=float, default=0.936,
                    help="argument in RandomResizedCrop")
parser.add_argument("--temp", type=float, default=1.0,
                    help="temperature on soft label")
parser.add_argument('--save_path', default='./FKD_effL2_475_soft_label', type=str, metavar='PATH',
                    help='path to save soft labels')
parser.add_argument('--reference_path', default=None, type=str, metavar='PATH',
                    help='path to existing soft labels files, we can use existing crop locations to generate new soft labels')
parser.add_argument('--input_size', default=475, type=int, metavar='S',
                    help='input size of teacher model')
parser.add_argument('--teacher_path', default='', type=str, metavar='TEACHER',
                    help='path of pre-trained teacher')
parser.add_argument('--teacher_source', default='timm', type=str, metavar='SOURCE',
                    help='source of pre-trained teacher models: pytorch or timm')
parser.add_argument('--label_type', default='marginal_smoothing_k5', type=str, metavar='TYPE',
                    help='type of generated soft labels')
parser.add_argument('--use_fp16', dest='use_fp16', action='store_true',
                    help='save soft labels as `fp16`')


sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)

def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def main():
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if os.path.isfile(args.teacher_path): 
        print("=> using pre-trained model from '{}'".format(args.teacher_path))
        model = models.__dict__[args.arch](pretrained=False)
        checkpoint = torch.load(args.teacher_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
    elif args.teacher_source == 'timm':
        # Timm pre-trained models
        print("=> using pre-trained model '{}'".format(args.arch))
        model = timm.create_model(args.arch, pretrained=True, num_classes=1000)
    elif args.teacher_source == 'pytorch':
        # PyTorch pre-trained models
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("'{}' currently is not supported. Please use pytorch, timm or your own pre-trained models as teachers.".format(args.teacher_source))
        # add your code of loading teacher here.
        return

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # freeze all layers
    for name, param in model.named_parameters():
            param.requires_grad = False

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    # BEIT, etc.
    # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                                  std=[0.5, 0.5, 0.5])
    # ResNet, efficientnet_l2_ns_475, etc.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = ImageFolder_FKD_GSL(
        num_crops=args.num_crops,
        save_path=args.save_path,
        reference_path=args.reference_path,
        root=traindir,
        transform=ComposeWithCoords(transforms=[
            RandomResizedCropWithCoords(size=args.input_size,
                                        scale=(args.min_scale_crops, args.max_scale_crops),
                                        interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlipWithRes(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    #(train_sampler is None) 
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, worker_init_fn=set_worker_sharing_strategy)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(int(256/224*args.input_size)),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # test the accuracy of teacher model
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    # generate soft labels
    generate_soft_labels(train_loader, model, args)


def generate_soft_labels(train_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    # switch to eval mode
    model.eval()

    end = time.time()
    for i, (images, target, flip_status, coords, path) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = torch.stack(images, dim=0).permute(1,0,2,3,4)
        flip_status = torch.stack(flip_status, dim=0).permute(1,0)
        coords = torch.stack(coords, dim=0).permute(1,0,2)

        for k in range(images.size()[0]):
            save_path = os.path.join(args.save_path,'/'.join(path[k].split('/')[-4:-1]))
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            new_filename = os.path.join(save_path,'/'.join(path[k].split('/')[-1:]).split('.')[0] + '.tar')
            if not os.path.exists(new_filename):
                if args.num_crops <= args.num_seg:
                    if args.gpu is not None:
                        images[k] = images[k].cuda(args.gpu, non_blocking=True)
                    if torch.cuda.is_available():
                        target = target.cuda(args.gpu, non_blocking=True)

                    # compute output
                    output = model(images[k])

                    output = nn.functional.softmax(output / args.temp, dim=1)
                    images[k] = images[k].detach()#.cpu()
                    output = output.detach().cpu()

                    output = label_quantization(output, args.label_type)

                    state = [coords[k].detach().numpy(), flip_status[k].detach().numpy(), output]
                    torch.save(state, new_filename)
                else:
                    output_all = []
                    for split in range(int(args.num_crops / args.num_seg)):
                        if args.gpu is not None:
                            images[k][split*args.num_seg:(split+1)*args.num_seg] = images[k][split*args.num_seg:(split+1)*args.num_seg].cuda(args.gpu, non_blocking=True)
                        if torch.cuda.is_available():
                            target = target.cuda(args.gpu, non_blocking=True)

                        # compute output
                        output = model(images[k][split*args.num_seg:(split+1)*args.num_seg])

                        output = nn.functional.softmax(output / args.temp, dim=1)
                        images[k][split*args.num_seg:(split+1)*args.num_seg] = images[k][split*args.num_seg:(split+1)*args.num_seg].detach()#.cpu()
                        output = output.detach().cpu()
                        output_all.append(output)

                    output_all = torch.cat(output_all, dim=0)
                    output_quan = label_quantization(output_all, args.label_type)

                    if args.use_fp16:
                        state = [np.float16(coords[k].detach().numpy()), flip_status[k].detach().numpy(), np.float16(output_quan)]
                    else:
                        state = [coords[k].detach().numpy(), flip_status[k].detach().numpy(), output_quan]

                    torch.save(state, new_filename)

        print(i,'/', len(train_loader), i/len(train_loader)*100, "%")


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

def label_quantization(full_label, label_type):
    # '(1) hard; (2) smoothing; (3) marginal_smoothing_k5 and marginal_renorm_k5; (4) marginal_smoothing_k10'
    output_quantized = []
    for kk,p in enumerate(full_label):
        # 1 hard
        if label_type == 'hard':
            output = torch.argmax(p)
            output_quantized.append(output)
        # 2 smoothing
        elif label_type == 'smoothing':
            output = torch.argmax(p)
            value = p[output]#.item()
            output_quantized.append(torch.stack([output, value], dim=0))
        # 3 marginal_smoothing_k5 and marginal_renorm_k5
        elif label_type == 'marginal_smoothing_k5' or label_type == 'marginal_renorm_k5':
            output = torch.argsort(p, descending=True)
            value = p[output[:5]]
            output_quantized.append(torch.stack([output[:5], value], dim=0))
        # 4 marginal_smoothing_k10
        elif label_type == 'marginal_smoothing_k10':
            output = torch.argsort(p, descending=True)
            value = p[output[:10]]
            output_quantized.append(torch.stack([output[:10], value], dim=0))

    output_quantized = torch.stack(output_quantized, dim=0)

    return output_quantized.detach().numpy()


class AverageMeter(object):
    """Computes and stores the average and current value"""
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
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
