"""
Evaluates CIFAR-100 models on a given directory of images.
"""

from __future__ import print_function

import argparse
import os
import pickle
from pprint import pprint
import time
import random

import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import models.cifar as models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

import matplotlib.pyplot as plt

import sys

model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

# Parse arguments
parser = argparse.ArgumentParser(description='Evaluation of CIFAR-100 models')

parser.add_argument('-o', '--output_dir', type=str, required=True, help='path to output passive attention heatmaps')
parser.add_argument('-m', '--method', type=str, required=True, help='passive attention method (guidedbp, guidedbpximage, smoothgradguidedbp, gradcam, scorecam)')

# Datasets
parser.add_argument('-d', '--data', type=str, required=True, help='path to dataset',)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet)')
parser.add_argument('--depth', type=int, default=110, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
        help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# CUDA
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
#use_cuda = torch.cuda.is_available()
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)


def main():
    # Data loading code
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))  # CIFAR-100 normalization
    transform = transforms.Compose([
        transforms.Resize(32),  # For input into CIFAR-100 models
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.ImageFolder(root=args.data, transform=transform)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)

    # Model
    print("==> creating model '{}'".format(args.arch))
    num_classes = 100
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                cardinality=args.cardinality,
                num_classes=num_classes,
                depth=args.depth,
                widen_factor=args.widen_factor,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                growthRate=args.growthRate,
                compressionRate=args.compressionRate,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                widen_factor=args.widen_factor,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                block_name=args.block_name,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model = torch.nn.DataParallel(model)
    if use_cuda:
        model = model.cuda()
    else:
        model = model.cpu()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Resume
    title = args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        if args.arch.startswith("vgg"):
            checkpoint["state_dict"] = {f"module.{key[:9]}{key[16:]}": val for key, val in checkpoint["state_dict"].items()}
            checkpoint["state_dict"]["module.classifier.weight"] = checkpoint["state_dict"]["module.classifiet"]
            checkpoint["state_dict"]["module.classifier.bias"] = checkpoint["state_dict"]["module.classifie"]
            del checkpoint["state_dict"]["module.classifiet"]
            del checkpoint["state_dict"]["module.classifie"]
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise Exception("Must provide --resume")

    if args.arch.startswith("vgg") or args.arch.startswith("alexnet"):
        sys.path.insert(0, './pytorch-cnn-visualizations/src/')
    else:
        sys.path.insert(0, './pytorch-cnn-visualizations-modified/src/')

    model = model.module

    print(model)

    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(loader))
    for i, (inputs, _) in enumerate(loader):
        if use_cuda:
            inputs = inputs.cuda()
        inputs = torch.autograd.Variable(inputs, requires_grad=True)

        targets = model(inputs).argmax()
        last_conv_layer = None

        if args.method == "guidedbp":
            from guided_backprop import GuidedBackprop
            gb = GuidedBackprop(model)
            the_map = gb.generate_gradients(inputs, targets)
            the_map = np.moveaxis(the_map, 0, -1)
        elif args.method == "guidedbpximage":
            from guided_backprop import GuidedBackprop
            gb = GuidedBackprop(model)
            the_map = gb.generate_gradients(inputs, targets)
            the_map *= inputs.detach().numpy()[0]
            the_map = np.moveaxis(the_map, 0, -1)
        elif args.method == "smoothgradguidedbp":
            from guided_backprop import GuidedBackprop
            from smooth_grad import generate_smooth_grad
            gb = GuidedBackprop(model)
            the_map = generate_smooth_grad(gb, inputs, targets, 30, 0.1)
            the_map = np.moveaxis(the_map, 0, -1)
        elif args.method == "gradcam":
            from gradcam import GradCam
            if args.arch.startswith("vgg") or args.arch.startswith("alexnet"):
                last_conv_layer = len(model.features._modules.items()) - 2
            else:
                last_conv_layer = list(model._modules.keys())[-3]
            gc = GradCam(model, last_conv_layer)
            the_map = gc.generate_cam(inputs, targets)
        elif args.method == "scorecam":
            from scorecam import ScoreCam
            if args.arch.startswith("vgg") or args.arch.startswith("alexnet"):
                last_conv_layer = len(model.features._modules.items()) - 2
            else:
                last_conv_layer = list(model._modules.keys())[-3]
            sc = ScoreCam(model, last_conv_layer)
            the_map = sc.generate_cam(inputs, targets)
        elif args.method == "cameras":
            # Note, this only works if model can take in images of arbitrary size
            sys.path.insert(0, './CAMERAS/')
            from CAMERAS import CAMERAS
            if args.arch.startswith("vgg") or args.arch.startswith("alexnet"):
                last_conv_layer = "features." + list(model.features._modules.keys())[-2]
            else:
                last_conv_layer = list(model._modules.keys())[-3]
            cameras = CAMERAS(model, last_conv_layer, inputResolutions=[32, 100, 10])
            the_map = cameras.run(inputs, targets).cpu().numpy()
        else:
            raise Exception("Invalid passive attention method")

        if the_map.ndim == 3:
            # Convert to 2D
            the_map = np.sum(np.abs(the_map), axis=2)

        name = os.path.basename(dataset.imgs[i][0])
        num = int(name[name.index("r_")+2:-4])
        subdir = os.path.join(args.output_dir, f"img{num}/")
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        arch = args.arch.replace("_", "")
        if last_conv_layer is None:
            fn = f"img{num}_network_{arch}_method_{args.method}.mat"
        else:
            fn = f"img{num}_network_{arch}_method_{args.method}_selector_{last_conv_layer}.mat"
        outpath = os.path.join(subdir, fn)
        data = {args.method: the_map}
        sio.savemat(outpath, data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                    batch=i + 1,
                    size=len(loader),
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    )
        bar.next()
    bar.finish()


if __name__ == '__main__':
    main()
