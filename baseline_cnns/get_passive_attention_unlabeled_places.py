"""
Evaluates Places365 models on a given directory of images.
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
import torchvision.models as models

import torch.nn.functional as F

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

import sys


# Parse arguments
parser = argparse.ArgumentParser(description='Evaluation of Places365 models')

parser.add_argument('-o', '--output_dir', type=str, required=True, help='path to output passive attention heatmaps')
parser.add_argument('-m', '--method', type=str, required=True, help='passive attention method (guidedbp, guidedbpximage, smoothgradguidedbp, gradcam, scorecam, cameras)')

# Datasets
parser.add_argument('-d', '--data', type=str, required=True, help='path to dataset',)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
        help='path to latest checkpoint (default: none)')
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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # places normalization
    transform = transforms.Compose([
        transforms.Resize(224),  # For input into places models
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.ImageFolder(root=args.data, transform=transform)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)

    # Model
    print("==> creating model '{}'".format(args.arch))

    if args.arch == "resnet50":
        sys.path.insert(0, './pytorch-cnn-visualizations-modified/src/')
        model = models.resnet50(num_classes=365)
    elif args.arch == "alexnet":
        sys.path.insert(0, './pytorch-cnn-visualizations/src/')
        model = models.alexnet(num_classes=365)
    else:
        raise Exception("Invalid architecture chosen")

    if use_cuda:
        model = model.cuda()
    else:
        model = model.cpu()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
    else:
        raise Exception("Must provide --resume")

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
            if args.arch == "alexnet":
                last_conv_layer = len(model.features._modules.items()) - 2
            else:
                last_conv_layer = list(model._modules.keys())[-3]
            gc = GradCam(model, last_conv_layer)
            the_map = gc.generate_cam(inputs, targets)
        elif args.method == "scorecam":
            from scorecam import ScoreCam
            if args.arch == "alexnet":
                last_conv_layer = len(model.features._modules.items()) - 2
            else:
                last_conv_layer = list(model._modules.keys())[-3]
            sc = ScoreCam(model, last_conv_layer)
            the_map = sc.generate_cam(inputs, targets)
        elif args.method == "cameras":
            sys.path.insert(0, './CAMERAS/')
            from CAMERAS import CAMERAS
            if args.arch == "alexnet":
                last_conv_layer = "features." + list(model.features._modules.keys())[-2]
            else:
                last_conv_layer = list(model._modules.keys())[-3]
            cameras = CAMERAS(model, last_conv_layer)
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
