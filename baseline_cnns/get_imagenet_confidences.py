"""
Evaluates ImageNet models on a given directory of images.
"""

from __future__ import print_function

import argparse
import os
import pickle
from pprint import pprint
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import models.imagenet as customized_models

import timm

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


# Parse arguments
parser = argparse.ArgumentParser(description='Evaluation of ImageNet models')

parser.add_argument('-o', '--confidences_out', type=str, default='confidences.txt', help='path to output file')
# Datasets
parser.add_argument('-d', '--data', type=str, required=True, help='path to dataset',)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    help='model architecture: ' +
                        ' (default: resnet)')
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
                                     std=[0.229, 0.224, 0.225])  # imagenet normalization
    transform = transforms.Compose([
        transforms.Resize(224),  # For input into imagenet models
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.ImageFolder(root=args.data, transform=transform)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch == "resnet101":
        model = models.resnet101(pretrained=True)
    elif args.arch == "alexnet":
        model = models.alexnet(pretrained=True)
    elif args.arch == "vgg16_bn":
        model = models.vgg16_bn(pretrained=True)
    elif args.arch == "efficientnet":
        model = timm.create_model("efficientnet_b0", pretrained=True)
    elif args.arch == "vit":
        model = timm.create_model("vit_small_patch16_224", pretrained=True)
    else:
        raise Exception("Invalid architecture chosen")

    if use_cuda:
        model = model.cuda()
    else:
        model = model.cpu()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    print(model)

    batch_time = AverageMeter()

    results = []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        bar = Bar('Processing', max=len(loader))
        for batch_idx, (inputs, targets) in enumerate(loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            output = nn.functional.softmax(model(inputs)).cpu().numpy()[0]

            results.append((dataset.imgs[batch_idx][0], *output))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                        batch=batch_idx + 1,
                        size=len(loader),
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        )
            bar.next()
        bar.finish()

    results.sort()
    with open(args.confidences_out, "w") as outfile:
        outfile.write("\n".join("\t".join(map(str, res)) for res in results))
    print("Wrote confidences to", args.confidences_out)


if __name__ == '__main__':
    main()
