import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import models.imagenet as customized_models

import numpy as np

from utils import Bar


# Models
customized_models_names = sorted(name for name in customized_models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(customized_models.__dict__[name]))
for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = customized_models_names

#use_cuda = torch.cuda.is_available()
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser(description='ABN ImageNet')
parser.add_argument('-o', '--confidences_out', type=str, default='confidences.txt', help='path to output file')
# Datasets
parser.add_argument('-d', '--data', type=str, required=True, help='path to dataset',)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--model', '-m', default='', type=str, metavar='PATH',
                    help='path to model (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


def main():
    im_size = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(im_size),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.ImageFolder(root=args.data, transform=transform)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                baseWidth=args.base_width,
                cardinality=args.cardinality,
                )
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    model = torch.nn.DataParallel(model)
    if use_cuda:
        model = model.cuda()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Load model
    title = 'ImageNet-' + args.arch
    if args.model:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.model), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise Exception("Pass in a checkpoint!")

    print(model)

    # switch to evaluate mode
    model.eval()

    results = []

    bar = Bar('Processing', max=len(loader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            output = nn.functional.softmax(model(inputs)[1]).cpu().numpy()[0]

            results.append((dataset.imgs[batch_idx][0], *output))

            bar.next()
        bar.finish()

    results.sort()
    with open(args.confidences_out, "w") as outfile:
        outfile.write("\n".join("\t".join(map(str, res)) for res in results))
    print("Wrote confidences to", args.confidences_out)


if __name__ == '__main__':
    main()
