import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import models.cifar as models

import cv2
from imageio import imread
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import scipy.io as sio

from utils import Bar, Logger, savefig


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Attention')
parser.add_argument('--model', '-m', default='', type=str, metavar='PATH',
                    help='path to model (default: none)')
parser.add_argument('--img', '-i', help='path to image')
parser.add_argument('--img_dir', '-d', help='path to images')
parser.add_argument('--output_dir', '-o', help='path to output attention heatmaps')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet)')
parser.add_argument('--depth', type=int, default=100, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Data
    im_size = 32
    transform_test = transforms.Compose([
        #transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    num_classes = 100

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Load model
    title = 'cifar-10-' + args.arch
    if args.model:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.model), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.model)
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint['state_dict'])

    if args.output_dir:
        print("\nwill save heatmaps\n")

    # switch to evaluate mode
    model.eval()

    if args.img:
        img_dir = ""
        filenames = [args.img]
    else:
        #img_dir = args.img_dir
        #filenames = os.listdir(img_dir)

        paths = []
        filenames = []
        for dirpath, dirnames, fns in os.walk(args.img_dir):
            for f in fns:
                if f.endswith(".jpg"):
                    filenames.append(f)
                    paths.append(os.path.join(dirpath, f))

    display_fig = len(filenames) == 1

    bar = Bar('Processing', max=len(filenames))
    for i, filename in enumerate(filenames):
        ## load image
        #path = os.path.join(img_dir, filename)
        path = paths[i]
        img = imread(path)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = np.array(Image.fromarray(img).resize((im_size, im_size)))
        orig_img = img.copy()
        img = img.transpose(2, 0, 1)
        img = img / 255.
        img = torch.FloatTensor(img).to(device)
        image = transform_test(img)  # (3, 32, 32)

        # compute output
        batch = image[np.newaxis, :, :, :]
        _, outputs, attention = model(batch)

        if args.output_dir:
            #outpath = os.path.join(args.output_dir, os.path.splitext(os.path.basename(filename))[0] + ".npy")
            num = int(filename[filename.index("r_")+2:-4])
            subdir = os.path.join(args.output_dir, f"img{num}/")
            if not os.path.exists(subdir):
                os.mkdir(subdir)
            arch = f"abn{args.arch}"
            fn = f"img{num}_network_{arch}_method_active.mat"
            outpath = os.path.join(subdir, fn)
        else:
            outpath = None

        attn_img = visualize_attn(img, attention, up_factor=32/attention.shape[2], hm_file=outpath)
        if display_fig:
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(orig_img)
            axs[1].imshow(attn_img)
            plt.show()

        bar.next()
    bar.finish()


def visualize_attn(img, attn, up_factor, hm_file=None):
    img = img.permute((1,2,0)).cpu().detach().numpy()
    if up_factor > 1:
        attn = F.interpolate(attn, scale_factor=up_factor, mode="bilinear", align_corners=False)
    if hm_file is not None:
        #np.save(hm_file, attn.cpu().detach().numpy()[0, 0])
        data = {"active": attn.cpu().detach().numpy()[0, 0]}
        sio.savemat(hm_file, data)
    attn = attn[0].permute((1,2,0)).mul(255).byte().cpu().detach().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    # add the heatmap to the image
    vis = 0.6 * img + 0.4 * attn
    return vis


if __name__ == '__main__':
    main()
