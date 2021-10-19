import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models
import models.imagenet as customized_models

import cv2
from imageio import imread
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import scipy.io as sio

from utils import Bar, Logger, savefig


# Models
customized_models_names = sorted(name for name in customized_models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(customized_models.__dict__[name]))
for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = customized_models_names

parser = argparse.ArgumentParser(description='PyTorch ImageNet Attention')
parser.add_argument('--model', '-m', default='', type=str, metavar='PATH',
                    help='path to model (default: none)')
parser.add_argument('--img', '-i', help='path to image')
parser.add_argument('--img_dir', '-d', help='path to images')
parser.add_argument('--output_dir', '-o', help='path to output attention heatmaps')
parser.add_argument('--initial_resize', '-r', type=int, help='lower the resolution of input images')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Data
    im_size = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        #transforms.ToTensor(),
        normalize,
    ])

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

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Load model
    title = 'ImageNet-' + args.arch
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
        if args.initial_resize is not None:
            img = np.array(Image.fromarray(img).resize((args.initial_resize, args.initial_resize)))
        img = np.array(Image.fromarray(img).resize((im_size, im_size)))
        orig_img = img.copy()
        img = img.transpose(2, 0, 1)
        img = img / 255.
        img = torch.FloatTensor(img).to(device)
        image = transform_test(img)  # (3, 224, 224)

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

        attn_img = visualize_attn(img, attention[0], up_factor=224/attention[0].shape[2], hm_file=outpath)
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
