import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from model1 import AttnVGG_before
from model2 import AttnVGG_after
from utilities import *

from imageio import imread
from PIL import Image
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="LearnToPayAttn-CIFAR100")

parser.add_argument("--img", "-i", help="path to image")
parser.add_argument("--img_dir", "-d", help="path to images")
parser.add_argument("--output_dir", "-o", help="path to output attention heatmaps")
parser.add_argument("--model", "-m", required=True, help="path to model")
parser.add_argument("--attn_mode", type=str, default="before", help='insert attention modules before OR after maxpooling layers')

parser.add_argument("--normalize_attn", action='store_true', help='if True, attention map is normalized by softmax; otherwise use sigmoid')

opt = parser.parse_args()


def main():
    im_size = 32
    mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    transform_test = transforms.Compose([
        #transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    print('done')


    ## load network
    print('\nloading the network ...\n')
    # (linear attn) insert attention befroe or after maxpooling?
    # (grid attn only supports "before" mode)
    if opt.attn_mode == 'before':
        print('\npay attention before maxpooling layers...\n')
        net = AttnVGG_before(im_size=im_size, num_classes=100,
            attention=True, normalize_attn=opt.normalize_attn, init='xavierUniform')
    elif opt.attn_mode == 'after':
        print('\npay attention after maxpooling layers...\n')
        net = AttnVGG_after(im_size=im_size, num_classes=100,
            attention=True, normalize_attn=opt.normalize_attn, init='xavierUniform')
    else:
        raise NotImplementedError("Invalid attention mode!")
    print('done')


    ## load model
    print('\nloading the model ...\n')
    state_dict = torch.load(opt.model, map_location=str(device))
    # Remove 'module.' prefix
    state_dict = {k[7:]: v for k, v in state_dict.items()}
    net.load_state_dict(state_dict)
    net = net.to(device)
    net.eval()
    print('done')


    model = net

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # base factor
    if opt.attn_mode == 'before':
        min_up_factor = 1
    else:
        min_up_factor = 2
    # sigmoid or softmax
    if opt.normalize_attn:
        vis_fun = visualize_attn_softmax
    else:
        vis_fun = visualize_attn_sigmoid

    if opt.output_dir:
        print("\nwill save heatmaps\n")

    if opt.img:
        img_dir = ""
        filenames = [opt.img]
    else:
        #img_dir = opt.img_dir
        #filenames = os.listdir(img_dir)

        paths = []
        filenames = []
        for dirpath, dirnames, fns in os.walk(opt.img_dir):
            for f in fns:
                if f.endswith(".jpg"):
                    filenames.append(f)
                    paths.append(os.path.join(dirpath, f))

    display_fig = len(filenames) == 1

    with torch.no_grad():
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

            if opt.output_dir:
                #file_prefix = os.path.join(opt.output_dir, os.path.splitext(os.path.basename(filename))[0])
                num = int(filename[filename.index("r_")+2:-4])
                subdir = os.path.join(opt.output_dir, f"img{num}/")
                if not os.path.exists(subdir):
                    os.mkdir(subdir)
                arch = "ltpavgg"
                fn = f"img{num}_network_{arch}_method_c2.mat"
                outpath = os.path.join(subdir, fn)
            else:
                file_prefix = None

            batch = image[np.newaxis, :, :, :]
            __, c1, c2, c3 = model(batch)
            if display_fig:
                fig, axs = plt.subplots(1, 4)
                axs[0].imshow(orig_img)
            #if c1 is not None:
            #    attn1 = vis_fun(img, c1, up_factor=min_up_factor, nrow=1, hm_file=None if file_prefix is None else file_prefix + "_c1.npy")
            #    if display_fig:
            #        axs[1].imshow(attn1.numpy().transpose(1, 2, 0))
            if c2 is not None:
                #attn2 = vis_fun(img, c2, up_factor=min_up_factor*2, nrow=1, hm_file=None if file_prefix is None else file_prefix + "_c2.npy")
                attn2 = vis_fun(img, c2, up_factor=min_up_factor*2, nrow=1, hm_file=outpath)
                if display_fig:
                    axs[2].imshow(attn2.numpy().transpose(1, 2, 0))
            #if c3 is not None:
            #    attn3 = vis_fun(img, c3, up_factor=min_up_factor*4, nrow=1, hm_file=None if file_prefix is None else file_prefix + "_c3.npy")
            #    if display_fig:
            #        axs[3].imshow(attn3.numpy().transpose(1, 2, 0))

            if display_fig:
                plt.show()


if __name__ == "__main__":
    main()
