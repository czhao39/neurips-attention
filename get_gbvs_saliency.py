# Download a GBVS implementation from https://github.com/shreelock/gbvs
# before running this script.

import os
import sys

import cv2
from saliency_models import gbvs
import scipy.io as sio


input_path = sys.argv[1]
output_path = sys.argv[2]

for name in os.listdir(input_path):
    num = int(name[name.index("r_")+2:-4])
    infile = os.path.join(input_path, name)
    subdir = os.path.join(output_path, f"img{num}/")
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    outfile = os.path.join(subdir, f"img{num}_method_gbvs.mat")

    img = cv2.imread(infile)
    the_map = gbvs.compute_saliency(img)
    data = {"gbvs": the_map}
    sio.savemat(outfile, data)
