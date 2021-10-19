# Download BMS executable from https://cs-people.bu.edu/jmzhang/BMS/BMS.html
# and update BMS_EXE before running this script.

import os
import subprocess
import sys
import tempfile

import imageio
import scipy.io as sio

BMS_EXE = "/home/czhao/Downloads/bms/BMS_CODE/BMS"

input_path = sys.argv[1] + "/"
output_path = sys.argv[2]

SAMPLE_STEP_SIZE = 8
MAX_DIM = 405
DILATION_WIDTH_1 = max(round(7 * MAX_DIM / 400), 1)
DILATION_WIDTH_2 = max(round(9 * MAX_DIM / 400), 1)
BLUR_STD = round(9 * MAX_DIM / 400)
COLOR_SPACE = 2
WHITENING = 1

with tempfile.TemporaryDirectory() as tmpdir:
    print("Running BMS...")
    args = [str(arg) for arg in [BMS_EXE, input_path, tmpdir + "/", SAMPLE_STEP_SIZE, DILATION_WIDTH_1, DILATION_WIDTH_2, BLUR_STD, COLOR_SPACE, WHITENING, MAX_DIM]]
    subprocess.run(args, check=True)

    print("Converting maps...")
    for name in os.listdir(tmpdir):
        num = int(name[name.index("r_")+2:-4])
        infile = os.path.join(tmpdir, name)
        subdir = os.path.join(output_path, f"img{num}/")
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        outfile = os.path.join(subdir, f"img{num}_method_bms.mat")
        the_map = imageio.imread(infile)
        data = {"bms": the_map}
        sio.savemat(outfile, data)
