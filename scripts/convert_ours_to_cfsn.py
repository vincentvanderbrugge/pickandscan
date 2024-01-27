import os
import argparse
import glob

import imageio
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
from new_utils.dataloaders import IPhoneData

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


def convert_cfsn_to_cfsn(input_path, output_path):

    in_depth_dir = os.path.join(input_path, "depth")
    in_rgb_dir = os.path.join(input_path, "rgb")

    out_depth_dir = os.path.join(output_path, "depth")
    out_rgb_dir = os.path.join(output_path, "rgb")

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    if not os.path.exists(out_depth_dir):
        os.makedirs(out_depth_dir, exist_ok=True)

    if not os.path.exists(out_rgb_dir):
        os.makedirs(out_rgb_dir, exist_ok=True)

    img_paths = glob.glob(os.path.join(in_rgb_dir, "*.png"))
    img_paths.sort()

    for path in tqdm(img_paths):
        img = Image.open(path)
        img = np.array(img)
        # img = cv2.resize(img, (480, 640))
        frame_id = int(re.findall('\d+', os.path.basename(path))[0])
        img_out_path = os.path.join(out_rgb_dir, '%05d.jpg' % frame_id)
        imageio.imwrite(img_out_path, img)

    depth_paths = glob.glob(os.path.join(in_depth_dir, "*.png"))
    depth_paths.sort()

    for path in tqdm(depth_paths):
        img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        img = img / 1000
        img = img.astype("float32")
        # img = cv2.resize(img, (480, 640))
        frame_id = int(re.findall('\d+', os.path.basename(path))[0])
        img_out_path = os.path.join(out_depth_dir, '%05d.exr' % frame_id)
        imageio.imwrite(img_out_path, img)

    intrinsics = IPhoneData(input_path).get_intrinsics()
    with open(os.path.join(output_path, "calibration.txt"), 'w') as file:
        # Write each integer to the file, separated by a space
        fx, fy, cx, cy = int(intrinsics[0,0]), int(intrinsics[1,1]), int(intrinsics[0,2]), int(intrinsics[1,2])
        file.write(" ".join(map(str, [fx, fy, cx, cy])))
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control conversion cofusion to cofusion.")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    convert_cfsn_to_cfsn(args.input, args.output)