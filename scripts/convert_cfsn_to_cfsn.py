import os
import argparse
import glob

import imageio
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

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

    img_paths = glob.glob(os.path.join(in_rgb_dir, "*.jpg"))
    img_paths.sort()

    for path in tqdm(img_paths):
        img = Image.open(img_paths[0])
        img = np.array(img)
        # img = cv2.resize(img, (192, 256))
        img_out_path = os.path.join(out_rgb_dir, os.path.basename(path))
        imageio.imwrite(img_out_path, img)

    depth_paths = glob.glob(os.path.join(in_depth_dir, "*.exr"))
    depth_paths.sort()

    for path in tqdm(depth_paths):
        img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        img = img.astype("float32")
        # img = cv2.resize(img, (192, 256))
        img_out_path = os.path.join(out_depth_dir, os.path.basename(path))
        imageio.imwrite(img_out_path, img)


    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control conversion cofusion to cofusion.")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    convert_cfsn_to_cfsn(args.input, args.output)