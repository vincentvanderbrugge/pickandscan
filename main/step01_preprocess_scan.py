import os
import sys

sys.path.append("/local/home/vincentv/code/motion_segment2")
import zipfile
# from utils.data.iphone_recordings.preprocess_r3d_file import *
# from ...utils.data.iphone_recordings.preprocess_r3d_file import *
from new_utils.preprocess_r3d_file import *

import glob
import json
import os
import yaml
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import liblzfse  # https://pypi.org/project/pyliblzfse/
import numpy as np
import png  # pip install pypng
import torch
import tyro
from natsort import natsorted
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm, trange
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from new_utils.utils import load_image


def gif_from_image_dir(image_dir, out_path, nth_frame=1, slow_down=1.0):
    # image_dir = "C://Users//Vincent//code//motionsegment//data//krishayush1//rgb"
    # out_path = "C://Users//Vincent//code//motionsegment//data//krishayush1//video.gif"

    image_paths = glob.glob(os.path.join(image_dir, "*.png"))
    image_paths.sort(key=lambda filename: int(filename.split("/")[-1][:-4]))

    # Create the figure and axes objects
    fig, ax = plt.subplots()
    # fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    # Set the initial image
    im = ax.imshow(load_image(image_paths[0]), animated=True)

    def update(i):
        image = load_image(image_paths[i * nth_frame])
        im.set_array(image)
        plt.title(f'Frame {i}')
        return im,

    print("Started creating gif.")

    # Create the animation object
    animation_fig = animation.FuncAnimation(fig, update, frames=len(image_paths) // nth_frame,
                                            interval=int(30 * nth_frame * slow_down), blit=True,
                                            repeat_delay=10, )

    animation_fig.save(out_path)

    print("Finished creating gif.")


def preprocess_r3d_folder(datapath, num_items=-1):
    metadata = None
    with open(os.path.join(datapath, "metadata"), "r") as f:
        metadata = json.load(f)

    # Keys in metadata dict
    # h, w, K, fps, dw, dh, initPose, poses, cameraType, frameTimestamps
    # print(metadata.keys())

    poses = get_poses(metadata)
    intrinsics_dict = get_intrinsics(metadata)

    color_paths = natsorted(glob.glob(os.path.join(datapath, "rgbd", "*.jpg")))
    depth_paths = natsorted(glob.glob(os.path.join(datapath, "rgbd", "*.depth")))
    conf_paths = natsorted(glob.glob(os.path.join(datapath, "rgbd", "*.conf")))

    os.makedirs(os.path.join(datapath, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(datapath, "conf"), exist_ok=True)
    os.makedirs(os.path.join(datapath, "depth"), exist_ok=True)
    os.makedirs(os.path.join(datapath, "poses"), exist_ok=True)

    cfg = {}
    cfg["dataset_name"] = "record3d"
    cfg["camera_params"] = {}
    cfg["camera_params"]["image_height"] = intrinsics_dict["h"]
    cfg["camera_params"]["image_width"] = intrinsics_dict["w"]
    cfg["camera_params"]["fx"] = intrinsics_dict["fx"].item()
    cfg["camera_params"]["fy"] = intrinsics_dict["fy"].item()
    cfg["camera_params"]["cx"] = intrinsics_dict["cx"].item()
    cfg["camera_params"]["cy"] = intrinsics_dict["cy"].item()
    cfg["camera_params"]["png_depth_scale"] = 1000.0
    print(cfg)
    with open(os.path.join(datapath, "dataconfig.yaml"), "w") as f:
        yaml.dump(cfg, f)

    num_items = len(color_paths) if num_items == -1 else num_items

    for i in trange(num_items):
        color = load_color(color_paths[i])
        depth = load_depth(depth_paths[i])
        conf = load_conf(conf_paths[i])
        basename = os.path.splitext(os.path.basename(color_paths[i]))[0]
        # color_path = os.path.splitext(os.path.basename(color_paths[i]))[0] + ".png"
        write_color(os.path.join(datapath, "rgb", basename + ".png"), color)
        # depth_path = os.path.splitext(os.path.basename(depth_paths[i]))[0] + ".png"
        write_depth(os.path.join(datapath, "depth", basename + ".png"), depth)
        # conf_path = os.path.splitext(os.path.basename(conf_paths[i]))[0] + ".npy"
        write_conf(os.path.join(datapath, "conf", basename + ".npy"), conf)
        write_pose(os.path.join(datapath, "poses", basename + ".npy"), poses[i])
        # c2w = poses[i]
        # frame = {
        #     "file_path": os.path.join("rgb", color_path),
        #     "depth_path": os.path.join("depth", depth_path),
        #     "conf_path": os.path.join("conf", conf_path),
        #     "transform_matrix": c2w.tolist(),
        # }
        # frames.append(frame)


def preprocess(input_path, output_path, config_path, num_items=-1, skip_gif=False):
    output_dir = output_path.replace("\\", "//")
    path_to_zip = input_path.replace("\\", "//")

    os.makedirs(output_dir, exist_ok=True)

    shutil.copyfile(config_path, os.path.join(output_path, "config.yaml"))

    # Unzip
    with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    # Preprocess folder
    preprocess_r3d_folder(output_dir, num_items=num_items)

    if not skip_gif:
        # Create video gif
        gif_from_image_dir(os.path.join(output_dir, "rgb"), os.path.join(output_dir, "video.gif"), slow_down=1.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')

    # path_to_zip = "C://Users//Vincent//Downloads//2023-09-07--18-27-49.r3d"
    # output_dir = "C://Users//Vincent//code//motionsegment//data//krishayush1"

    args = parser.parse_args()

    path_to_zip = args.input
    output_dir = args.output

    # path_to_zip = path_to_zip.replace("")
    output_dir = output_dir.replace("\\", "//")
    path_to_zip = path_to_zip.replace("\\", "//")

    os.makedirs(output_dir, exist_ok=True)

    # Unzip
    with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    # # Preprocess folder
    preprocess_r3d_folder(output_dir)

    # Create video gif
    gif_from_image_dir(os.path.join(output_dir, "rgb"), os.path.join(output_dir, "video.gif"), slow_down=1.5)

