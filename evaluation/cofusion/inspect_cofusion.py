import os
import argparse
import glob

import imageio
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import open3d as o3d
import re
from new_utils.utils import *


def overlay_gif(img_paths, mask_paths, label, out_path, nth_frame=1, slow_down=1.0):
    fig, ax = plt.subplots()
    # fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    # Set the initial image
    try:
        im = ax.imshow(create_masked_frame(load_image(img_paths[0]), load_image(mask_paths[0]) == label), animated=True)
    except:
        pass

    def update(i):
        image = create_masked_frame(load_image(img_paths[i]), load_image(mask_paths[i]) == label)
        im.set_array(image)
        plt.title(f'Frame {i}')
        return im,

    print("Started creating gif.")

    # Create the animation object
    animation_fig = animation.FuncAnimation(fig, update, frames=len(img_paths) // nth_frame,
                                            interval=int(30 * nth_frame * slow_down), blit=True,
                                            repeat_delay=10, )

    animation_fig.save(out_path)

    print("Finished creating gif.")
    pass


def inspect(cfn_out_path, scene_dir):
    all_mask_paths = glob.glob(os.path.join(cfn_out_path, "outSeg*"))
    all_mask_paths.sort(key= lambda path: int(re.findall('\d+', os.path.basename(path))[0]))

    inspection_dir = os.path.join(cfn_out_path, "inspection")
    overlay_dir = os.path.join(inspection_dir, "overlay")
    img_dir = os.path.join(scene_dir, "rgb")

    if not os.path.exists(inspection_dir):
        os.makedirs(inspection_dir, exist_ok=True)

    if not os.path.exists(overlay_dir):
        os.makedirs(overlay_dir, exist_ok=True)

    object_occurrences = {}

    pattern = re.compile(r'\d+')

    for mask_path in all_mask_paths:
        match = re.search(pattern, os.path.basename(mask_path))
        frame_id = int(match.group())

        mask = np.array(Image.open(mask_path))
        labels = np.unique(mask)

        for label in labels:
            if label not in list(object_occurrences.keys()):
                object_occurrences[label] = [frame_id]
            else:
                object_occurrences[label].append(frame_id)

    for label in list(object_occurrences.keys()):
        if label == 0:
            continue
        img_paths = []
        # for frame_id in object_occurrences[label]:
        #     img_paths.append(os.path.join(img_dir))
        img_paths = [img_path for img_path in glob.glob(os.path.join(img_dir, "*")) if int(re.findall('\d+', os.path.basename(img_path))[0]) in object_occurrences[label]]
        mask_paths = [mask_path for mask_path in all_mask_paths if int(re.findall('\d+', os.path.basename(mask_path))[0]) in object_occurrences[label]]

        img_paths.sort(key=lambda path: int(re.findall('\d+', os.path.basename(path))[0]))
        mask_paths.sort(key=lambda path: int(re.findall('\d+', os.path.basename(path))[0]))

        out_path = os.path.join(overlay_dir, f"overlay_object{label}.gif")

        print(f"Creating overlay_object{label}.gif")
        overlay_gif(img_paths, mask_paths, label, out_path)
    pass




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control conversion cofusion to cofusion.")
    # parser.add_argument("--pcd", required=True)
    # parser.add_argument("--mask", required=True)
    # parser.add_argument("--rgb", required=True)
    parser.add_argument("--cfn_out", required=True)
    parser.add_argument("--scene", required=True)
    args = parser.parse_args()

    inspect(args.cfn_out, args.scene)