from utils.data.dataloaders import IPhoneTracks, KinectPreprocessedData, IPhoneData, IPhoneTracks
from segmentation.segment import Segmenter
import matplotlib.pyplot as plt
from utils.utils import create_masked_frame
from matplotlib.animation import FuncAnimation
from IPython import display
from skimage.io import imsave
import matplotlib.animation as animation
import os
from utils.utils import load_image
from tqdm import tqdm
import numpy as np
import argparse
from argparse import Namespace
import shutil


def get_visualization(frame_id, scene_data, mask_dir):
    object_mask = load_image(os.path.join(mask_dir, '%05d.png' % frame_id)) == 1

    img = scene_data.get_single_color_frame(str(frame_id))
    visualization = create_masked_frame(img, object_mask)
    return visualization


if __name__ == '__main__':

    # Config
    config = {'input': 'C://Users//Vincent//code//motionsegment//data//multi2',
              'start_frame': 0,
              'nth_frame': 1,
              'slow_down': 2.5,
              'manipulator': 'fingertips',
              'comparison_frame_id': 0}

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-i', '--input')
    args = parser.parse_args()
    final_config = args.__dict__
    final_config.update(config)
    args = Namespace(**final_config)

    scene_dir = args.input
    process_dir = os.path.join(scene_dir, 'process_dir')
    mask_dir = os.path.join(process_dir, 'xmem//xmem_output//video1')

    scene_data = IPhoneData(scene_dir)

    frame_ids = [int(filename[:5]) for filename in os.listdir(mask_dir)]

    os.makedirs(os.path.join(process_dir, 'bundlesdf_input', 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(process_dir, 'bundlesdf_input', 'depth'), exist_ok=True)
    os.makedirs(os.path.join(process_dir, 'bundlesdf_input', 'masks'), exist_ok=True)

    with open(os.path.join(process_dir, 'bundlesdf_input', 'cam_K.txt'), 'wb') as file:
        np.savetxt(file, scene_data.get_intrinsics(), delimiter=' ', newline='\n', fmt='%.5e')

    print("Copying frames to bundlesdf input folder.")
    for frame_id in tqdm(frame_ids):
        color_frame = scene_data.get_single_color_frame(scene_data.frames[frame_id])
        depth_frame = scene_data.get_single_depth_frame(scene_data.frames[frame_id])
        mask_frame = load_image(os.path.join(mask_dir, '%05d.png' % frame_id))

        imsave(os.path.join(process_dir, 'bundlesdf_input', 'rgb', '%05d.png' % int(frame_id)), color_frame, check_contrast=False)
        imsave(os.path.join(process_dir, 'bundlesdf_input', 'depth', '%05d.png' % int(frame_id)), depth_frame, check_contrast=False)
        imsave(os.path.join(process_dir, 'bundlesdf_input', 'masks', '%05d.png' % int(frame_id)), mask_frame, check_contrast=False)
        pass

    target_dir = os.path.join(process_dir, 'bundlesdf_input')
    archived = shutil.make_archive(target_dir, 'zip', target_dir)




