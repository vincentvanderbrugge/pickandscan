from utils.data.dataloaders import IPhoneTracks, KinectPreprocessedData, IPhoneData, IPhoneTracks
from segmentation.segment import Segmenter
import matplotlib.pyplot as plt
from utils.utils import create_masked_frame
from matplotlib.animation import FuncAnimation
from IPython import display
from skimage.io import imsave
import os
import numpy as np
from tqdm import tqdm
import shutil
import argparse
from argparse import Namespace
import json



if __name__ == '__main__':

    # Configure inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    args = parser.parse_args()

    config = {'input': '/local/home/vincentv/code/motion_segment2/data/dayfour2',
              'iou_threshold': 0.75,
              'n_consecutive_frames': 30,
              'track_start': 300,
              'track_end': 700}
    final_config = args.__dict__
    final_config.update(config)
    args = Namespace(**final_config)

    scene_dir = args.input
    comparison_frame_id = 0
    # frame_id = 582

    process_dir = os.path.join(scene_dir, 'process_dir')
    dataset_name = scene_dir.split('/')[-1]

    scene_data = IPhoneData(scene_dir)
    # scene_data.load_computed_tracks()
    scene_data.load_poses()

    segmenter = Segmenter(scene_data, distance_treshold=0.02, n_comparisons=5)

    # # Get object mask
    # object_mask = segmenter.get_manipulated_object_mask_from_single_comparison(frame_number=frame_id,
    #                                                                comparison_frame_index=comparison_frame_id)

    # # TODO implement for multiple detected tracks
    # track_info = json.load(open(os.path.join(process_dir, "track_info.json")))
    # selected_frame_id_for_xmem = track_info["tracks"][0]["selected_frame_for_xmem"]
    # frame_id = selected_frame_id_for_xmem

    selected_frame_id_for_xmem = 1135
    frame_id = selected_frame_id_for_xmem


    object_mask = segmenter.scene_loader.get_single_object_candidate_mask(selected_frame_id_for_xmem)

    # Create mask visualization
    img = scene_data.get_single_color_frame(str(frame_id))
    visualization = create_masked_frame(img, object_mask)
    plt.imshow(visualization)
    plt.show()

    # Make XMEM folder structure
    if not os.path.isdir(os.path.join(process_dir, "xmem", f"xmem_input_{dataset_name}", "JPEGImages", "video1")):
        os.makedirs(os.path.join(process_dir, "xmem", f"xmem_input_{dataset_name}", "JPEGImages", "video1"), exist_ok=True)

    if not os.path.isdir(os.path.join(process_dir, "xmem", f"xmem_input_{dataset_name}", "Annotations", "video1")):
        os.makedirs(os.path.join(process_dir, "xmem", f"xmem_input_{dataset_name}", "Annotations", "video1"), exist_ok=True)

    # Save mask
    imsave(
        os.path.join(process_dir, "xmem", f"xmem_input_{dataset_name}", "Annotations", "video1", '%05d.png' % frame_id),
        object_mask.astype(np.uint8))

    # Save mask visualization
    imsave(os.path.join(process_dir, "xmem", f'{frame_id}_visualization.png'), visualization)

    # Save images as .jpg
    print("Copying frames to xmem folder hierarchy.")
    for frame in tqdm(scene_data.frames):
        img = scene_data.get_single_color_frame(frame)
        imsave(os.path.join(process_dir, "xmem", f"xmem_input_{dataset_name}", "JPEGImages", "video1", '%05d.jpg' % int(frame)), img)

    target_dir = os.path.join(process_dir, "xmem", f"xmem_input_{dataset_name}")
    target_zip = os.path.join(process_dir, "xmem", f"xmem_input_{dataset_name}.zip")
    archived = shutil.make_archive(target_dir, 'zip', target_dir)

