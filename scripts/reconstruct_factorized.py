import argparse
from argparse import Namespace
import yaml
import os
import sys

from main.step01_preprocess_scan import preprocess
from main.step02_gradslam_poses import estimate_poses_gradslam
from main.step03_hand_segmentation import segment_hands
from main.step04_initial_pointcloud import generate_initial_pointcloud
from main.step05_object_masks import segment_object_candidates
from main.step06_track_separation import separate_tracks
from main.step07_best_frame_selection import select_best_frames
from main.step08_xmem_tracking import track_masks_xmem
from main.step08b_non_maximum_suppression import non_maximum_suppression
from main.step09_bundlesdf_preparation import create_helper_visualizations, prepare_bundlesdf_input
from new_utils.utils import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    parser.add_argument('-n', '--num_items', type=int, default=-1)  # up to which timeframe to process; -1 for all
    parser.add_argument('-s', '--step', type=int, default=0)  # which step to run next (which state to achieve next)
    parser.add_argument('--skip_gif', action='store_true')

    args = parser.parse_args()

    # d = yaml.safe_load(open(args.config, 'r'))
    # d['config'] = args.config
    # args = vars(args)
    # args.update(d)
    # args = Namespace(**args)
    config = load_config(args)

    # config.scene_dir = args.scene_dir

    state = load_or_initialize_scene_dir(config)

    # if args.step == -1:
    #     state = get_state(config.scene_dir)
    # else:
    #     state = args.step - 1
    #
    # # if config doesn't exist, make it
    # if not os.path.exists(config.scene_dir):
    #     os.makedirs(config.scene_dir, exist_ok=True)
    #
    # if not os.path.exists(os.path.join(config.scene_dir, 'config.yaml')):
    #     config_dict = yaml.safe_load(open(args.config, 'r'))
    #     with open(os.path.join(config.scene_dir, "config.yaml"), 'w') as file:
    #         yaml.safe_dump(config_dict, file, sort_keys=False)

    if state < 1:
        # Step01: Preprocess (unpack directories, make gif of RGB images) - gradslam5;0o/pL&MUY<*i9o0p/'py
        print("Step01: Preprocessing started.")
        write_state(config.scene_dir, 0)
        preprocess(config.input, config.scene_dir, config.config, num_items=config.num_items, skip_gif=config.skip_gif)
        write_state(config.scene_dir, 1)

    if state < 2:
        # Step02: Gradslam (unpacks the poses from the ARKit - gradslam5
        print("Step02: Gradslam started.")
        estimate_poses_gradslam(config.scene_dir)
        write_state(config.scene_dir, 2)

    if state < 3:
        # Step03: Hand segmentation (segments out hand in each image, xmem_saves masks & visualizations)
        print("Step03: Hand segmentation started.")
        segment_hands(config.scene_dir)
        write_state(config.scene_dir, 3)

    if state < 4:
        # Step04: Generate & save initial pointcloud (pointcloud before anything is moved; for movement detection
        print("Step04: Initial point cloud extraction started.")
        generate_initial_pointcloud(config.scene_dir, config)
        write_state(config.scene_dir, 4)

    if state < 5:
        # Step05: Heuristic moving mask estimation
        print("Step05: Heuristic moving mask started.")
        segment_object_candidates(config.scene_dir)
        write_state(config.scene_dir, 5)

    if state < 6:
        # Step06: Track separation
        print("Step06: Track separation started.")
        separate_tracks(config.scene_dir, config)
        write_state(config.scene_dir, 6)

    if state < 7:
        # Step07: Best frame selection
        print("Step07: Best frame selection started.")
        select_best_frames(config.scene_dir, config)
        write_state(config.scene_dir, 7)

    if state < 8:
        # Step08: XMem tracking
        print("Step08: XMem tracking started.")
        track_masks_xmem(config.scene_dir, config)
        write_state(config.scene_dir, 8)

    if state < 8.5:
        # Step08b: Object mask non-maximum suppression
        print("Step08b: non-maximum suppression started.")
        non_maximum_suppression(config.scene_dir, config)
        write_state(config.scene_dir, 8.5)

    if state < 9:
        # Step09: helper visualizations / bundlesdf preparation
        print("Step09: bundleprep started.")
        # create_helper_visualizations(config.scene_dir)
        prepare_bundlesdf_input(config.scene_dir)
        write_state(config.scene_dir, 9)

    pass
