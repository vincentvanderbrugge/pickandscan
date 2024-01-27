import argparse
from argparse import Namespace
import yaml
import os
import sys
sys.path.append("/local/home/vincentv/code/motion_segment2")

from step0_preprocess import preprocess
from step00_gradslam import perform_gradslam
from step01_handsegmentation import perform_handsegmentation
from step02_initial_pointcloud import generate_initial_pointcloud
from step03_object_mask_candidates import get_object_candidate_masks
from step04_track_separation import perform_track_separation
from step05_best_frame_selection import perform_best_frame_selection
from stepa_xmem import perform_xmem_tracking
from stepb_bundleprep import create_helper_visualizations, prepare_bundlesdf_input


def write_state(scene_dir, state):
    config_dict = yaml.safe_load(open(os.path.join(scene_dir, "config.yaml"), 'r'))
    config_dict["state"] = state
    yaml.safe_dump(config_dict, open(os.path.join(scene_dir, "config.yaml"), 'w'), sort_keys=False)
    return


def get_state(scene_dir):

    if not os.path.exists(scene_dir):
        return -1

    try:
        yaml.safe_load(open(os.path.join(scene_dir, "config.yaml"), 'r'))
    except FileNotFoundError:
        raise FileNotFoundError("Scene dir was created but without config yaml.")

    config_dict = yaml.safe_load(open(os.path.join(scene_dir, "config.yaml"), 'r'))

    try:
        return config_dict["state"]
    except KeyError:
        return -1


def write_output_to_config(scene_dir, key, value):
    config_dict = yaml.safe_load(open(os.path.join(scene_dir, "config.yaml"), 'r'))
    config_dict["output"][key] = value
    yaml.safe_dump(config_dict, open(os.path.join(scene_dir, "config.yaml"), 'w'), sort_keys=False)
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--input')
    parser.add_argument('-c', '--config')
    parser.add_argument('-n', '--num_items', type=int, default=-1) #up to which timeframe to process; -1 for all
    parser.add_argument('-s', '--step', type=int, default=-1)#which step to run next (which state to achieve next)
    parser.add_argument('--skip_gif', action='store_true')
    # parser.add_argument('-o', '--output')

    args = parser.parse_args()

    d = yaml.safe_load(open(args.config, 'r'))
    d['config'] = args.config
    args = vars(args)
    args.update(d)
    args = Namespace(**args)

    path_to_zip = args.input
    scene_dir = args.scene_dir

    if args.step == -1:
        state = get_state(scene_dir)
    else:
        state = args.step - 1

    # if config doesn't exist, make it
    if not os.path.exists(scene_dir):
        os.makedirs(scene_dir, exist_ok=True)

    if not os.path.exists(os.path.join(scene_dir, 'config.yaml')):
        config_dict = yaml.safe_load(open(args.config, 'r'))
        with open(os.path.join(scene_dir, "config.yaml"), 'w') as file:
            yaml.safe_dump(config_dict, file, sort_keys=False)


    if state < 0:
        # Step0: Preprocess (unpack directories, make gif of RGB images) - gradslam5;0o/pL&MUY<*i9o0p/'py
        print("Step0a: Preprocessing started.")
        write_state(scene_dir, -1)
        preprocess(path_to_zip, scene_dir, args.config, num_items=args.num_items, skip_gif=args.skip_gif)
        write_state(scene_dir, 0)

    if state < 1:
        #Step00: Gradslam (unpacks the poses from the ARKit - gradslam5
        print("Step0b: Gradslam started.")
        perform_gradslam(scene_dir)
        write_state(scene_dir, 1)

    if state < 2:
        #Step01: Hand segmentation (segments out hand in each image, xmem_saves masks & visualizations)
        print("Step1: Hand segmentation started.")
        perform_handsegmentation(scene_dir)
        write_state(scene_dir, 2)

    if state < 3:
        #Step02: Generate & save initial pointcloud (pointcloud before anything is moved; for movement detection
        print("Step2: Initial point cloud extraction started.")
        generate_initial_pointcloud(scene_dir)
        write_state(scene_dir, 3)

    if state < 4:
        #Step03: Heuristic moving mask estimation
        print("Step3: Heuristic moving mask started.")
        get_object_candidate_masks(scene_dir)
        write_state(scene_dir, 4)

    if state < 5:
    # if True:
        #Step04: Track separation
        print("Step4: Track separation started.")
        perform_track_separation(scene_dir)
        write_state(scene_dir, 5)

    if state < 6:
    # if True:
        #Step05: Best frame selection
        print("Step5: Best frame selection started.")
        perform_best_frame_selection(scene_dir)
        write_state(scene_dir, 6)

    if state < 7:
        #Step06: XMem tracking
        print("Step6: XMem tracking started.")
        perform_xmem_tracking(scene_dir)
        write_state(scene_dir, 7)

    if state < 8:
        #Step07: helper visualizations / bundlesdf preparation
        print("Step7: bundleprep started.")
        create_helper_visualizations(scene_dir)
        prepare_bundlesdf_input(scene_dir)

        # write_state(scene_dir, 8)
    pass




