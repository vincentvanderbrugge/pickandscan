from utils.data.dataloaders import IPhoneTracks, KinectPreprocessedData, IPhoneData, IPhoneTracks
from segmentation.segment import Segmenter
import matplotlib.pyplot as plt
from utils.utils import create_masked_frame
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from IPython import display
import os
import argparse
import numpy as np
from skimage.io import imsave
from tqdm import tqdm
from argparse import Namespace
import json
import yaml


def write_output_to_config(scene_dir, key, value):
    config_dict = yaml.safe_load(open(os.path.join(scene_dir, "config.yaml"), 'r'))
    config_dict["output"][key] = value
    yaml.safe_dump(config_dict, open(os.path.join(scene_dir, "config.yaml"), 'w'), sort_keys=False)
    return


def get_visualization(segmenter, frame_id, comparison_frame_id):
    object_mask = segmenter.get_manipulated_object_mask_from_single_comparison(frame_number=frame_id,
                                                                               comparison_frame_index=comparison_frame_id)

    img = scene_data.get_single_color_frame(str(frame_id))
    visualization = create_masked_frame(img, object_mask)
    return visualization


def get_iou(mask_a, mask_b):
    logical_mask_a = mask_a == 1
    logical_mask_b = mask_b == 1
    intersection = np.logical_and(mask_a, mask_b)
    union = np.logical_or(mask_a, mask_b)
    iou = intersection.sum() / union.sum()
    return iou
    # raise NotImplementedError


def longest_true_period(array):
    periods = []

    start_index = 0
    in_period = False

    for i in range(len(array)):

        if array[i]:
            if not in_period:
                start_index = i
                in_period = True

        if not array[i]:
            if in_period:
                periods.append((start_index, i - 1))
                in_period = False

        if i == len(array) - 1:
            if in_period and array[i]:
                periods.append((start_index, i))

    if len(periods) > 0:
        return max(periods, key= lambda period: period[1] - period[0]), periods
    else:
        return None, periods





def plot_with_offset(distances_fingertip_static, arm_entry_index):
    return plt.plot([i for i in range(arm_entry_index, arm_entry_index + len(distances_fingertip_static))],
             distances_fingertip_static)


def perform_best_frame_selection(scene_dir):
    nth_frame = 1
    slow_down = 2.5
    comparison_frame_id = 0
    iou_threshold = 0.75

    # out_dir = os.path.join(scene_dir, 'process_dir', 'object_candidate_masks')
    vis_dir = os.path.join(scene_dir, 'process_dir', 'best_frame_visualizations')
    debugging_vis_dir = os.path.join(scene_dir, 'process_dir', 'object_candidate_masks_debugging')

    scene_data = IPhoneData(scene_dir)

    # scene_data.load_computed_tracks()
    scene_data.load_poses()

    segmenter = Segmenter(scene_data, distance_treshold=0.02, n_comparisons=5)

    n_frames = len(scene_data.frames)

    # os.makedirs(out_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    tracks = yaml.safe_load(open(os.path.join(scene_dir, 'config.yaml'), 'r'))["output"]["object_tracks"]

    object_candidate_masks = []

    visualizations = []

    for track_id, track in enumerate(tracks):

        hand_mask_areas = []
        hand_distances = []
        fingertip_distances = []

        ious = {}

        for frame_id in tqdm(range(track["period"][0], track["period"][1], nth_frame)):

            if not 1 <= frame_id <= len(segmenter.scene_loader.frames) - 1:
                continue

            current_mask = segmenter.scene_loader.get_single_object_candidate_mask(frame_id + 1)
            previous_mask = segmenter.scene_loader.get_single_object_candidate_mask(frame_id)
            next_mask = segmenter.scene_loader.get_single_object_candidate_mask(frame_id + 2)

            ious[frame_id] = get_iou(current_mask, previous_mask)

        iou_array = np.array(list(ious.values()))
        max_period, periods = longest_true_period(iou_array > iou_threshold)

        if len(periods) == 0:
            print("Warning: track {} has no periods with IoU above threshold".format(track_id))
            continue

        # selected_index = (max_period[0] + max_period[1]) // 2
        selected_index = np.argmax(iou_array[max_period[0]:max_period[1]])+max_period[0]
        selected_index_iou = np.max(iou_array[max_period[0]:max_period[1]])

        selected_index += track["period"][0] + 1

        mask = segmenter.scene_loader.get_single_object_candidate_mask(selected_index)
        img = scene_data.get_single_color_frame(str(selected_index))
        visualization = create_masked_frame(img, mask)
        visualizations.append(visualization)
        best_frame_selection_plot(list(ious.values()), track["period"][0], periods, selected_index, selected_index_iou)
        # imsave(os.path.join(out_dir, str(frame_id) + '.png'), mask.astype(np.uint8), check_contrast=False)
        plt.savefig(os.path.join(scene_dir, 'process_dir', 'best_frame_visualizations', 'best_frame_selection.png'))
        imsave(os.path.join(vis_dir, f"object{track_id}_frame{selected_index}.png"), visualization.astype(np.uint8), check_contrast=False)

        # track_infos["tracks"][track_id]["selected_frame_for_xmem"] = selected_index
        tracks[track_id]["best_frame"] = int(selected_index)

    # hand_mask_areas = np.array(hand_mask_areas)

    write_output_to_config(scene_dir, "object_tracks", tracks)
    # json.dump(track_infos, open(os.path.join(scene_dir, 'process_dir', 'track_info.json'), 'w'))


def best_frame_selection_plot(ious, track_start, periods, selected_index, selected_index_iou, text_yoffset=0.03):

    # IoU line
    iou_line, = plot_with_offset(ious, track_start)
    iou_line.set_label("IoU")
    plt.ylim(0,1.1)

    # Bands
    periods = [[period[0] + track_start + 1, period[1] + track_start + 1] for period in periods]
    for i, period in enumerate(periods):
        band_color = 'dimgrey' if i % 2 == 0 else 'darkgrey'
        plt.axvspan(period[0], period[1], facecolor=band_color, alpha=0.5)

    # Arrow marker
    marker_coordinates = [selected_index,
                        selected_index_iou + 0.03]
    plt.plot(*marker_coordinates, marker=r'$\downarrow$', color='red')

    # Text at arrow marker
    text_coordinates = marker_coordinates
    text_coordinates[1] += text_yoffset
    plt.text(*text_coordinates, f"best frame at {selected_index}", ha='center', fontsize=11.0)

    # Threshold line
    threshold_line = plt.hlines(0.75, track_start, track_start + len(ious), color="red")
    threshold_line.set_label("Threshold")

    plt.vlines(selected_index, 0, selected_index_iou, linestyle="dotted", color="black")

    # Title, axis labels, legend
    plt.title("Cross-frame manipulated object mask IoU")
    plt.xlabel("Frame number")
    plt.ylabel("IoU")
    plt.legend()


if __name__ == '__main__':

    # Configure inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    args = parser.parse_args()

    config = {'input': 'C://Users//Vincent//code//motionsegment//data//multi2',
              'iou_threshold': 0.75,
              'n_consecutive_frames': 30,
              'track_start': 300,
              'track_end': 750}
    final_config = args.__dict__
    final_config.update(config)
    args = Namespace(**final_config)

    scene_dir = args.input
    nth_frame = 1
    slow_down = 2.5
    comparison_frame_id = 0

    out_dir = os.path.join(scene_dir, 'process_dir', 'object_candidate_masks')
    vis_dir = os.path.join(scene_dir, 'process_dir', 'object_candidate_masks_visualizations')
    debugging_vis_dir = os.path.join(scene_dir, 'process_dir', 'object_candidate_masks_debugging')

    scene_data = IPhoneData(scene_dir)

    # scene_data.load_computed_tracks()
    scene_data.load_poses()

    segmenter = Segmenter(scene_data, distance_treshold=0.02, n_comparisons=5)

    n_frames = len(scene_data.frames)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    track_infos = json.load(open(os.path.join(scene_dir, 'process_dir', 'track_info.json'), 'r'))

    object_candidate_masks = []

    visualizations = []

    for track_id, track in enumerate(track_infos["tracks"]):

        hand_mask_areas = []
        hand_distances = []
        fingertip_distances = []

        ious = {}



        for frame_id in tqdm(range(track["track_range"][0], track["track_range"][1], nth_frame)):

            if not 1 <= frame_id <= len(segmenter.scene_loader.frames) - 1:
                continue

            current_mask = segmenter.scene_loader.get_single_object_candidate_mask(frame_id+1)
            previous_mask = segmenter.scene_loader.get_single_object_candidate_mask(frame_id)
            next_mask = segmenter.scene_loader.get_single_object_candidate_mask(frame_id + 2)

            ious[frame_id] = get_iou(current_mask, previous_mask)

        iou_array = np.array(list(ious.values()))
        max_period = longest_true_period(iou_array > args.iou_threshold)
        selected_index = (max_period[0] + max_period[1]) // 2

        selected_index += track["track_range"][0] + 1

        track_infos["tracks"][track_id]["selected_frame_for_xmem"] = selected_index
    # hand_mask_areas = np.array(hand_mask_areas)

    json.dump(track_infos, open(os.path.join(scene_dir, 'process_dir', 'track_info.json'), 'w'))

    pass