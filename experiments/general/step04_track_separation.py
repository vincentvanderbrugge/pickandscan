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
from utils.data.iphone_recordings.datasets_common import *
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.rgbdimages import RGBDImages
from argparse import Namespace
import open3d as o3d
import scipy
import yaml


def increase_keys_by_fixed_number(dictionary, increase_by):
    updated_dict = {}
    for key, value in dictionary.items():
        updated_key = key + increase_by
        updated_dict[updated_key] = value
    return updated_dict


def write_output_to_config(scene_dir, key, value):
    config_dict = yaml.safe_load(open(os.path.join(scene_dir, "config.yaml"), 'r'))
    try:
        config_dict["output"][key] = value
    except KeyError:
        config_dict["output"] = {}
        config_dict["output"][key] = value
    yaml.dump(config_dict, open(os.path.join(scene_dir, "config.yaml"), 'w'), sort_keys=False)
    return


def get_zero_crossings(array):
    signs = np.sign(array)
    signs = [1 if el == 1 else el for el in signs]
    signs = [-1 if el == -1 else el for el in signs]
    signs = [0 if el == 0 or np.isnan(el) else el for el in signs]

    state_transitions = {"u": {1: ("pos", 0),
                               -1: ("neg", 0),
                               0: ("u", 0)},
                         "pos": {1: ("pos", 0),
                                 -1: ("neg", -1),
                                 0: ("upos", 0)},
                         "neg": {1: ("pos", 1),
                                 -1: ("neg", 0),
                                 0: ("uneg", 0)},
                         "upos": {1: ("pos", 0),
                                  -1: ("neg", -1),
                                  0: ("upos", 0)},
                         "uneg": {1: ("pos", 1),
                                  -1: ("neg", 0),
                                  0: ("uneg", 0)}
                         }

    sign_changes = [0]


    if np.isnan(array[0]):
        state = "u"
    elif array[0] > 0:
        state = "pos"
    else:
        state = "neg"

    states = [state]

    for i, sign in enumerate(signs[1:]):
        next_state, sign_change = state_transitions[state][sign]
        sign_changes.append(sign_change)
        states.append(next_state)
        state = next_state

    zero_crossings = np.nonzero(sign_changes)[0]
    zero_crossings = {idx: sign_changes[idx] for idx in zero_crossings}

    return zero_crossings


def get_visualization(segmenter, frame_id, comparison_frame_id):
    object_mask = segmenter.get_manipulated_object_mask_from_single_comparison(frame_number=frame_id,
                                                                               comparison_frame_index=comparison_frame_id)

    img = scene_data.get_single_color_frame(str(frame_id))
    visualization = create_masked_frame(img, object_mask)
    return visualization


def plot_with_offset(distances_fingertip_static, arm_entry_index, ax=None):
    ax = ax or plt.gca()
    return ax.plot([i for i in range(arm_entry_index, arm_entry_index + len(distances_fingertip_static))],
             distances_fingertip_static)


def hand_distances_plot(distances_fingertip_static,
                        distances_fingertip_object,
                        arm_entry_index):
    line_static, = plot_with_offset(distances_fingertip_static, arm_entry_index)
    line_object, = plot_with_offset(distances_fingertip_object, arm_entry_index)
    line_static.set_label('hand-initial distance')
    line_object.set_label('hand-object distance')
    plt.title("Distance trajectories for interaction detection")
    plt.xlabel("Frame number")
    plt.ylabel("Distance (m)")
    plt.legend()
    plt.show()


def interaction_detection_plot(distances_fingertip_static,
                               distances_fingertip_object,
                               dist_delta,
                               arm_entry_index,
                               segments,
                               size=(8,8),
                               hspace=0.1):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=size, gridspec_kw={'height_ratios': [1, 1]})
    plt.subplots_adjust(hspace=hspace)

    # Subplot 1: Distance trajectories
    plot_with_offset(distances_fingertip_static, arm_entry_index, ax1)
    plot_with_offset(distances_fingertip_object, arm_entry_index, ax1)
    y_range = (plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0])
    y_text_label_positions = [0.8, 0.65, 0.5]

    for i, segment in enumerate(segments):
        band_color = 'dimgrey' if i % 2 == 0 else 'darkgrey'
        y_position_multiplier = y_text_label_positions[i % 3]


        # Bands
        ax1.axvspan(segment['period'][0], segment['period'][1], facecolor=band_color, alpha=0.5)

        # Text labels
        text_coordinates = [(segment['period'][0] + segment['period'][1]) / 2, plt.gca().get_ylim()[0] + y_range * y_position_multiplier]
        ax1.text(*text_coordinates, f"interaction {i}", ha='center', fontsize=11.0)

        # Arrow markers
        marker_coordinates = text_coordinates
        marker_coordinates[1] -= y_range * 0.03
        ax1.plot(*marker_coordinates, marker=r'$\downarrow$', color='black')

    ax1.legend(["$d(hand, static)$"])
    ax1.set_xlabel("frame number")
    ax1.set_ylabel("distance (m)")
    # plt.title("Distances for track separation")

    # Subplot 2: Zero-crossings
    plot_with_offset(dist_delta, arm_entry_index, ax2)
    y_range = (plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0])
    y_text_label_positions = [0.5, 0.35, 0.2]

    for i, segment in enumerate(segments):
        band_color = 'dimgrey' if i % 2 == 0 else 'darkgrey'
        y_position_multiplier = y_text_label_positions[i % 3]


        # Bands
        ax2.axvspan(segment['period'][0], segment['period'][1], facecolor=band_color, alpha=0.5)

        # Text labels
        text_coordinates = [(segment['period'][0] + segment['period'][1]) / 2, plt.gca().get_ylim()[0] + y_range * y_position_multiplier]
        ax2.text(*text_coordinates, f"interaction {i}", ha='center', fontsize=11.0)

        # Arrow markers
        marker_coordinates = text_coordinates
        marker_coordinates[1] -= y_range * 0.03
        ax2.plot(*marker_coordinates, marker=r'$\downarrow$', color='black')

    plt.axhline(y=0, color='red')
    ax2.legend([r"$\Delta = d(hand, static) - d(hand, object)$"], loc="lower right")
    ax2.set_xlabel("frame number")
    ax2.set_ylabel("distance delta (m)")
        # ax2.title("Distances for track separation")

    # Create an invisible subplot for the title
    ax_title = fig.add_subplot(111, frameon=False)
    ax_title.set_xticks([])
    ax_title.set_yticks([])
    ax_title.set_title("Distances for track separation", fontsize=16)


def perform_track_separation(scene_dir):
    comparison_frame_id = 0

    nth_frame = 1

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

    object_candidate_masks = []

    visualizations = []

    hand_distances = []
    distances_fingertip_static = []

    # Load static pointcloud
    initial_pcd_path = os.path.join(scene_dir, "process_dir", "initial_points.npy")
    static_pointcloud_vector = o3d.utility.Vector3dVector(np.load(initial_pcd_path))
    static_pointcloud = o3d.geometry.PointCloud(static_pointcloud_vector)
    static_pointcloud = static_pointcloud.voxel_down_sample(0.01)

    arm_entry_index = yaml.safe_load(open(os.path.join(scene_dir, 'config.yaml')))["output"]["arm_entry_index"]

    # arm_entry_index = np.load(os.path.join(scene_dir, "process_dir", "arm_entry_index.npy"))
    arm_entry_index = int(arm_entry_index)

    # if not os.path.exists(os.path.join(scene_dir, "process_dir", "track_separation", "distances_fingertip_static.npy")):
    if True:

        all_distances_fingertip_static = []
        all_distances_fingertip_object = []
        for frame_id in tqdm(range(arm_entry_index, len(segmenter.scene_loader.frames), nth_frame)):
            # Get mask
            distances_fingertip_static, distances_fingertip_object = segmenter.get_fingertip_distance(frame_id,
                                                                                                      static_pointcloud)

            # point cloud in frame

            # moving candidate point cloud

            # finger point cloud

            all_distances_fingertip_static.append(distances_fingertip_static)
            all_distances_fingertip_object.append(distances_fingertip_object)
            pass

        distances_fingertip_static = [np.median(dists) for dists in all_distances_fingertip_static]
        distances_fingertip_object = [np.median(dists) for dists in all_distances_fingertip_object]

        os.makedirs(os.path.join(scene_dir, "process_dir", "track_separation"), exist_ok=True)

        distances_fingertip_static = np.array(distances_fingertip_static)
        distances_fingertip_object = np.array(distances_fingertip_object)

        np.save(os.path.join(scene_dir, "process_dir", "track_separation", "distances_fingertip_static.npy"),
                distances_fingertip_static)
        np.save(os.path.join(scene_dir, "process_dir", "track_separation", "distances_fingertip_object.npy"),
                distances_fingertip_object)

    else:
        distances_fingertip_static = np.load(
            os.path.join(scene_dir, "process_dir", "track_separation", "distances_fingertip_static.npy"))
        distances_fingertip_object = np.load(
            os.path.join(scene_dir, "process_dir", "track_separation", "distances_fingertip_object.npy"))

    distances_fingertip_static = scipy.signal.medfilt(distances_fingertip_static, 11)
    distances_fingertip_object = scipy.signal.medfilt(distances_fingertip_object, 11)

    lower_threshold = 0.05
    upper_threshold = 0.15

    # Find track separation
    segments = []

    ended = False
    started = False
    confirmed = False
    current_start = None

    for i, dist in enumerate(distances_fingertip_static):

        if not started and dist < lower_threshold:
            current_start = i + arm_entry_index
            started = True

        if started and dist > upper_threshold:
            confirmed = True

        if confirmed and dist < lower_threshold:
            end = i + arm_entry_index
            segments.append((current_start, end))
            started = False

    # Write track separation

    # pass

    # Figure out track separation
    dist_delta = distances_fingertip_static - distances_fingertip_object
    zero_crossings = get_zero_crossings(dist_delta)
    zero_crossings = increase_keys_by_fixed_number(zero_crossings, arm_entry_index)

    segments = []
    start_idx = None
    for idx, sign_change in zero_crossings.items():
        if start_idx is None and sign_change == 1:
            start_idx = idx

        if start_idx is not None and sign_change == -1:
            segments.append({"period": [int(start_idx), int(idx)]})
            start_idx = None

    segments = [s for s in segments if segments["period"][1] - segments["period"][0] > 50]

    os.makedirs(os.path.join(scene_dir, 'process_dir', 'best_frame_visualizations'), exist_ok=True)
    interaction_detection_plot(distances_fingertip_static, distances_fingertip_object, dist_delta, arm_entry_index, segments)
    plt.savefig(os.path.join(scene_dir, 'process_dir', 'best_frame_visualizations', 'track_separation.png'))
    # Write track separation to config file
    write_output_to_config(scene_dir, 'object_tracks', segments)
    pass


if __name__ == '__main__':

    # Configure inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    args = parser.parse_args()
    config = {'nth_frame': 1}
    config.update(args.__dict__)
    args = Namespace(**config)

    scene_dir = args.input
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

    object_candidate_masks = []

    visualizations = []

    hand_distances = []
    distances_fingertip_static = []

    # Load static pointcloud
    initial_pcd_path = os.path.join(scene_dir, "process_dir", "initial_points.npy")
    static_pointcloud_vector = o3d.cpu.pybind.utility.Vector3dVector(np.load(initial_pcd_path))
    static_pointcloud = o3d.cpu.pybind.geometry.PointCloud(static_pointcloud_vector)

    arm_entry_index = np.load(os.path.join(scene_dir, "process_dir", "arm_entry_index.npy"))
    arm_entry_index = int(arm_entry_index)

    all_distances_fingertip_static = []

    for frame_id in tqdm(range(arm_entry_index, len(segmenter.scene_loader.frames), args.nth_frame)):
        # Get mask
        distances_fingertip_static = segmenter.get_fingertip_distance(frame_id, static_pointcloud)

        all_distances_fingertip_static.append(distances_fingertip_static)
        pass

    pass
