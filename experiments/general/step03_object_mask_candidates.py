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
import open3d as o3d


def get_visualization(segmenter, frame_id, comparison_frame_id):
    object_mask = segmenter.get_manipulated_object_mask_from_single_comparison(frame_number=frame_id,
                                                                               comparison_frame_index=comparison_frame_id)

    img = scene_data.get_single_color_frame(str(frame_id))
    visualization = create_masked_frame(img, object_mask)
    return visualization


def get_object_candidate_masks(scene_dir):
    # Config
    config = {'start_frame': 0,
              'nth_frame': 1,
              'slow_down': 2.5,
              'manipulator': 'fingertips',
              'comparison_frame_id': 0}
    # parser = argparse.ArgumentParser(description="")
    # parser.add_argument('-i', '--input')
    # args = parser.parse_args()
    # final_config = args.__dict__
    # final_config.update(config)
    args = Namespace(**config)

    out_dir = os.path.join(scene_dir, 'process_dir', 'object_candidate_masks')
    vis_dir = os.path.join(scene_dir, 'process_dir', 'object_candidate_masks_visualizations')
    debugging_vis_dir = os.path.join(scene_dir, 'process_dir', 'object_candidate_masks_debugging')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    scene_data = IPhoneData(scene_dir)
    # scene_data.load_computed_tracks()
    scene_data.load_poses()

    segmenter = Segmenter(scene_data, distance_treshold=0.02, n_comparisons=5)
    n_frames = len(scene_data.frames)
    object_candidate_masks = []
    visualizations = []

    # Load static pointcloud
    initial_pcd_path = os.path.join(scene_dir, "process_dir", "initial_points.npy")
    static_pointcloud_vector = o3d.utility.Vector3dVector(np.load(initial_pcd_path))
    static_pointcloud = o3d.geometry.PointCloud(static_pointcloud_vector)
    static_pointcloud = static_pointcloud.voxel_down_sample(0.01)

    for frame_id in tqdm(range(args.start_frame, n_frames, args.nth_frame)):
        # frame_id=1135
        mask = segmenter.get_manipulated_object_mask_from_single_comparison(frame_id,
                                                                            args.comparison_frame_id,
                                                                            debugging=True,
                                                                            use_for_distance=args.manipulator,
                                                                            use_static_pointcloud=True,
                                                                            static_point_cloud=static_pointcloud,
                                                                            debugging_visualizations_dir=debugging_vis_dir)
        # Get mask

        # Save & visualize
        img = scene_data.get_single_color_frame(str(frame_id))
        visualization = create_masked_frame(img, mask)
        visualizations.append(visualization)
        imsave(os.path.join(out_dir, str(frame_id) + '.png'), mask.astype(np.uint8), check_contrast=False)
        imsave(os.path.join(vis_dir, str(frame_id) + '.png'), visualization.astype(np.uint8), check_contrast=False)

        object_candidate_masks.append(mask)
        visualizations.append(visualization)

    # Create visualization gif
    fig, ax = plt.subplots()
    im = ax.imshow(visualizations[0], animated=True)

    def update(i):
        frame_id = i
        print(f"Visualizing frame {frame_id}")
        im.set_array(visualizations[i])
        plt.title(f'Frame {frame_id}')
        return im,

    # Create the animation object
    animation_fig = animation.FuncAnimation(fig, update, frames=n_frames // args.nth_frame,
                                            interval=int(30 * args.nth_frame * args.slow_down), blit=True,
                                            repeat_delay=10, )
    animation_fig.save(os.path.join(scene_dir, 'process_dir', 'object_candidate_masks.gif'))


if __name__ == '__main__':

    # Config
    config = {'input': 'C://Users//Vincent//code//motionsegment//data//multi2',
              'start_frame': 457,
              'nth_frame': 5,
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
    out_dir = os.path.join(scene_dir, 'process_dir', 'object_candidate_masks')
    vis_dir = os.path.join(scene_dir, 'process_dir', 'object_candidate_masks_visualizations')
    debugging_vis_dir = os.path.join(scene_dir, 'process_dir', 'object_candidate_masks_debugging')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    scene_data = IPhoneData(scene_dir)
    # scene_data.load_computed_tracks()
    scene_data.load_poses()

    segmenter = Segmenter(scene_data, distance_treshold=0.02, n_comparisons=5)
    n_frames = len(scene_data.frames)
    object_candidate_masks = []
    visualizations = []

    # Load static pointcloud
    initial_pcd_path = os.path.join(scene_dir, "process_dir", "initial_points.npy")
    static_pointcloud_vector = o3d.cpu.pybind.utility.Vector3dVector(np.load(initial_pcd_path))
    static_pointcloud = o3d.cpu.pybind.geometry.PointCloud(static_pointcloud_vector)

    for frame_id in tqdm(range(args.start_frame, 500, args.nth_frame)):

        mask = segmenter.get_manipulated_object_mask_from_single_comparison(frame_id,
                                                                            args.comparison_frame_id,
                                                                            debugging=True,
                                                                            use_for_distance=args.manipulator,
                                                                            use_static_pointcloud=True,
                                                                            static_point_cloud=static_pointcloud,
                                                                            debugging_visualizations_dir=debugging_vis_dir)
        # Get mask

        # Save & visualize
        img = scene_data.get_single_color_frame(str(frame_id))
        visualization = create_masked_frame(img, mask)
        visualizations.append(visualization)
        imsave(os.path.join(out_dir, str(frame_id) + '.png'), mask.astype(np.uint8))
        imsave(os.path.join(vis_dir, str(frame_id) + '.png'), visualization.astype(np.uint8))

        object_candidate_masks.append(mask)
        visualizations.append(visualization)

    # Create visualization gif
    fig, ax = plt.subplots()
    im = ax.imshow(visualizations[0], animated=True)

    def update(i):
        frame_id = i
        print(f"Visualizing frame {frame_id}")
        im.set_array(visualizations[i])
        plt.title(f'Frame {frame_id}')
        return im,

    # Create the animation object
    animation_fig = animation.FuncAnimation(fig, update, frames=n_frames//args.nth_frame, interval=int(30*args.nth_frame*args.slow_down), blit=True,
                                            repeat_delay=10, )
    animation_fig.save(os.path.join(scene_dir, 'process_dir', 'object_candidate_masks.gif'))

