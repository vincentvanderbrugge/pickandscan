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
import open3d as o3d
from argparse import Namespace

# from all_in_one import write_output_to_config


def write_output_to_config(scene_dir, key, value):
    config_dict = yaml.safe_load(open(os.path.join(scene_dir, "config.yaml"), 'r'))
    try:
        config_dict["output"][key] = value
    except KeyError:
        config_dict["output"] = {}
        config_dict["output"][key] = value
    yaml.safe_dump(config_dict, open(os.path.join(scene_dir, "config.yaml"), 'w'), sort_keys=False)
    return


def get_visualization(segmenter, frame_id, comparison_frame_id):
    object_mask = segmenter.get_manipulated_object_mask_from_single_comparison(frame_number=frame_id,
                                                                               comparison_frame_index=comparison_frame_id)

    img = scene_data.get_single_color_frame(str(frame_id))
    visualization = create_masked_frame(img, object_mask)
    return visualization


def generate_initial_pointcloud(input_path):
    scene_dir = input_path
    out_dir = os.path.join(scene_dir, 'process_dir', 'object_candidate_masks')
    vis_dir = os.path.join(scene_dir, 'process_dir', 'object_candidate_masks_visualizations')
    debugging_vis_dir = os.path.join(scene_dir, 'process_dir', 'object_candidate_masks_debugging')

    scene_data = IPhoneData(scene_dir)
    scene_data.load_poses()

    segmenter = Segmenter(scene_data, distance_treshold=0.02, n_comparisons=5)

    n_frames = len(scene_data.frames)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    object_candidate_masks = []

    visualizations = []

    hand_mask_areas = []
    hand_distances = []
    fingertip_distances = []

    for frame_id in tqdm(range(0, len(segmenter.scene_loader.frames))):
        # Get mask
        hand_mask_area = segmenter.get_hand_mask_area(frame_id)
        hand_mask_areas.append(hand_mask_area)

    hand_mask_areas = np.array(hand_mask_areas)

    arm_entry_index = np.where((hand_mask_areas > 1000))[0].min()

    write_output_to_config(scene_dir, "arm_entry_index", int(arm_entry_index))

    cfg = load_dataset_config(
        os.path.join(scene_dir, "dataconfig.yaml")
    )
    dataset = Record3DDataset(
        config_dict=cfg,
        basedir=scene_dir,
        sequence=None,
        start=0,
        end=-1,
        stride=1,
        # desired_height=680,
        # desired_width=1200,
        desired_height=192,
        desired_width=144,
    )

    colors, depths, poses = [], [], []
    intrinsics = None
    for idx in range(len(dataset)):
        _color, _depth, intrinsics, _pose = dataset[idx]
        colors.append(_color)
        depths.append(_depth)
        poses.append(_pose)
    colors = torch.stack(colors)
    depths = torch.stack(depths)
    poses = torch.stack(poses)
    colors = colors.unsqueeze(0)
    depths = depths.unsqueeze(0)
    intrinsics = intrinsics.unsqueeze(0).unsqueeze(0)
    poses = poses.unsqueeze(0)
    colors = colors.float()
    depths = depths.float()
    intrinsics = intrinsics.float()
    poses = poses.float()

    # Filter data to static part
    colors_static = colors[:, :arm_entry_index, ...]
    depths_static = depths[:, :arm_entry_index, ...]
    poses_static = poses[:, :arm_entry_index, ...]

    # create rgbdimages object
    rgbdimages_static = RGBDImages(
        colors_static,
        depths_static,
        intrinsics,
        poses_static,
        channels_first=False,
        has_embeddings=False,  # KM
    )

    # SLAM
    slam = PointFusion(odom="gt", dsratio=1, device="cuda:0", use_embeddings=False)
    pointclouds, recovered_poses = slam(rgbdimages_static)

    print(pointclouds.colors_padded.shape)
    pcd = pointclouds.open3d(0)
    recovered_poses = np.asarray(recovered_poses.cpu())[0]
    arr_list = recovered_poses.tolist()

    o3d.visualization.draw_geometries([pcd])

    np.save(os.path.join(scene_dir, "process_dir", "initial_points.npy"), np.array(pcd.points))
    np.save(os.path.join(scene_dir, "process_dir", "initial_points_colors.npy"), np.array(pcd.colors))



if __name__ == '__main__':

    # Configure inputs
    config = {'input': 'C://Users//Vincent//code//motionsegment//data//multi2'}

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    args = parser.parse_args()
    final_config = {}
    final_config = args.__dict__
    final_config.update(config)
    args = Namespace(**final_config)

    scene_dir = args.input
    out_dir = os.path.join(scene_dir, 'process_dir', 'object_candidate_masks')
    vis_dir = os.path.join(scene_dir, 'process_dir', 'object_candidate_masks_visualizations')
    debugging_vis_dir = os.path.join(scene_dir, 'process_dir', 'object_candidate_masks_debugging')

    scene_data = IPhoneData(scene_dir)
    scene_data.load_poses()

    segmenter = Segmenter(scene_data, distance_treshold=0.02, n_comparisons=5)

    n_frames = len(scene_data.frames)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    object_candidate_masks = []

    visualizations = []

    hand_mask_areas = []
    hand_distances = []
    fingertip_distances = []


    for frame_id in tqdm(range(0, len(segmenter.scene_loader.frames))):

        # Get mask
        hand_mask_area = segmenter.get_hand_mask_area(frame_id)
        hand_mask_areas.append(hand_mask_area)


    hand_mask_areas = np.array(hand_mask_areas)

    arm_entry_index = np.where((hand_mask_areas > 0))[0].min()

    cfg = load_dataset_config(
        os.path.join(scene_dir, "dataconfig.yaml")
    )
    dataset = Record3DDataset(
        config_dict=cfg,
        basedir=scene_dir,
        sequence=None,
        start=0,
        end=-1,
        stride=1,
        # desired_height=680,
        # desired_width=1200,
        desired_height=240,
        desired_width=320,
    )

    colors, depths, poses = [], [], []
    intrinsics = None
    for idx in range(len(dataset)):
        _color, _depth, intrinsics, _pose = dataset[idx]
        colors.append(_color)
        depths.append(_depth)
        poses.append(_pose)
    colors = torch.stack(colors)
    depths = torch.stack(depths)
    poses = torch.stack(poses)
    colors = colors.unsqueeze(0)
    depths = depths.unsqueeze(0)
    intrinsics = intrinsics.unsqueeze(0).unsqueeze(0)
    poses = poses.unsqueeze(0)
    colors = colors.float()
    depths = depths.float()
    intrinsics = intrinsics.float()
    poses = poses.float()

    # Filter data to static part
    colors_static = colors[:, :arm_entry_index, ...]
    depths_static = depths[:, :arm_entry_index, ...]
    poses_static = poses[:, :arm_entry_index, ...]



    # create rgbdimages object
    rgbdimages_static = RGBDImages(
        colors_static,
        depths_static,
        intrinsics,
        poses_static,
        channels_first=False,
        has_embeddings=False,  # KM
    )

    # SLAM
    slam = PointFusion(odom="gt", dsratio=1, device="cuda:0", use_embeddings=False)
    pointclouds, recovered_poses = slam(rgbdimages_static)

    print(pointclouds.colors_padded.shape)
    pcd = pointclouds.open3d(0)
    recovered_poses = np.asarray(recovered_poses.cpu())[0]
    arr_list = recovered_poses.tolist()

    o3d.visualization.draw_geometries([pcd])

    np.save(os.path.join(scene_dir, "process_dir", "initial_points.npy"), np.array(pcd.points))
    np.save(os.path.join(scene_dir, "process_dir", "initial_points_colors.npy"), np.array(pcd.colors))

    pass