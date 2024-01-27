import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import glob
import open3d as o3d
import yaml
from copy import deepcopy
import argparse
from scipy.spatial.transform import *

root_color = (66, 245, 155)
axis_color = {'x': (255, 0, 0),
              'y': (0, 255, 0),
              'z': (0, 0, 255)}

from new_utils.utils import *
from new_utils.dataloaders import KinectTracks, KinectPreprocessedData, SceneData, IPhoneData
from ground_truth_placement import Visualizer


TIMESTAMP_TO_INDEX = lambda timestamp: int(timestamp / 41666)


def load_config(args):
    d = yaml.safe_load(open(args.config, 'r'))
    d['config'] = args.config
    args = vars(args)
    args.update(d)
    args = Namespace(**args)
    return args


class Visualizer:

    def __init__(self, data: SceneData, n_frames_considered=None):
        # self.scan_dir = scan_dir
        # self.track_dir = track_dir
        # self.legal_periods = legal_periods

        self.scene_loader = data
        # self.tracks = tracks

        self.frames = self.scene_loader.get_frames()
        if n_frames_considered is None:
            self.n_frames_considered = len(self.frames)
        self.frames = self.frames[:n_frames_considered]

        self.intrinsics = data.get_intrinsics()

    def get_pcd_at_frame(self, frame_id):
        # return self.scene_loader.pointclouds[self.scene_loader.frames.index(frame_id)]

        # Make RGB point cloud
        color = self.scene_loader.get_single_color_frame(frame_id)
        depth = self.scene_loader.get_single_depth_frame(frame_id)
        rgb_pointcloud, defined_mask = get_point_cloud2(depth, color, self.intrinsics)

        return rgb_pointcloud, defined_mask

    def reconstruction_in_pointcloud(self, scene_dir, bundelsdf_dir, frame):
        # Load object
        object_path = os.path.join(bundlesdf_dir, frame, 'nerf', 'mesh_real_world.obj')

        # Load pointcloud
        rgb_pointcloud = self.load_pointcloud_from_memory(int(frame))

        # Display object at pose
        object_mesh = o3d.io.read_triangle_mesh(object_path)
        object_mesh.compute_vertex_normals()

        # drawer = o3d.visualization.draw(mesh)
        object_pose = get_pose(frame, bundlesdf_dir)
        object_mesh.transform(object_pose)
        o3d.visualization.draw_geometries([rgb_pointcloud, object_mesh])
        pass

    def ground_truth_in_pointcloud(self, scene_dir, bundelsdf_dir, frame, gt_object_path, alignment_path):
        # Load pointcloud
        rgb_pointcloud = self.load_pointcloud_from_memory(int(frame))

        # Load reconstruction
        object_path = os.path.join(bundlesdf_dir, frame, 'nerf', 'mesh_real_world.obj')
        reconstruction_mesh = o3d.io.read_triangle_mesh(object_path)
        reconstruction_mesh.compute_vertex_normals()

        # Load ground truth
        gt_mesh = o3d.io.read_triangle_mesh(gt_object_path)
        gt_mesh.compute_vertex_normals()

        # drawer = o3d.visualization.draw(mesh)
        gt_alignment_info = yaml.safe_load(open(alignment_path, 'r'))
        # gt_mesh.transform(object_pose)

        # Place object at estimated pose
        object_pose = get_pose(frame, bundlesdf_dir)
        reconstruction_mesh.transform(object_pose)

        # Scale
        scale = tuple(gt_alignment_info["scale"].values())[0]
        gt_mesh.scale(scale, center=(0, 0, 0))

        # Rotation
        xyz_angles = list(gt_alignment_info["rotation"].values())
        xyz_angles = [angle * 2 * np.pi / 360 for angle in xyz_angles]
        gt_rot_matrix = gt_mesh.get_rotation_matrix_from_xyz(xyz_angles)
        gt_mesh.rotate(gt_rot_matrix, center=(0, 0, 0))

        # Transformation
        xyz_translation = tuple(gt_alignment_info["position"].values())
        gt_mesh.translate(xyz_translation)

        coordinate_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

        o3d.visualization.draw_geometries([rgb_pointcloud, gt_mesh, reconstruction_mesh, coordinate_mesh])
        pass

    def load_pointcloud_from_memory(self, frame_id):
        pointcloud, points_defined_in_image_mask = self.get_pcd_at_frame(frame_id)
        # current_pose = self.scene_loader.poses[frame_id]
        # pointcloud = pointcloud.transform(current_pose)
        return pointcloud


def get_pose(frame, path):
    return read_matrix_from_txt_file(os.path.join(path, 'ob_in_cam', '%05d.txt' % frame))


def transform_mesh(mesh, translation, rotation, scale):
    # Scale
    mesh.scale(scale, center=(0, 0, 0))

    # Rotation
    xyz_angles = deepcopy(rotation)
    xyz_angles = [angle * 2 * np.pi / 360 for angle in xyz_angles]
    rot_matrix = mesh.get_rotation_matrix_from_xyz(xyz_angles)
    mesh.rotate(rot_matrix, center=(0, 0, 0))

    # Transformation
    mesh.translate(translation)


def transform_pcd(pcd, translation, rotation, scale=1.0):
    # Scale
    pcd.scale(scale, center=(0, 0, 0))

    # Rotation
    xyz_angles = deepcopy(rotation)
    xyz_angles = [angle * 2 * np.pi / 360 for angle in xyz_angles]
    rot_matrix = pcd.get_rotation_matrix_from_xyz(xyz_angles)
    pcd.rotate(rot_matrix, center=(0, 0, 0))

    # Transformation
    pcd.translate(translation)


def transform_mesh_from_quat(mesh, translation, quaternion, scale=1.0):
    # Scale
    mesh.scale(scale, center=(0, 0, 0))

    # Rotation
    # rot_matrix = Rotation.from_quat(quaternion).as_matrix()
    rot_matrix = o3d.geometry.TriangleMesh.get_rotation_matrix_from_quaternion(quaternion)
    mesh.rotate(rot_matrix, center=(0, 0, 0))

    # Transformation
    mesh.translate(translation)


def transform_pcd_from_quat(pcd, translation, quaternion):

    # Rotation

    # rot_matrix = Rotation.from_quat(quaternion).as_matrix()
    rot_matrix = o3d.geometry.TriangleMesh.get_rotation_matrix_from_quaternion(quaternion)
    pcd.rotate(rot_matrix, center=(0, 0, 0))

    # Transformation
    pcd.translate(translation)


def load_cofusion_pose_from_txt(path_to_txt, frame_id):
    numbers_list = []
    with open(path_to_txt, 'r') as file:
        for line in file:
            # Remove leading and trailing whitespaces and split the line into a list of strings
            numbers_as_strings = line.strip().split()

            # Convert the list of strings to a list of floats
            numbers = [float(num) for num in numbers_as_strings]

            numbers[0] = TIMESTAMP_TO_INDEX(numbers[0])

            # Add the list of numbers to the overall list
            numbers_list.append(numbers)
    pose = [line for line in numbers_list if line[0] == frame_id][0][1:]
    return pose


def compute_chamfer_distance(pc1, pc2):
    """
    Compute Chamfer distance between two point clouds.

    Args:
    pc1 (open3d.geometry.PointCloud): First point cloud.
    pc2 (open3d.geometry.PointCloud): Second point cloud.

    Returns:
    float: Chamfer distance between the two point clouds.
    """
    distances1 = pc1.compute_point_cloud_distance(pc2)
    distances2 = pc2.compute_point_cloud_distance(pc1)

    chamfer_dist = np.mean([np.mean(distances1), np.mean(distances2)])
    return chamfer_dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    args = parser.parse_args()

    config = load_config(args)

    scene_data = IPhoneData(args.scene_dir)
    scene_data.load_poses()

    visualizer = Visualizer(scene_data)

    rgb_pointcloud = visualizer.load_pointcloud_from_memory(int(args.evaluation_frame))

    if not os.path.exists(os.path.join(config.scene_dir, 'eval_dir')):
        os.makedirs(os.path.join(config.scene_dir, 'eval_dir'))

    pointcloud_path = os.path.join(config.scene_dir, 'eval_dir', f'frame{config.evaluation_frame}.ply')
    if not os.path.exists(pointcloud_path):
        o3d.io.write_point_cloud(pointcloud_path, rgb_pointcloud)


    # Load reconstruction
    reconstruction_pcd = o3d.io.read_point_cloud(os.path.join(config.cofusion_output_dir, f'outcloud-{config.cofusion_object_id}.ply'))

    # Load ground truth
    gt_mesh = o3d.io.read_triangle_mesh(args.ground_truth_model)
    gt_mesh.compute_vertex_normals()
    # gt_alignment_info = yaml.safe_load(open(alignment_path, 'r'))

    # Place reconstruction
    # object_pose = load_cofusion_pose_from_txt(os.path.join(config.cofusion_output_dir, f'outposes-{config.cofusion_object_id}.txt'), config.evaluation_frame)
    # reconstruction_quat = object_pose[3:]
    # reconstruction_translation = object_pose[:3]
    # transform_pcd_from_quat(reconstruction_pcd, reconstruction_translation, reconstruction_quat)
    # rotmat = Rotation.from_quat(quat).as_matrix()
    # transformation_matrix = np.eye(4)
    # transformation_matrix[:3, :3] = rotmat
    # transformation_matrix[:3, 3] = object_pose[:3]
    # reconstruction_pcd.transform(transformation_matrix)
    xyz_angles_reconstruction = list(config.reconstruction_placement["rotation"].values())
    xyz_translation_reconstruction = tuple(config.reconstruction_placement["position"].values())
    transform_pcd(reconstruction_pcd, xyz_translation_reconstruction, xyz_angles_reconstruction)

    reconstruction_axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # transform_mesh_from_quat(reconstruction_axes, reconstruction_translation, reconstruction_quat, 0.3)
    transform_mesh(reconstruction_axes, xyz_translation_reconstruction, xyz_angles_reconstruction, scale=0.3)

    # Place ground truth
    scale = tuple(config.ground_truth_placement["scale"].values())[0]
    xyz_angles = list(config.ground_truth_placement["rotation"].values())
    xyz_translation = tuple(config.ground_truth_placement["position"].values())
    transform_mesh(gt_mesh, xyz_translation, xyz_angles, scale)

    # Ground truth coordinate frame
    gt_axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
    transform_mesh(gt_axes, xyz_translation, xyz_angles, 0.3)

    # # Transform both to origin
    # # gt_mesh.translate(tuple(-object_pose[:3, 3]))
    # object_pose[:3, 3] = 0
    # reconstruction_mesh.transform(object_pose)
    # xyz_translation = list(xyz_translation)
    # xyz_translation[0] += -0.4
    # xyz_translation[1] += -0.1
    # xyz_translation[2] += -0.05
    # xyz_angles =[0,0,0]
    # transform_mesh(reconstruction_mesh, xyz_translation, xyz_angles, 1)

    # Visualize
    coordinate_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([rgb_pointcloud, gt_mesh, reconstruction_pcd, coordinate_mesh, gt_axes])
    # o3d.visualization.draw_geometries([reconstruction_mesh])

    trans_init = np.eye(4)
    # source_pcd = reconstruction_mesh.sample_points_uniformly(args.icp["num_sample_points"])
    source_pcd = reconstruction_pcd
    target_pcd = gt_mesh.sample_points_uniformly(args.icp["num_sample_points"])
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, args.icp["termination_threshold"], trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    source_pcd.transform(reg_p2p.transformation)
    source_pcd.paint_uniform_color([1, 0, 0])
    target_pcd.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([source_pcd, target_pcd])

    err = compute_chamfer_distance(source_pcd, target_pcd)
    print(f"Error: {err}")
    pass
