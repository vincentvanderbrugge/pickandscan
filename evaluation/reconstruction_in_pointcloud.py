import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import glob
import open3d as o3d

scene_dir = '/local/home/vincentv/code/motion_segment2/data/refactor_test_data1226_1432'
bundlesdf_dir = '/local/home/vincentv/code/motion_segment2/data/data1226_1432_object0'
out_path = '/local/home/vincentv/code/motion_segment2/data/misc/pose_video.avi'
frame = '00040'

root_color = (66, 245, 155)
axis_color = {'x': (255,0,0),
              'y': (0,255,0),
              'z': (0,0,255)}


from new_utils.utils import *
from new_utils.dataloaders import KinectTracks, KinectPreprocessedData, SceneData, IPhoneData


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

    def load_pointcloud_from_memory(self, frame_id):

        pointcloud, points_defined_in_image_mask = self.get_pcd_at_frame(frame_id)
        # current_pose = self.scene_loader.poses[frame_id]
        # pointcloud = pointcloud.transform(current_pose)
        return pointcloud

def get_pose(frame, path):
    return read_matrix_from_txt_file(os.path.join(path, 'ob_in_cam', f'{frame}.txt'))

if __name__ == "__main__":
    # poses = read_poses(bundlesdf_dir)
    # create_pose_axis_video(scene_dir, bundlesdf_dir, out_path)
    scene_data = IPhoneData(scene_dir)
    scene_data.load_poses()

    visualizer = Visualizer(scene_data)
    # visualizer.load_pointclouds()
    visualizer.reconstruction_in_pointcloud(scene_dir, bundlesdf_dir, frame)