from utils.data.dataloaders import IPhoneTracks, KinectPreprocessedData, IPhoneData, IPhoneTracks
# from visualization.visualize_tracks2 import TrackVisualizer
import os

import copy

import matplotlib
import numpy as np
import open3d as o3d
from tqdm import tqdm
# import keyboard
import time
import os
# import keyboard
from pynput import keyboard

from new_utils.utils import *
from new_utils.dataloaders import KinectTracks, KinectPreprocessedData, SceneData


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

    def visualize(self):

        # Create visualization window
        global rgb_pointcloud
        # rgb_pointcloud = self.get_legal_visualization_pcd_at_frame(0)
        rgb_pointcloud = self.load_pcd_at_frame('0')

        keep_running = True

        global current_frame_index
        current_frame_index = 0

        global vis
        vis = o3d.visualization.VisualizerWithKeyCallback()

        def update():
            global vis
            global current_frame_index
            global rgb_pointcloud
            print(f"Released: {current_frame_index}")
            # new_rgb_pointcloud = self.get_legal_visualization_pcd_at_frame(current_frame_index)
            new_rgb_pointcloud = self.load_pcd_at_frame(str(current_frame_index))
            rgb_pointcloud.points = new_rgb_pointcloud.points
            rgb_pointcloud.colors = new_rgb_pointcloud.colors

            vis.update_geometry(rgb_pointcloud)

            time.sleep(0.05)

        def forward_one(visualizer):
            global current_frame_index
            current_frame_index += 1
            update()

        def forward_ten(visualizer):
            global current_frame_index
            current_frame_index += 10
            update()

        def backward_one(visualizer):
            global current_frame_index
            current_frame_index -= 1
            update()

        def backward_ten(visualizer):
            global current_frame_index
            current_frame_index -= 10
            update()

        vis.create_window(height=480, width=640)
        vis.register_key_callback(78, forward_one)
        vis.register_key_callback(77, forward_ten)
        vis.register_key_callback(66, backward_one)
        vis.register_key_callback(86, backward_ten)
        vis.add_geometry(rgb_pointcloud)
        vis.run()

        while keep_running:
            keep_running = vis.poll_events()
            vis.update_renderer()

        vis.destroy_window()

    def get_legal_visualization_pcd_at_frame(self, frame_index):

        frame = self.frames[frame_index]
        t = frame_index

        # Make RGB point cloud
        color = self.scene_loader.get_single_color_frame(frame)
        depth = self.scene_loader.get_single_depth_frame(frame)
        rgb_pointcloud = get_point_cloud2(depth, color, self.intrinsics)[0]

        return rgb_pointcloud

    def get_point_cloud2(depth, color, intrinsics):

        depthimg = o3d.geometry.Image((depth / 1).astype(np.uint16))
        colorimg = o3d.geometry.Image(color.astype(np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(colorimg, depthimg, convert_rgb_to_intensity=False)
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=depth.shape[1],
                                                                     height=depth.shape[0],
                                                                     fx=intrinsics[0, 0],
                                                                     fy=intrinsics[1, 1],
                                                                     cx=intrinsics[0, 2],
                                                                     cy=intrinsics[1, 2])
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
        # pcd2 = o3d.geometry.PointCloud.create_from_depth_image(depthimg, pinhole_camera_intrinsic, depth_scale=1.0)
        defined_mask = np.asarray(rgbd.depth) != 0

        # o3d.visualization.draw_geometries([pcd])
        return pcd, defined_mask

    def load_pointclouds(self):

        self.scene_loader.pointclouds = []
        self.scene_loader.points_defined_in_image_masks = []

        for frame_id in tqdm(self.frames):

            pointcloud, points_defined_in_image_mask = self.get_pcd_at_frame(frame_id)
            current_pose = self.scene_loader.poses[self.scene_loader.frames.index(frame_id)]
            pointcloud = pointcloud.transform(current_pose)
            try:
                self.scene_loader.pointclouds.append(pointcloud)
                self.scene_loader.points_defined_in_image_masks.append(points_defined_in_image_mask)
            except:
                pass

        return

    def load_pcd_at_frame(self, frame_id):

        return self.scene_loader.pointclouds[self.scene_loader.frames.index(frame_id)]

        # # Make RGB point cloud
        # color = self.scene_loader.get_single_color_frame(frame_id)
        # depth = self.scene_loader.get_single_depth_frame(frame_id)
        # rgb_pointcloud, defined_mask = get_point_cloud2(depth, color, self.intrinsics)
        #
        # return rgb_pointcloud, defined_mask

    def get_pcd_at_frame(self, frame_id):

        # return self.scene_loader.pointclouds[self.scene_loader.frames.index(frame_id)]

        # Make RGB point cloud
        color = self.scene_loader.get_single_color_frame(frame_id)
        depth = self.scene_loader.get_single_depth_frame(frame_id)
        rgb_pointcloud, defined_mask = get_point_cloud2(depth, color, self.intrinsics)

        return rgb_pointcloud, defined_mask


if __name__ == "__main__":

    scene_dir = "/local/home/vincentv/code/motion_segment2/data/data1221_1525"
    scene_data = IPhoneData(scene_dir)
    scene_data.load_poses()
    # track_data = IPhoneTracks(track_dir)

    # scene_data.load_computed_tracks()
    # scene_data.load_poses()

    visualizer = Visualizer(scene_data)
    visualizer.load_pointclouds()
    visualizer.visualize()