import os
import numpy as np
import torch
from tqdm import tqdm
import json
import cv2
import yaml
from scipy.spatial.transform import Rotation
import random

from new_utils.utils import load_image, read_matrix_from_txt_file, colmap_pose_to_transformation_matrix


class SceneData:

    def __init__(self, scene_dir, downsample_ratio=1):

        self.scene_dir = scene_dir

        self.process_dir = os.path.join(self.scene_dir, 'process_dir')
        if not os.path.isdir(self.process_dir):
            os.makedirs(self.process_dir)

        self.downsample_ratio = downsample_ratio
        self.frames = self.get_frames()

        self.positions_3d = None
        self.visibilities_3d = None
        self.positions_2d = None
        self.visibilities_2d = None
        self.legalities = None
        self.legal_periods = None
        self.n_occlusion_cutoff = 2


        self.poses = None

        self.pointclouds = None
        self.points_defined_in_image_masks = None

    def load_computed_tracks(self):

        self.positions_3d = np.load(os.path.join(self.process_dir, 'tracks',  'positions_3d.npy'))
        self.visibilities_3d = np.load(os.path.join(self.process_dir, 'tracks', 'visibilities_3d.npy'))
        self.positions_2d = np.load(os.path.join(self.process_dir, 'tracks', 'positions_2d.npy'))
        self.visibilities_2d = np.load(os.path.join(self.process_dir, 'tracks', 'visibilities_2d.npy'))

        self.legalities = np.bool_(np.zeros_like(self.visibilities_3d))

        legal_periods = []

        for i in range(self.visibilities_2d.shape[1]):
            legal_periods.append(self.legal_period(self.visibilities_2d[:, i]))

        self.legal_periods = np.array(legal_periods)

        for i in range(self.legalities.shape[1]):
            self.legalities[self.legal_periods[i, 0]:self.legal_periods[i, 1] + 1, i] = True

    def legal_period(self, visibility):

        visible_indices = np.where(visibility)[0]

        if len(visible_indices) == 0:
            return [np.nan, np.nan]
        elif len(visible_indices) == 1:
            return [visible_indices[0], visible_indices[0]]

        start = visible_indices[0]
        previous_index = start

        for index in visible_indices[1:]:
            if index - previous_index - 1 > self.n_occlusion_cutoff:
                return [start, previous_index]
            previous_index = index

        return [start, visible_indices[-1]]

    def load_poses(self):

        raise NotImplementedError

    def get_frames(self):
        raise NotImplementedError

    def get_video(self):
        color_frames = []

        print("Loading color frames.")

        for frame in tqdm(self.frames):
            color_frames.append(self.get_single_color_frame(frame))

        video = np.array(color_frames)
        video = video.transpose([0, 3, 1, 2])
        video = video[None, ...]
        video = torch.tensor(video)
        video = video.to(torch.float32)

        return video

    def get_depth(self):

        depth_frames = []

        print("Loading depth frames.")

        for frame in tqdm(self.frames):
            depth_frames.append(self.get_single_depth_frame(frame))

        depth = np.array(depth_frames)

        return depth

    def get_intrinsics(self):
        raise NotImplementedError

    def get_single_color_frame(self, frame):
        raise NotImplementedError

    def get_single_depth_frame(self):
        raise NotImplementedError


class HODData(SceneData):

    def get_frames(self):

        frames = os.listdir(os.path.join(self.scene_dir, 'rgb'))

        frames = [frame[:-4] for frame in frames]
        frames.sort(key=lambda x: int(x))

        return frames

    def get_intrinsics(self):
        return read_matrix_from_txt_file(os.path.join(self.scene_dir, 'cam_K.txt'))

    def get_single_color_frame(self, frame):
        return self.load_image_from_bundle(frame, self.scene_dir)

    def get_single_depth_frame(self, frame):
        return self.load_depth_from_bundle(frame, self.scene_dir)

    @staticmethod
    def load_depth_from_bundle(frame, path_to_bundle):
        depth = load_image(os.path.join(path_to_bundle, 'depth', f'{frame}'))
        return depth

    @staticmethod
    def load_image_from_bundle(frame, path_to_bundle):
        image = load_image(os.path.join(path_to_bundle, 'rgb', f'{frame}'))
        return image


class KinectPreprocessedData(SceneData):

    def get_frames(self):
        frames = os.listdir(os.path.join(self.scene_dir, 'color'))

        frames = [frame[:-4] for frame in frames]
        frames.sort(key=lambda x: int(x))

        return frames

    def load_poses(self):

        self.poses = np.array(json.load(open(os.path.join(self.scene_dir, 'poses.json'))))

    def load_colmap_poses(self):

        lines = open(os.path.join(self.process_dir, 'colmap', 'images.txt')).readlines()
        lines = [line for line in lines if ".png" in line]
        pose_dict = {}
        for pose_line in lines:
            pose_arr = pose_line.split(' ')
            frame_id = int(pose_arr[0])-1
            quaternion = np.array([float(q) for q in pose_arr[1:-2]])
            pass
        self.poses = None

    def preprocess_colmap_poses(self):

        lines = open(os.path.join(self.process_dir, 'colmap', 'text_export', 'images.txt')).readlines()
        lines = [line for line in lines if ".png" in line]
        pose_dict = {}
        for pose_line in lines:
            pose_arr = pose_line.split(' ')
            frame_id = int(pose_arr[0])-1
            colmap_pose = np.array([float(q) for q in pose_arr[1:-2]])
            pose_dict[frame_id] = colmap_pose_to_transformation_matrix(colmap_pose)
            pass
        colmap_to_kinect_scale = self.get_colmap_to_kinect_scale()
        self.poses = None

    def get_colmap_to_kinect_scale(self, num_keypoints=100):

        keypoints = self.load_colmap_keypoints()
        sampled_keypoints = random.sample(keypoints, num_keypoints)

        kinect_depths = [self.get_kinect_depth_for_keypoint(keypoint) for keypoint in sampled_keypoints]
        colmap_depths = [self.get_colmap_depth_for_keypoint(keypoint) for keypoint in sampled_keypoints]

        kinect_depths = np.array(kinect_depths)
        colmap_depths = np.array(colmap_depths)

        return self.optimal_scale_factor(kinect_depths, colmap_depths)

    @staticmethod
    def optimal_scale_factor(depths1, depths2):

        raise NotImplementedError

    def get_kinect_depth_for_keypoint(self, keypoint):

        raise NotImplementedError

    def get_colamp_depth_for_keypoint(self, keypoint):

        raise NotImplementedError


    def load_colmap_keypoints(self):

        raise NotImplementedError

    def get_intrinsics(self):
        intrinsic_file = open(os.path.join(self.scene_dir, 'intrinsics.json'))
        intrinsic_array = json.load(intrinsic_file)['intrinsic_matrix']
        intrinsic_matrix = np.array([intrinsic_array[:3], intrinsic_array[3:6], intrinsic_array[6:9]]).transpose()
        intrinsic_matrix[:2, ...] = intrinsic_matrix[:2, ...] / self.downsample_ratio
        return intrinsic_matrix

    def get_single_color_frame(self, frame):
        color = self.load_image_from_bundle(frame, self.scene_dir)
        resolution = [color.shape[1] // self.downsample_ratio, color.shape[0] // self.downsample_ratio]
        color = cv2.resize(color, resolution)
        return color

    def get_single_depth_frame(self, frame):
        depth = self.load_depth_from_bundle(frame, self.scene_dir)
        resolution = [depth.shape[1]//self.downsample_ratio, depth.shape[0]//self.downsample_ratio]
        depth = cv2.resize(depth.astype(np.float64), resolution, interpolation=cv2.INTER_NEAREST).astype(np.int32)
        return depth

    @staticmethod
    def load_depth_from_bundle(frame, path_to_bundle):
        return load_image(os.path.join(path_to_bundle, 'depth', f'{frame}.png'))

    @staticmethod
    def load_image_from_bundle(frame, path_to_bundle):
        return load_image(os.path.join(path_to_bundle, 'color', f'{frame}.png'))


class IPhoneData(SceneData):

    def get_frames(self):
        frames = os.listdir(os.path.join(self.scene_dir, 'rgb'))

        frames = [frame[:-4] for frame in frames]
        # frames.sort()
        frames.sort(key=lambda x: int(x))

        return frames

    def get_intrinsics(self):
        # intrinsic_file = open(os.path.join(self.scene_dir, 'intrinsics.json'))
        # intrinsic_array = json.load(intrinsic_file)['intrinsic_matrix']

        intrinsics_dict = yaml.safe_load(open(os.path.join(self.scene_dir, 'dataconfig.yaml')))
        intrinsic_matrix = np.zeros([3, 3])
        intrinsic_matrix[0, 0] = intrinsics_dict['camera_params']['fx']
        intrinsic_matrix[1, 1] = intrinsics_dict['camera_params']['fy']
        intrinsic_matrix[0, 2] = intrinsics_dict['camera_params']['cx']
        intrinsic_matrix[1, 2] = intrinsics_dict['camera_params']['cy']
        # intrinsic_matrix = np.array([intrinsic_array[:3], intrinsic_array[3:6], intrinsic_array[6:9]]).transpose()
        intrinsic_matrix[:2, ...] = intrinsic_matrix[:2, ...] / self.downsample_ratio
        intrinsic_matrix[2, 2] = 1
        return intrinsic_matrix

    def load_poses(self):

        self.poses = np.array(json.load(open(os.path.join(self.process_dir, 'poses.json'))))

    def get_single_color_frame(self, frame):
        color = load_image(os.path.join(self.scene_dir, 'rgb', f'{frame}.png'))
        resolution = [color.shape[1] // self.downsample_ratio, color.shape[0] // self.downsample_ratio]
        color = cv2.resize(color, resolution)
        return color

    def get_single_depth_frame(self, frame):
        depth = load_image(os.path.join(self.scene_dir, 'depth', f'{frame}.png'))
        resolution = [depth.shape[1]//self.downsample_ratio, depth.shape[0]//self.downsample_ratio]
        depth = cv2.resize(depth.astype(np.float64), resolution, interpolation=cv2.INTER_NEAREST).astype(np.int32)
        return depth

    def get_single_hand_mask(self, frame):
        mask = load_image(os.path.join(self.scene_dir, 'process_dir', 'hand_masks', f'{frame}.png'))
        resolution = [mask.shape[1]//self.downsample_ratio, mask.shape[0]//self.downsample_ratio]
        mask = cv2.resize(mask.astype(np.float64), resolution, interpolation=cv2.INTER_NEAREST).astype(np.int32)
        return mask

    def get_single_object_candidate_mask(self, frame):
        mask = load_image(os.path.join(self.scene_dir, 'process_dir', 'object_candidate_masks', f'{frame}.png'))
        resolution = [mask.shape[1]//self.downsample_ratio, mask.shape[0]//self.downsample_ratio]
        mask = cv2.resize(mask.astype(np.float64), resolution, interpolation=cv2.INTER_NEAREST).astype(np.int32)
        return mask

    # @staticmethod
    # def load_depth_from_bundle(frame, path_to_bundle):
    #     return load_image(os.path.join(path_to_bundle, 'depth', f'{frame}.png'))
    #
    # @staticmethod
    # def load_image_from_bundle(frame, path_to_bundle):
    #     return load_image(os.path.join(path_to_bundle, 'color', f'{frame}.png'))


class KinectTracks:

    def __init__(self, scene_dir):

        self.scene_dir = scene_dir
        self.poses = np.array(json.load(open(os.path.join(self.scene_dir, 'poses.json'))))
        self.positions_3d = np.load(os.path.join(self.scene_dir, 'positions_3d_corrected.npy'))
        self.visibilities_3d = np.load(os.path.join(self.scene_dir, 'visibilities_3d_corrected.npy'))
        self.positions_2d = np.load(os.path.join(self.scene_dir, 'positions_2d.npy'))
        self.visibilities_2d = np.load(os.path.join(self.scene_dir, 'visibilities_2d.npy'))
        return


class IPhoneTracks:

    def __init__(self, scene_dir):

        self.scene_dir = scene_dir
        self.poses = np.array(json.load(open(os.path.join(self.scene_dir, 'poses.json'))))
        self.positions_3d = np.load(os.path.join(self.scene_dir, 'positions_3d.npy'))
        self.visibilities_3d = np.load(os.path.join(self.scene_dir, 'visibilities_3d.npy'))
        self.positions_2d = np.load(os.path.join(self.scene_dir, 'positions_2d.npy'))
        self.visibilities_2d = np.load(os.path.join(self.scene_dir, 'visibilities_2d.npy'))
        return






