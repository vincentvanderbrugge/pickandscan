from typing_extensions import Unpack

import copy

import matplotlib
import numpy as np
import open3d as o3d
from tqdm import tqdm
# import keyboard
import time
import os
# import keyboard
# from pynput import keyboard
import random
import matplotlib.pyplot as plt
from skimage.io import imsave
import distinctipy

from new_utils.utils import *
from new_utils.dataloaders import KinectTracks, KinectPreprocessedData, SceneData


class Segmenter:

    def __init__(self,
                 data: SceneData,
                 n_frames_considered=None,
                 distance_treshold=0.1,
                 color_threshold=20,
                 use_color_constraint=False,
                 n_comparisons=5):

        self.scene_loader = data
        self.distance_threshold = distance_treshold
        self.color_threshold = color_threshold
        self.use_color_constraint = use_color_constraint
        self.n_comparisons = n_comparisons

        self.frames = self.scene_loader.get_frames()
        if n_frames_considered is None:
            self.n_frames_considered = len(self.frames)
        self.frames = self.frames[:n_frames_considered]

        self.intrinsics = data.get_intrinsics()

        self.pointclouds = None

    def get_pcd_at_frame(self, frame_id):

        # Make RGB point cloud
        color = self.scene_loader.get_single_color_frame(frame_id)
        depth = self.scene_loader.get_single_depth_frame(frame_id)
        rgb_pointcloud, defined_mask = get_point_cloud2(depth, color, self.intrinsics)

        return rgb_pointcloud, defined_mask

    def load_pointclouds(self):

        self.scene_loader.pointclouds = []
        self.scene_loader.points_defined_in_image_masks = []

        print("Loading per-frame pointclouds")

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

    def get_moving_mask(self, frame_index):

        frame_index = self.scene_loader.frames[frame_index]

        if self.scene_loader.pointclouds is None:
            self.load_pointclouds()

        comparison_frames = random.choices(self.scene_loader.frames, k=self.n_comparisons)
        difference_masks = []

        for comparison_frame_id in comparison_frames:
            current_difference_mask = self.get_difference_mask(frame_index, comparison_frame_id)
            difference_masks.append(current_difference_mask)

        mask = self.difference_mask_pooling(difference_masks)

        return mask == 1

    def get_moving_mask_from_single_comparison(self, frame_index, comparison_frame_index):

        frame_index = self.scene_loader.frames[frame_index]
        comparison_frame_index = self.scene_loader.frames[comparison_frame_index]

        if self.pointclouds is None:
            self.load_pointclouds()

        difference_mask = self.get_difference_mask(frame_index, comparison_frame_index)

        return difference_mask == 1

    def get_manipulated_object_mask_given_comparison_frame_id(self, comparison_frame_index):

        # object_mask_candidates = []

        # for i in tqdm(range(len(self.scene_loader.frames))):
        #     object_mask_candidate = self.get_manipulated_object_mask_from_single_comparison(i, comparison_frame_index)
        #     object_mask_candidates.append(object_mask_candidate)

        hand_masks = [self.scene_loader.get_single_hand_mask(str(frame_index)) for frame_index in
                      range(len(self.scene_loader.frames))]
        object_candidate_masks = [self.scene_loader.get_single_object_candidate_mask(str(frame_index)) for frame_index
                                  in range(len(self.scene_loader.frames))]

        if self.pointclouds is None:
            self.load_pointclouds()

        metrics = {}

        metrics['candidate_mask_area'] = [mask.sum() for mask in object_candidate_masks]

        metrics['candidate_mask_area_delta'] = []

        for i in range(1, len(object_candidate_masks) - 1):
            delta1 = np.linalg.norm(metrics['candidate_mask_area'][i - 1] - metrics['candidate_mask_area'][i])
            delta2 = np.linalg.norm(metrics['candidate_mask_area'][i] - metrics['candidate_mask_area'][i + 1])
            delta = (delta1 + delta2) / 2
            metrics['candidate_mask_area_delta'].append(delta)

        metrics['candidate_mask_area_delta'] = [max(metrics['candidate_mask_area_delta']) + 1] + metrics[
            'candidate_mask_area_delta'] + [max(metrics['candidate_mask_area_delta']) + 1]

        for frame_number in tqdm(range(len(self.scene_loader.frames))):
            pointcloud = self.pointclouds[frame_number]
            hand_mask = hand_masks[frame_number]
            hand_pointcloud = self.get_masked_subpointcloud(pointcloud, hand_mask)
            # nonhand_pointcloud
            pass

        argmax_object_mask_size = metrics['candidate_mask_area'].index(max(metrics['candidate_mask_area']))
        argmin_mask_size_delta = metrics['candidate_mask_area_delta'].index(min(metrics['candidate_mask_area_delta']))

        best_mask_index = argmin_mask_size_delta

        best_mask = object_candidate_masks[best_mask_index]

        return best_mask, best_mask_index

        pass

    @staticmethod
    def create_masked_frame(image, mask):
        color = (255, 0, 0)

        # mask[:,:,1:] = 0
        logical_mask = mask == 1
        mask_img = np.zeros_like(image)
        mask_img[logical_mask] = color
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.array([gray, gray, gray]).transpose([1, 2, 0])
        try:
            gray[logical_mask] = cv2.addWeighted(gray, 0.5, mask_img, 0.5, 1)[logical_mask]
        except:
            pass

        return gray

    @staticmethod
    def create_multi_masked_frame(image, mask):
        color = (255, 0, 0)
        n_nonzero_masks = len(np.unique(mask)) - 1
        colors = distinctipy.get_colors(n_nonzero_masks)
        colors = [(255 * np.array(color)).astype(np.uint8).tolist() for color in colors]

        # mask[:,:,1:] = 0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.array([gray, gray, gray]).transpose([1, 2, 0])
        mask_img = np.zeros_like(image)
        for i, mask_id in enumerate(np.unique(mask).tolist()[1:]):
            # if mask_id == 0:
            #     continue
            logical_mask = mask == mask_id

            mask_img[logical_mask] = colors[i]
        any_mask = mask != 0
        gray[any_mask] = cv2.addWeighted(gray, 0.5, mask_img, 0.5, 1)[any_mask]

        return gray

    def get_manipulated_object_mask_from_single_comparison(self,
                                                           frame_number,
                                                           comparison_frame_index,
                                                           use_for_distance='fingertips',
                                                           use_static_pointcloud=False,
                                                           static_point_cloud=None,
                                                           debugging=False,
                                                           debugging_visualizations_dir=None,
                                                           num_pixel_threshold=100):

        frame_index = self.scene_loader.frames[frame_number]
        comparison_frame_index = self.scene_loader.frames[comparison_frame_index]

        if self.scene_loader.pointclouds is None:
            self.load_pointclouds()

        if use_static_pointcloud:
            difference_mask = self.get_difference_mask_from_pcd(frame_index, static_point_cloud)
        else:
            assert static_point_cloud is not None
            difference_mask = self.get_difference_mask(frame_index, comparison_frame_index)

        moving_mask = difference_mask == 1

        ## TODO choose single region (contiguous, close to hand points)

        # TODO remove hand
        hand_mask = self.scene_loader.get_single_hand_mask(str(frame_index))
        moving_mask[hand_mask == 1] = 0

        # TODO get regions
        next_blob_id = 1
        blob_mask = np.zeros(moving_mask.shape, dtype=np.uint32)
        for v in range(moving_mask.shape[0]):
            for u in range(moving_mask.shape[1]):
                if moving_mask[v, u]:
                    left_pixel_blob_id = blob_mask[v, max(u - 1, 0)]
                    top_pixel_blob_id = blob_mask[max(v - 1, 0), u]
                    if left_pixel_blob_id != 0 and top_pixel_blob_id != 0:
                        blob_mask[v, u] = left_pixel_blob_id
                        if left_pixel_blob_id != top_pixel_blob_id:
                            blob_mask[blob_mask == top_pixel_blob_id] = left_pixel_blob_id
                    elif left_pixel_blob_id != 0:
                        blob_mask[v, u] = left_pixel_blob_id
                    elif top_pixel_blob_id != 0:
                        blob_mask[v, u] = top_pixel_blob_id
                    else:
                        blob_mask[v, u] = next_blob_id
                        next_blob_id += 1
                else:
                    blob_mask[v, u] = 0

        area_percentage_threshold = (0.05) ** 2
        # num_pixel_threshold = moving_mask.shape[0] * moving_mask.shape[1] * area_percentage_threshold

        next_final_blob_id = 1
        for blob_id in range(1, next_blob_id):
            blob_size = np.sum(blob_mask == blob_id)
            if num_pixel_threshold > blob_size > 0:
                blob_mask[blob_mask == blob_id] = 0
            elif blob_size >= num_pixel_threshold:
                blob_mask[blob_mask == blob_id] = next_final_blob_id
                next_final_blob_id += 1

        # TODO get point cloud for each region

        # TODO get point cloud for hand

        # TODO compare, choose best one, get mask
        pointcloud = self.scene_loader.pointclouds[frame_number]
        depth_defined_mask = self.scene_loader.points_defined_in_image_masks[frame_number]
        blob_ids = [blob_id for blob_id in range(1, len(np.unique(blob_mask)))]
        blob_pcd_dict = {blob_id: self.get_masked_subpointcloud(pointcloud, blob_mask == blob_id, depth_defined_mask) for blob_id in
                         blob_ids}

        hand_pointcloud = self.get_masked_subpointcloud(pointcloud, hand_mask, depth_defined_mask)
        hand_mask_indices = np.where(hand_mask == 1)


        # If there are no blobs, return zero mask
        if not np.any(np.nonzero(blob_mask)):
            return np.zeros_like(blob_mask).astype(np.bool_)




        # If there is no hand, choose the biggest blob (most pixels)
        if len(hand_pointcloud.points) == 0:
            blob_sizes = {blob_id: np.sum(blob_mask == blob_id) for blob_id in blob_ids}
            argmax_blob_size = max(blob_sizes, key=blob_sizes.get)
            return blob_mask == argmax_blob_size

        try:
            hand_mask_height = hand_mask_indices[0].max() - hand_mask_indices[0].min()
        except:
            print("hi")
            pass

        finger_tip_mask = copy.deepcopy(hand_mask)
        finger_tip_cutoff = int(hand_mask_indices[0].min() + 0.1 * hand_mask_height)
        finger_tip_mask[finger_tip_cutoff:, :] = 0
        finger_tip_pointcloud = self.get_masked_subpointcloud(pointcloud, finger_tip_mask, depth_defined_mask)

        # use_for_distance = 'fingertips'

        mean_distances_to_hand = {}
        min_distances_to_hand = {}
        for blob_id in blob_ids:
            blob_pointcloud = blob_pcd_dict[blob_id]
            if use_for_distance == 'hand':
                distance_to_hand = hand_pointcloud.compute_point_cloud_distance(blob_pointcloud)
            elif use_for_distance == 'fingertips':
                distance_to_hand = finger_tip_pointcloud.compute_point_cloud_distance(blob_pointcloud)
                if len(distance_to_hand) == 0:
                    blob_sizes = {blob_id: np.sum(blob_mask == blob_id) for blob_id in blob_ids}
                    argmax_blob_size = max(blob_sizes, key=blob_sizes.get)
                    return blob_mask == argmax_blob_size
            else:
                raise NotImplementedError
            mean_distance_to_hand = np.array(distance_to_hand).mean()
            try:
                min_distance_to_hand = np.array(distance_to_hand).min()
            except:
                pass

            mean_distances_to_hand[blob_id] = mean_distance_to_hand
            min_distances_to_hand[blob_id] = min_distance_to_hand
            pass

        try:
            argmin_mean_distance = min(mean_distances_to_hand, key=mean_distances_to_hand.get)
        except:
            pass
        argmin_min_distance = min(min_distances_to_hand, key=min_distances_to_hand.get)

        if argmin_min_distance != argmin_mean_distance:
            print(
                "Warning: mean and min distance to the arm return different point clouds as argmin. Returning argmin_mean.")

        if debugging:
            os.makedirs(debugging_visualizations_dir, exist_ok=True)
            image = self.scene_loader.get_single_color_frame(str(frame_index))

            hand_blobs_multi_mask = hand_mask

            if use_for_distance == 'hand':
                blob_start_index = 2
            elif use_for_distance == 'fingertips':
                blob_start_index = 3
                hand_blobs_multi_mask[finger_tip_mask == 1] = 2
            else:
                raise NotImplementedError

            n_blobs = len(np.unique(blob_mask)) - 1
            for blob_id in range(1, n_blobs+1):
                hand_blobs_multi_mask[blob_mask == blob_id] = blob_start_index - 1 + blob_id

            debugging_visualization = self.create_multi_masked_frame(image, hand_blobs_multi_mask)

            imsave(os.path.join(debugging_visualizations_dir, frame_index + '.png'),
                   debugging_visualization.astype(np.uint8), check_contrast=False)

        return blob_mask == argmin_mean_distance

    def get_hand_distance(self, frame_number):
        frame_index = self.scene_loader.frames[frame_number]

        if self.scene_loader.pointclouds is None:
            self.load_pointclouds()

        pointcloud = self.scene_loader.pointclouds[frame_number]
        hand_mask = self.scene_loader.get_single_hand_mask(str(frame_index))

        hand_pointcloud = self.get_masked_subpointcloud(pointcloud, hand_mask)
        not_hand_pointcloud = self.get_masked_subpointcloud(pointcloud, np.logical_not(hand_mask))
        # hand_mask_indices = np.where(hand_mask == 1)

        # hand_mask_height = hand_mask_indices[0].max() - hand_mask_indices[0].min()
        # finger_tip_mask = copy.deepcopy(hand_mask)
        # finger_tip_cutoff = int(hand_mask_indices[0].min() + 0.1 * hand_mask_height)
        # finger_tip_mask[finger_tip_cutoff:, :] = 0
        # finger_tip_pointcloud = self.get_masked_subpointcloud(pointcloud, finger_tip_mask)

        distances = np.array(not_hand_pointcloud.compute_point_cloud_distance(hand_pointcloud))

        raise NotImplementedError

    def get_fingertip_distance(self, frame_number, static_pointcloud):
        frame_index = self.scene_loader.frames[frame_number]

        if self.scene_loader.pointclouds is None:
            self.load_pointclouds()

        pointcloud = self.scene_loader.pointclouds[frame_number]
        depth_defined_mask = self.scene_loader.points_defined_in_image_masks[frame_number]
        hand_mask = self.scene_loader.get_single_hand_mask(str(frame_index))
        object_candidate_mask = self.scene_loader.get_single_object_candidate_mask(str(frame_index))

        # hand_pointcloud = self.get_masked_subpointcloud(pointcloud, hand_mask)
        # not_hand_pointcloud = self.get_masked_subpointcloud(pointcloud, np.logical_not(hand_mask))

        if np.sum(hand_mask) == 0:
            return np.array([np.nan]), np.array([np.nan])

        hand_mask_indices = np.where(hand_mask == 1)
        hand_mask_height = hand_mask_indices[0].max() - hand_mask_indices[0].min()
        finger_tip_mask = copy.deepcopy(hand_mask)
        finger_tip_cutoff = int(hand_mask_indices[0].min() + 0.1 * hand_mask_height)
        finger_tip_mask[finger_tip_cutoff:, :] = 0

        if np.sum(finger_tip_mask) == 0:
            return np.array([np.nan]), np.array([np.nan])

        fingertip_pointcloud = self.get_masked_subpointcloud(pointcloud, finger_tip_mask, depth_defined_mask)
        object_candidate_pointcloud = self.get_masked_subpointcloud(pointcloud, object_candidate_mask, depth_defined_mask)

        distances_fingertip_to_static = np.array(fingertip_pointcloud.compute_point_cloud_distance(static_pointcloud))
        distances_fingertip_to_object_candidate = np.array(fingertip_pointcloud.compute_point_cloud_distance(object_candidate_pointcloud))

        return distances_fingertip_to_static, distances_fingertip_to_object_candidate

        raise NotImplementedError

    def get_hand_mask_area(self, frame_number):

        frame_index = self.scene_loader.frames[frame_number]

        hand_mask = self.scene_loader.get_single_hand_mask(str(frame_index))


        return hand_mask.sum()

    def get_masked_subpointcloud(self, pointcloud, mask, depth_defined_mask):
        # indices = np.ravel_multi_index(np.where(mask), mask.shape)
        pixel_coords = list(zip(*np.where(mask)))
        indices = self.pixel_coords_to_point_ids(pixel_coords, depth_defined_mask)
        masked_subpointcloud = pointcloud.select_by_index(indices)
        return masked_subpointcloud

    def get_moving_mask_from_cotracks(self, frame_id):

        self.scene_loader.load_computed_tracks()
        self.scene_loader.load_poses()
        self.load_pointclouds()

        num_tracks = self.scene_loader.positions_2d.shape[1]

        depth_sample_frame = self.scene_loader.get_single_depth_frame(self.scene_loader.frames[0])
        height, width = depth_sample_frame.shape[:2]

        all_positions_world_frame = {}

        for track_id in range(0, num_tracks):
            if not self.scene_loader.visibilities_2d[frame_id, track_id]:
                continue
            current_positions_world_frame = []
            for t in range(self.scene_loader.legal_periods[track_id][0],
                           self.scene_loader.legal_periods[track_id][1] + 1):
                # if self.scene_loader.visibilities_3d[t, track_id]:

                u, v = self.scene_loader.positions_2d[t, track_id, :]
                u, v = round(u), round(v)
                point_id = v * width + u
                try:
                    point3d_worldframe = np.asarray(self.pointclouds[t].points)[point_id]
                except:
                    pass

                current_positions_world_frame.append(point3d_worldframe)
            u, v = self.scene_loader.positions_2d[frame_id, track_id, :]
            u, v = round(u), round(v)
            all_positions_world_frame[(u, v)] = current_positions_world_frame

        raise NotImplementedError

    def difference_mask_pooling(self, difference_masks):
        pooled_mask = np.zeros(difference_masks[0].shape)
        difference_masks = np.concatenate([mask[..., None] for mask in difference_masks], axis=2)
        difference_masks = difference_masks.astype(np.uint8)
        for u in range(pooled_mask.shape[0]):
            for v in range(pooled_mask.shape[1]):
                pooled_mask[u, v] = np.argmax(np.bincount(difference_masks[u, v]))
        return pooled_mask
        raise NotImplementedError

    def get_difference_mask(self, frame1_id, frame2_id):

        pcd1 = self.scene_loader.pointclouds[self.scene_loader.frames.index(frame1_id)]

        pcd2 = self.scene_loader.pointclouds[self.scene_loader.frames.index(frame2_id)]
        has_close_correspondence = self.has_close_correspondence(pcd1, pcd2)

        color = self.scene_loader.get_single_color_frame(frame1_id)
        depth1 = self.scene_loader.get_single_depth_frame(frame1_id)
        depth2 = self.scene_loader.get_single_depth_frame(frame2_id)
        points_defined_in_image_mask = self.scene_loader.points_defined_in_image_masks[
            self.scene_loader.frames.index(frame1_id)]
        mask_has_proximal_neighbor = self.close_correspondence_to_mask(has_close_correspondence,
                                                                       points_defined_in_image_mask, depth1)

        pose2 = self.scene_loader.poses[self.scene_loader.frames.index(frame2_id)]
        overlapping_mask = self.get_overlapping_mask(pcd1, pose2)
        occluded_mask = self.get_occluded_mask(pcd1, self.scene_loader.points_defined_in_image_masks[self.scene_loader.frames.index(frame1_id)], depth2, pose2)

        mask_moving = np.zeros(mask_has_proximal_neighbor.shape)
        mask_moving[mask_has_proximal_neighbor] = 0
        mask_moving[np.logical_and(np.logical_not(mask_has_proximal_neighbor), np.logical_not(occluded_mask))] = 1
        mask_moving[np.logical_not(overlapping_mask)] = 2

        return mask_moving

    def get_difference_mask_from_pcd(self, frame1_id, comparison_pcd):

        pcd1 = self.scene_loader.pointclouds[self.scene_loader.frames.index(frame1_id)]

        # pcd2 = self.scene_loader.pointclouds[self.scene_loader.frames.index(frame2_id)]
        pcd2 = comparison_pcd
        has_close_correspondence = self.has_close_correspondence(pcd1, pcd2)

        color = self.scene_loader.get_single_color_frame(frame1_id)
        depth1 = self.scene_loader.get_single_depth_frame(frame1_id)
        # depth2 = self.scene_loader.get_single_depth_frame(frame2_id)
        points_defined_in_image_mask = self.scene_loader.points_defined_in_image_masks[
            self.scene_loader.frames.index(frame1_id)]
        mask_has_proximal_neighbor = self.close_correspondence_to_mask(has_close_correspondence,
                                                                       points_defined_in_image_mask, depth1)

        # pose2 = self.scene_loader.poses[self.scene_loader.frames.index(frame2_id)]
        # overlapping_mask = self.get_overlapping_mask(pcd1, pose2)
        # occluded_mask = self.get_occluded_mask(pcd1, self.scene_loader.points_defined_in_image_masks[
        #     self.scene_loader.frames.index(frame1_id)], depth2, pose2)

        mask_moving = np.zeros(mask_has_proximal_neighbor.shape)
        mask_moving[mask_has_proximal_neighbor] = 0
        # mask_moving[np.logical_and(np.logical_not(mask_has_proximal_neighbor), np.logical_not(occluded_mask))] = 1
        mask_moving[np.logical_not(mask_has_proximal_neighbor)] = 1
        # mask_moving[np.logical_not(overlapping_mask)] = 2

        return mask_moving

    def get_overlapping_mask(self, source_pointcloud, target_pose):

        depth_sample_frame = self.scene_loader.get_single_depth_frame(self.scene_loader.frames[0])
        height, width = depth_sample_frame.shape[:2]

        overlapping_mask = np.zeros_like(depth_sample_frame)

        for pixel_id in range(len(source_pointcloud.points)):
            pixel_coords = (pixel_id // width, pixel_id % width)
            point = np.asarray(source_pointcloud.points)[pixel_id]
            u, v = self.project_onto_image_plane(point, target_pose, self.intrinsics)
            overlapping_mask[pixel_coords] = (0 <= u <= width) and (0 <= v <= height)
            pass

        return overlapping_mask

    def get_occluded_mask(self, target_pointcloud, target_points_defined_in_image, comparison_depth, comparison_pose):

        # depth_target_in_comparison_frame = self.project_point_cloud_to_depth(target_pointcloud, comparison_pose)

        # A point is occluded when the depth in the comparison frame is defined,
        # and smaller than the distance of the point to the camera in z-direction (depth_target_in_comparison_frame)
        # occluded_mask = np.logical_and(depth_target_in_comparison_frame > comparison_depth, comparison_depth != 0)

        height, width = comparison_depth.shape[:2]

        occluded_mask = np.zeros_like(comparison_depth) == 1
        # projected_depth = projected_depth.astype(np.float64)
        points = np.asarray(target_pointcloud.points)

        point_indices = [i for i in range(np.sum(target_points_defined_in_image))]
        index_frame = np.zeros_like(target_points_defined_in_image).astype(np.uint32)
        index_frame[...] = -1
        index_frame[target_points_defined_in_image] = point_indices

        for pixel_id in range(len(target_pointcloud.points)):

            point_3d = points[pixel_id]

            # Get pixel coordinates in comparison frame
            u_comparison, v_comparison = self.project_onto_image_plane(point_3d, comparison_pose, self.intrinsics)
            u_comparison, v_comparison = round(u_comparison), round(v_comparison)

            # If outside field-of-view of comparison frame, the point is not occluded - skip
            if not (0 <= u_comparison < width and 0 <= v_comparison < height):
                continue

            # Get depth of point in comparison frame
            world_to_camera_transform = inverse_transform(comparison_pose)
            point_3d_camera = np.matmul(world_to_camera_transform, np.array(list(point_3d) + [1]))
            depth = point_3d_camera[2] * 1000

            # If depth at comparison pixel to which the point projects exists,
            # and is smaller than the depth of the point, the point is occluded
            if depth > 1.05 * comparison_depth[v_comparison, u_comparison] > 0:
                v_target, u_target = [element[0] for element in np.where(index_frame == pixel_id)]
                occluded_mask[v_target, u_target] = True

        return occluded_mask

    @staticmethod
    def point_id_to_pixel_coords(point_id, points_defined_mask):
        # height, width = points_defined_mask.shape
        point_indices = [i for i in range(np.sum(points_defined_mask))]
        index_frame = np.zeros_like(points_defined_mask).astype(np.uint32)
        index_frame[points_defined_mask] = point_indices
        u, v = [element[0] for element in np.where(index_frame == point_id)]
        return u, v

    @staticmethod
    def pixel_coords_to_point_ids(pixel_coords, points_defined_mask):
        # height, width = points_defined_mask.shape
        point_indices = [i for i in range(np.sum(points_defined_mask))]
        index_frame = np.zeros_like(points_defined_mask).astype(np.uint32)
        index_frame[points_defined_mask] = point_indices
        indices = []
        for pixel in pixel_coords:
            if not points_defined_mask[pixel]:
                continue
            indices.append(index_frame[pixel])
        # u, v = [element[0] for element in np.where(index_frame == point_id)]
        return indices

    def project_point_cloud_to_depth(self, pointcloud, camera_pose):
        depth_sample_frame = self.scene_loader.get_single_depth_frame(self.scene_loader.frames[0])
        height, width = depth_sample_frame.shape[:2]

        projected_depth = np.zeros_like(depth_sample_frame)
        projected_depth = projected_depth.astype(np.float64)
        points = np.asarray(pointcloud.points)

        for pixel_id in range(len(pointcloud.points)):
            # pixel_coords = (pixel_id // width, pixel_id % width)
            point_3d = points[pixel_id]
            u, v = self.project_onto_image_plane(point_3d, camera_pose, self.intrinsics)
            u, v = round(u), round(v)
            world_to_camera_transform = inverse_transform(camera_pose)
            point_3d_camera = np.matmul(world_to_camera_transform, np.array(list(point_3d) + [1]))
            depth = point_3d_camera[2]
            if not (0 <= u < width and 0 <= v < height):
                continue
            if projected_depth[v, u] == 0:
                projected_depth[v, u] = depth
            else:
                projected_depth[v, u] = min(depth, projected_depth[v, u])


        return projected_depth

    def pixel_id_to_pixel_coords(self, pixel_id, frame_id):

        depth_defined_mask = self.scene_loader.points_defined_in_image_masks[int(frame_id)]

    @staticmethod
    def project_onto_image_plane(point_3d, camera_pose, camera_intrinsics):
        world_to_camera_transform = inverse_transform(camera_pose)
        point_3d_camera = np.matmul(world_to_camera_transform, np.array(list(point_3d) + [1]))
        camera_intrinsics[2, 2] = 1
        pixel = np.matmul(camera_intrinsics, point_3d_camera[:3])
        pixel = pixel[:2] / pixel[2]

        return pixel

    @staticmethod
    def visualize_pcd_partition(pcd, partition_array):
        ncolors = np.zeros_like(np.asarray(pcd.colors))
        ncolors[partition_array, :] = [0, 1, 0]
        ncolors[np.logical_not(partition_array), :] = [1, 0, 0]
        pcd.colors = o3d.cpu.pybind.utility.Vector3dVector(ncolors)
        o3d.visualization.draw_geometries([pcd])

    def has_close_correspondence(self, pcd1, pcd2):

        if not self.use_color_constraint:
            dists = pcd1.compute_point_cloud_distance(pcd2)
            dists = np.asarray(dists)
            return dists < self.distance_threshold
        else:
            raise NotImplementedError
            # print("building kd tree")
            # pcd2_tree = o3d.geometry.KDTreeFlann(pcd2)
            # print("kd tree built")
            # for i, point in enumerate(np.asarray(pcd1.points)):
            #     [k, idx, _] = pcd2_tree.search_radius_vector_3d(point, self.distance_threshold)
            #     if k == 0:
            #         continue
            #     else:
            #         pass
            #     pass
            #
            # return None

    @staticmethod
    def close_correspondence_to_mask(has_close_correspondence, points_defined_in_image_mask, depth):
        # print("Warning: TODO implement case where not all pixels are defined / correspond to points")

        mask = np.zeros_like(depth)
        mask[points_defined_in_image_mask] = has_close_correspondence
        # mask = np.reshape(has_close_correspondence, depth.shape)
        mask = mask == 1

        return mask
