import numpy as np
from tqdm import tqdm

from utils.data.dataloaders import KinectTracks, KinectPreprocessedData
from visualization.visualize_tracks2 import TrackVisualizer
# from tracking.trackers import CoTracker
import os


def get_3d_tracks(track_data, scene_data):



    intrinsics = scene_data.get_intrinsics()

    print("Getting 3D tracks.")

    all_positions_3d = []
    all_visibilities_3d = []

    for t in tqdm(range(track_data.positions_2d.shape[0])):

        positions_3d = []
        visibilities_3d = []

        depth = scene_data.get_single_depth_frame(scene_data.frames[t])

        for track_id in range(track_data.positions_2d.shape[1]):

            pixel = list(track_data.positions_2d[t, track_id].astype(np.int32))

            if not (pixel[0] < 0 or pixel[0] >= depth.shape[1] or pixel[1] < 0 or pixel[1] >= depth.shape[0]):
                pixel_depth = depth[pixel[1], pixel[0]]

                if track_data.visibilities_2d[t, track_id] and pixel_depth > 0:
                    positions_3d.append(backproject(pixel, pixel_depth, intrinsics))
                    visibilities_3d.append(True)

                    continue

            positions_3d.append(np.zeros([3]) * np.nan)
            visibilities_3d.append(False)

        all_positions_3d.append(positions_3d)
        all_visibilities_3d.append(visibilities_3d)

    positions_3d = np.array(all_positions_3d)
    # positions_3d = np.array(all_positions_3d).transpose([1, 0, 2])
    visibilities_3d = np.array(all_visibilities_3d)

    return positions_3d, visibilities_3d


def backproject(point, depth, intrinsics):

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    z = depth
    x = (point[0] - cx) * depth / fx
    y = (point[1] - cy) * depth / fy

    return np.array([x, y, z])


if __name__ == '__main__':

    # Load 3D tracks
    scene_dir = '/home/vincent/code/motion_segment2/data/mustard8'
    track_dir = '/home/vincent/Downloads/mustard8_tracks1'

    scene_data = KinectPreprocessedData(scene_dir, downsample_ratio=2)
    track_data = KinectTracks(track_dir)

    visualizer = TrackVisualizer(scene_data, track_data)

    correct_postions3d = True

    if correct_postions3d:
        positions3d_corrected = get_3d_tracks(track_data, scene_data)

    pass

    visualizer.visualize()