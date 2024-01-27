from utils.data.dataloaders import KinectTracks, KinectPreprocessedData
from visualization.visualize_tracks2 import TrackVisualizer
import os

if __name__ == '__main__':

    # Load 3D tracks
    scene_dir = 'C://Users//Vincent//code//motionsegment//data//mustard8'
    # track_dir = '/home/vincent/Downloads/mustard8_tracks1'

    scene_data = KinectPreprocessedData(scene_dir)
    scene_data.load_computed_tracks()
    # scene_data.load_poses()
    scene_data.load_colmap_poses()


    visualizer = TrackVisualizer(scene_data)
    visualizer.visualize2()


    # track_data.load_poses()


    # Transform to canonical coordinate frame


    # Extract moved points