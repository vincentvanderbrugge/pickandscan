from utils.data.dataloaders import IPhoneTracks, KinectPreprocessedData, IPhoneData, IPhoneTracks
from visualization.visualize_tracks2 import TrackVisualizer
import os

if __name__ == '__main__':

    # Load 3D IPhoneTracks
    scene_dir = 'C://Users//Vincent//code//motionsegment//data//wiper_flat'
    # track_dir = 'C://Users//Vincent//code//motionsegment//data//output//wiper_flat'

    scene_data = IPhoneData(scene_dir)
    # track_data = IPhoneTracks(track_dir)

    scene_data.load_computed_tracks()
    scene_data.load_poses()

    visualizer = TrackVisualizer(scene_data)
    visualizer.visualize2()


    # track_data.load_poses()


    # Transform to canonical coordinate frame


    # Extract moved points