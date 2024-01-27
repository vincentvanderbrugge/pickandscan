from utils.data.dataloaders import KinectTracks, KinectPreprocessedData
import os

if __name__ == '__main__':

    scene_dir = 'C://Users//Vincent//code//motionsegment//data//mustard8'

    scene_data = KinectPreprocessedData(scene_dir)
    scene_data.preprocess_colmap_poses()