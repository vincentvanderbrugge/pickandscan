from utils.data.dataloaders import IPhoneTracks, KinectPreprocessedData, IPhoneData, IPhoneTracks
from segmentation.segment import Segmenter
import matplotlib.pyplot as plt
from utils.utils import create_masked_frame
import os

if __name__ == '__main__':

    # Load 3D IPhoneTracks
    scene_dir = 'C://Users//Vincent//code//motionsegment//data//wiper_combined'
    # track_dir = 'C://Users//Vincent//code//motionsegment//data//output//wiper_flat'

    scene_data = IPhoneData(scene_dir)
    # track_data = IPhoneTracks(track_dir)

    scene_data.load_computed_tracks()
    scene_data.load_poses()

    frame_id = 100

    segmenter = Segmenter(scene_data, distance_treshold=0.02, n_comparisons=5)
    # moving_mask = segmenter.get_moving_mask(frame_id)
    moving_mask = segmenter.get_moving_mask_from_cotracks(50)

    img = scene_data.get_single_color_frame(str(frame_id))
    visualization = create_masked_frame(img, moving_mask)
    plt.imshow(visualization)

    # track_data.load_poses()


    # Transform to canonical coordinate frame


    # Extract moved points