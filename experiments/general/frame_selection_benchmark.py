from utils.data.dataloaders import IPhoneTracks, KinectPreprocessedData, IPhoneData, IPhoneTracks
from segmentation.segment import Segmenter
import matplotlib.pyplot as plt
from utils.utils import create_masked_frame
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from IPython import display
import os


def get_visualization(segmenter, frame_id, comparison_frame_id):
    object_mask = segmenter.get_manipulated_object_mask_from_single_comparison(frame_number=frame_id,
                                                                               comparison_frame_index=comparison_frame_id)

    img = scene_data.get_single_color_frame(str(frame_id))
    visualization = create_masked_frame(img, object_mask)
    return visualization


if __name__ == '__main__':


    # Configuration
    scene_dir = 'C://Users//Vincent//code//motionsegment//data//wiper_combined'
    nth_frame = 1
    comparison_frame_id = 0

    scene_data = IPhoneData(scene_dir)
    scene_data.load_poses()

    segmenter = Segmenter(scene_data, distance_treshold=0.02, n_comparisons=5)

    n_frames = len(scene_data.frames)

    object_masks = []

    object_mask, object_mask_index = segmenter.get_manipulated_object_mask_given_comparison_frame_id(comparison_frame_id)

    img = segmenter.scene_loader.get_single_color_frame(object_mask_index)
    overlay = segmenter.create_masked_frame(img, object_mask)

    plt.imshow(overlay)

    pass


