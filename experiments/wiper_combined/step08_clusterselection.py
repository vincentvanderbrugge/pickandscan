from utils.data.dataloaders import IPhoneTracks, KinectPreprocessedData, IPhoneData, IPhoneTracks
from segmentation.segment import Segmenter
import matplotlib.pyplot as plt
from utils.utils import create_masked_frame
from matplotlib.animation import FuncAnimation
from IPython import display
from skimage.io import imsave
import os
import numpy as np

if __name__ == '__main__':

    # Load 3D IPhoneTracks
    scene_dir = 'C://Users//Vincent//code//motionsegment//data//wiper_combined'
    process_dir = 'C://Users//Vincent//code//motionsegment//data//wiper_combined//process_dir'
    # track_dir = 'C://Users//Vincent//code//motionsegment//data//output//wiper_flat'

    scene_data = IPhoneData(scene_dir)
    # track_data = IPhoneTracks(track_dir)

    scene_data.load_computed_tracks()
    scene_data.load_poses()

    segmenter = Segmenter(scene_data, distance_treshold=0.02, n_comparisons=5)

    comparison_frame_id = 0

    # for frame_id in range(10, 110, 10):
    #
    # # frame_id = 80
    #
    #     object_mask = segmenter.get_manipulated_object_mask_from_single_comparison(frame_number=frame_id,
    #                                                                    comparison_frame_index=comparison_frame_id)
    #
    #     img = scene_data.get_single_color_frame(str(frame_id))
    #     visualization = create_masked_frame(img, object_mask)
    #     plt.imshow(visualization)
    #     plt.show()

    # for frame_id in range(10, 110, 10):

    frame_id = 80

    object_mask = segmenter.get_manipulated_object_mask_from_single_comparison(frame_number=frame_id,
                                                                   comparison_frame_index=comparison_frame_id)

    img = scene_data.get_single_color_frame(str(frame_id))
    visualization = create_masked_frame(img, object_mask)
    plt.imshow(visualization)
    plt.show()
    imsave(os.path.join(process_dir,  str(frame_id) + '.png'),
           object_mask.astype(np.uint8))
    # imsave(os.path.join(process_dir, 'hand_masks', fname + '.png'), mask_img.astype(np.uint8))


    pass
