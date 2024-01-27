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

    scene_dir = 'C://Users//Vincent//code//motionsegment//data//wiper_combined'
    comparison_frame_id = 0
    frame_id = 35

    process_dir = os.path.join(scene_dir, 'process_dir')

    scene_data = IPhoneData(scene_dir)
    scene_data.load_computed_tracks()
    scene_data.load_poses()

    segmenter = Segmenter(scene_data, distance_treshold=0.02, n_comparisons=5)

    # Get object mask
    object_mask = segmenter.get_manipulated_object_mask_from_single_comparison(frame_number=frame_id,
                                                                   comparison_frame_index=comparison_frame_id)

    # Create mask visualization
    img = scene_data.get_single_color_frame(str(frame_id))
    visualization = create_masked_frame(img, object_mask)
    plt.imshow(visualization)
    plt.show()

    # Make XMEM folder structure
    if not os.path.isdir(os.path.join(process_dir, "xmem", "xmem_input", "JPEGImages", "video1")):
        os.makedirs(os.path.join(process_dir, "xmem", "xmem_input", "JPEGImages", "video1"), exist_ok=True)

    if not os.path.isdir(os.path.join(process_dir, "xmem", "xmem_input", "Annotations", "video1")):
        os.makedirs(os.path.join(process_dir, "xmem", "xmem_input", "Annotations", "video1"), exist_ok=True)

    # Save mask
    imsave(
        os.path.join(process_dir, "xmem", "xmem_input", "Annotations", "video1", f'{scene_data.frames[frame_id]}.png'),
        object_mask.astype(np.uint8))

    # Save mask visualization
    imsave(os.path.join(process_dir, "xmem", f'{frame_id}_visualization.png'), visualization)

    # Save images as .jpg
    for frame in scene_data.frames:
        img = scene_data.get_single_color_frame(frame)
        imsave(os.path.join(process_dir, "xmem", "xmem_input", "JPEGImages", "video1", '%05d.jpg' % int(frame)), img)

    pass
