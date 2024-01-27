from utils.data.dataloaders import IPhoneTracks, KinectPreprocessedData, IPhoneData, IPhoneTracks
from segmentation.segment import Segmenter
import matplotlib.pyplot as plt
from utils.utils import create_masked_frame
from matplotlib.animation import FuncAnimation
from IPython import display
from skimage.io import imsave
import matplotlib.animation as animation
import os
from utils.utils import load_image
from tqdm import tqdm
import numpy as np


def get_visualization(frame_id, scene_data, mask_dir):
    object_mask = load_image(os.path.join(mask_dir, '%05d.png' % frame_id)) == 1

    img = scene_data.get_single_color_frame(str(frame_id))
    visualization = create_masked_frame(img, object_mask)
    return visualization


if __name__ == '__main__':

    scene_dir = 'C://Users//Vincent//code//motionsegment//data//wiper_combined'
    process_dir = os.path.join(scene_dir, 'process_dir')
    mask_dir = os.path.join(process_dir, 'xmem//xmem_output//wiper_combined//video1')

    scene_data = IPhoneData(scene_dir)
    scene_data.load_computed_tracks()
    scene_data.load_poses()

    frame_ids = [int(filename[:5]) for filename in os.listdir(mask_dir)]

    os.makedirs(os.path.join(process_dir, 'bundle_sdf_input', 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(process_dir, 'bundle_sdf_input', 'depth'), exist_ok=True)
    os.makedirs(os.path.join(process_dir, 'bundle_sdf_input', 'masks'), exist_ok=True)

    with open(os.path.join(process_dir, 'bundle_sdf_input', 'cam_K.txt'), 'wb') as file:
        np.savetxt(file, scene_data.get_intrinsics(), delimiter=' ', newline='\n', fmt='%.5e')

    print("Copying frames to bundlesdf input folder.")
    for frame_id in tqdm(frame_ids):
        color_frame = scene_data.get_single_color_frame(scene_data.frames[frame_id])
        depth_frame = scene_data.get_single_depth_frame(scene_data.frames[frame_id])
        mask_frame = load_image(os.path.join(mask_dir, '%05d.png' % frame_id))

        imsave(os.path.join(process_dir, 'bundle_sdf_input', 'rgb', '%05d.png' % int(frame_id)), color_frame)
        imsave(os.path.join(process_dir, 'bundle_sdf_input', 'depth', '%05d.png' % int(frame_id)), depth_frame)
        imsave(os.path.join(process_dir, 'bundle_sdf_input', 'masks', '%05d.png' % int(frame_id)), mask_frame)
        pass



    # Create the figure and axes objects
    fig, ax = plt.subplots()

    start_index = frame_ids[0]
    # last_index =

    # Set the initial image
    im = ax.imshow(get_visualization(start_index, scene_data, mask_dir), animated=True)

    nth_frame = 1
    slow_down = 2.5


    def update(i):
        frame_id = start_index + i * nth_frame
        im.set_array(get_visualization(frame_id, scene_data, mask_dir))
        plt.title(f'Frame {frame_id}')
        return im,


    # Create the animation object
    animation_fig = animation.FuncAnimation(fig, update, frames=len(frame_ids) // nth_frame,
                                            interval=int(30 * nth_frame * slow_down), blit=True,
                                            repeat_delay=10, )

    animation_fig.save("C://Users//Vincent//code//motionsegment//data//wiper_combined//process_dir//xmem//xmem_output.gif")

    # if not os.path.isdir(os.path.join(process_dir, 'bundle_sdf_input')):



