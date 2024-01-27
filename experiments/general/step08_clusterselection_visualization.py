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

    scene_dir = 'C://Users//Vincent//code//motionsegment//data//krishayush1'

    scene_data = IPhoneData(scene_dir)

    # scene_data.load_computed_tracks()
    scene_data.load_poses()

    segmenter = Segmenter(scene_data, distance_treshold=0.02, n_comparisons=5)

    comparison_frame_id = 0

    # Create the figure and axes objects
    fig, ax = plt.subplots()

    # Set the initial image
    im = ax.imshow(get_visualization(segmenter, 10, 0), animated=True)

    nth_frame = 1
    n_frames = len(scene_data.frames)
    slow_down = 2.5

    def update(i):
        frame_id = 10 + i*nth_frame
        print(f"Visualizing frame {frame_id}")
        im.set_array(get_visualization(segmenter, frame_id, 0))
        plt.title(f'Frame {frame_id}')
        return im,


    # Create the animation object
    animation_fig = animation.FuncAnimation(fig, update, frames=n_frames//nth_frame, interval=int(30*nth_frame*slow_down), blit=True,
                                            repeat_delay=10, )

    animation_fig.save("C://Users//Vincent//code//motionsegment//data//krishayush1//process_dir//motionsegment2.gif")

