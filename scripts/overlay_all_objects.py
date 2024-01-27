import argparse
from argparse import Namespace
import yaml
import os

from new_utils.utils import *
from evaluation.cofusion.inspect_cofusion2 import get_colors

def create_multi_masked_frame(image, mask, colors):
    # color = (255, 0, 0)
    n_nonzero_masks = len(np.unique(mask)) - 1
    # colors = distinctipy.get_colors(n_nonzero_masks)
    # colors = [(255 * np.array(color)).astype(np.uint8).tolist() for color in colors]

    # mask[:,:,1:] = 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.array([gray, gray, gray]).transpose([1, 2, 0])
    mask_img = np.zeros_like(image)
    for mask_id in np.unique(mask):
        if mask_id == 0:
            continue
        logical_mask = mask == mask_id
        try:
            mask_img[logical_mask] = colors[mask_id-1]
        except:
            pass
    any_mask = mask != 0
    gray[any_mask] = cv2.addWeighted(gray, 0.5, mask_img, 0.5, 1)[any_mask]

    return gray

def get_multi_mask(index, mask_paths):
    multi_mask = np.zeros_like(load_image(mask_paths[0][0]))
    for object_id in range(len(mask_paths)):
        multi_mask = np.maximum(multi_mask, load_image(mask_paths[object_id][index]) * (object_id + 1))

    return multi_mask

def overlay_gif(img_paths, mask_paths, num_labels, out_path, nth_frame=1, slow_down=1.0):
    fig, ax = plt.subplots()
    # fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    colors = get_colors(num_labels)
    img_paths.sort(key=lambda path: int(os.path.basename(path)[:-4]))
    # Set the initial image
    try:
        im = ax.imshow(create_multi_masked_frame(load_image(img_paths[0]), get_multi_mask(0, mask_paths), colors), animated=True)
        pass
    except:
        pass

    def update(i):
        image = create_multi_masked_frame(load_image(img_paths[i]), get_multi_mask(i, mask_paths), colors)
        im.set_array(image)
        print(f"animation {i} of {len(img_paths)}")
        plt.title(f'Frame {i}')
        return im,

    print("Started creating gif.")

    # Create the animation object
    animation_fig = animation.FuncAnimation(fig, update, frames=len(img_paths) // nth_frame,
                                            interval=int(30 * nth_frame * slow_down), blit=True,
                                            repeat_delay=10, )

    animation_fig.save(out_path)

    print("Finished creating gif.")
    pass


def create_overlay_all(scene_dir, stride=10):

    intermediaries = get_scene_intermediaries(scene_dir)
    object_tracks = intermediaries["object_tracks"]
    scene_name = scene_dir.split('/')[-1]


    global_imgs = glob.glob(os.path.join(scene_dir, 'rgb', "*.png"))
    mask_paths = [glob.glob(
            os.path.join(scene_dir, 'process_dir', 'xmem', f'xmem_input_{scene_name}_object{i}', 'video1', "*.png")) for i in range(len(object_tracks))]
    for i in range(len(object_tracks)):
        mask_paths[i].sort(key= lambda path: int(os.path.basename(path)[:-4]))

    overlay_gif(global_imgs, mask_paths, num_labels=len(object_tracks), out_path=os.path.join(scene_dir, 'process_dir', 'overlay_all_global.gif'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    # parser.add_argument('-t', '--tracks', nargs='+', type=int)
    parser.add_argument('-s', '--stride', default=10, type=int)

    args = parser.parse_args()

    config = load_config(args)

    create_overlay_all(config.scene_dir, stride=config.stride)

