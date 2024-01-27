import argparse
from argparse import Namespace
import yaml
import os

from new_utils.utils import *

def create_overlay(scene_dir, tracks, stride=10):

    intermediaries = get_scene_intermediaries(scene_dir)
    object_tracks = intermediaries["object_tracks"]
    scene_name = scene_dir.split('/')[-1]




        # Make active gif
    #active_imgs = [os.path.join(scene_dir, 'rgb', f'{i}.png') for i in range(*track['period'])]
    #active_masks = [
    #    os.path.join(scene_dir, 'process_dir', 'xmem', f'xmem_input_{scene_name}_object{id}', 'video1', '%05d.png' % i)
    #    for i in range(*track['period'])]
    #overlay_gif(active_imgs, active_masks, os.path.join(scene_dir, 'process_dir', 'xmem', f'overlay_active{id}.gif'))

    for track in tracks:
        # Make global gif
        global_imgs = glob.glob(os.path.join(scene_dir, 'rgb', "*.png"))
        global_masks = glob.glob(
            os.path.join(scene_dir, 'process_dir', 'xmem', f'xmem_input_{scene_name}_object{track}', 'video1', "*.png"))
        global_imgs.sort(key=lambda filename: int(filename.split("/")[-1][:-4]))
        global_masks.sort(key=lambda filename: int(filename.split("/")[-1][:-4]))
        global_imgs = global_imgs[::stride]
        global_masks = global_masks[::stride]
        overlay_gif(global_imgs, global_masks,
                    os.path.join(scene_dir, 'process_dir', 'xmem', f'overlay_global{track}.gif'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    parser.add_argument('-t', '--tracks', nargs='+', type=int)
    parser.add_argument('-s', '--stride', default=10, type=int)

    args = parser.parse_args()

    config = load_config(args)

    create_overlay(config.scene_dir, config.tracks, config.stride)

