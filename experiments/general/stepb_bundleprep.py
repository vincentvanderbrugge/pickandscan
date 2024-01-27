import os
import yaml
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import shutil
from skimage.io import imsave

from utils.utils import *
from step0_preprocess import gif_from_image_dir
from utils.data.dataloaders import IPhoneData


def create_helper_visualizations(scene_dir):

    intermediaries = get_scene_intermediaries(scene_dir)
    object_tracks = intermediaries["object_tracks"]
    scene_name = scene_dir.split('/')[-1]

    for id, track in enumerate(object_tracks):


        # Make active gif
        active_imgs = [os.path.join(scene_dir, 'rgb', f'{i}.png') for i in range(*track['period'])]
        active_masks = [
            os.path.join(scene_dir, 'process_dir', 'xmem', f'xmem_input_{scene_name}_object{id}', 'video1', '%05d.png' % i)
            for i in range(*track['period'])]
        overlay_gif(active_imgs, active_masks, os.path.join(scene_dir, 'process_dir', 'xmem', f'overlay_active{id}.gif'))

        # # Make global gif
        # global_imgs = glob.glob(os.path.join(scene_dir, 'rgb', "*.png"))
        # global_masks = glob.glob(
        #     os.path.join(scene_dir, 'process_dir', 'xmem', f'xmem_input_{scene_name}_object{id}', 'video1', "*.png"))
        # global_imgs.sort(key=lambda filename: int(filename.split("/")[-1][:-4]))
        # global_masks.sort(key=lambda filename: int(filename.split("/")[-1][:-4]))
        # overlay_gif(global_imgs, global_masks,
        #             os.path.join(scene_dir, 'process_dir', 'xmem', f'xmem_input_{scene_name}_object{id}',
        #                          'overlay_global.gif'))

    pass


def prepare_bundlesdf_input(scene_dir):
    intermediaries = get_scene_intermediaries(scene_dir)
    object_tracks = intermediaries["object_tracks"]
    scene_name = scene_dir.split('/')[-1]
    process_dir = os.path.join(scene_dir, 'process_dir')
    scene_data = IPhoneData(scene_dir)

    for id, track in enumerate(object_tracks):

        mask_dir = os.path.join(scene_dir, 'process_dir', 'xmem', f'xmem_input_{scene_name}_object{id}', 'video1')

        frame_ids = [int(filename[:5]) for filename in os.listdir(mask_dir)]

        bundlesdf_input_dir = os.path.join(process_dir, 'bundlesdf_inputs', f'bundlesdf_input_{scene_name}_object{id}')

        os.makedirs(os.path.join(bundlesdf_input_dir, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(bundlesdf_input_dir, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(bundlesdf_input_dir, 'masks'), exist_ok=True)

        with open(os.path.join(bundlesdf_input_dir, 'cam_K.txt'), 'wb') as file:
            np.savetxt(file, scene_data.get_intrinsics(), delimiter=' ', newline='\n', fmt='%.5e')

        print(f"Copying frames to bundlesdf input folder {id}.")
        for frame_id in tqdm(frame_ids):
            color_frame = scene_data.get_single_color_frame(scene_data.frames[frame_id])
            depth_frame = scene_data.get_single_depth_frame(scene_data.frames[frame_id])
            mask_frame = load_image(os.path.join(mask_dir, '%05d.png' % frame_id))

            imsave(os.path.join(bundlesdf_input_dir, 'rgb', '%05d.png' % int(frame_id)), color_frame,
                   check_contrast=False)
            imsave(os.path.join(bundlesdf_input_dir, 'depth', '%05d.png' % int(frame_id)), depth_frame,
                   check_contrast=False)
            imsave(os.path.join(bundlesdf_input_dir, 'masks', '%05d.png' % int(frame_id)), mask_frame,
                   check_contrast=False)
            pass

        target_dir = bundlesdf_input_dir
        archived = shutil.make_archive(target_dir, 'zip', target_dir)
    pass