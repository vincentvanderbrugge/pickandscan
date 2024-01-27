from utils.data.dataloaders import IPhoneTracks, KinectPreprocessedData, IPhoneData, IPhoneTracks
from skimage.io import imsave
import os
from utils.utils import load_image
from tqdm import tqdm
import numpy as np


def prepare_for_bundlesdf(scene_dir, xmem_dir):


    object_name = "_".join(xmem_dir.split('/')[-1].split('_')[-2:])

    process_dir = os.path.join(scene_dir, 'process_dir')
    mask_dir = os.path.join(process_dir, "xmem", f"xmem_input_{object_name}", "video1")

    os.makedirs(os.path.join(scene_dir, "process_dir", "bundlesdf_inputs", object_name), exist_ok=True)

    frame_ids = [int(filename[:5]) for filename in os.listdir(mask_dir)]

    os.makedirs(os.path.join(process_dir, 'bundlesdf_inputs', f'{object_name}', 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(process_dir, 'bundlesdf_inputs', f'{object_name}', 'depth'), exist_ok=True)
    os.makedirs(os.path.join(process_dir, 'bundlesdf_inputs', f'{object_name}', 'masks'), exist_ok=True)

    scene_data = IPhoneData(scene_dir)

    with open(os.path.join(process_dir, 'bundlesdf_inputs', f'{object_name}', 'cam_K.txt'), 'wb') as file:
        np.savetxt(file, scene_data.get_intrinsics(), delimiter=' ', newline='\n', fmt='%.5e')

    print("Copying frames to bundlesdf input folder.")
    for frame_id in tqdm(frame_ids):
        color_frame = scene_data.get_single_color_frame(scene_data.frames[frame_id])
        depth_frame = scene_data.get_single_depth_frame(scene_data.frames[frame_id])
        mask_frame = load_image(os.path.join(mask_dir, '%05d.png' % frame_id))

        imsave(os.path.join(process_dir, 'bundlesdf_inputs', f'{object_name}', 'rgb', '%05d.png' % int(frame_id)), color_frame,
               check_contrast=False)
        imsave(os.path.join(process_dir, 'bundlesdf_inputs', f'{object_name}', 'depth', '%05d.png' % int(frame_id)), depth_frame,
               check_contrast=False)
        imsave(os.path.join(process_dir, 'bundlesdf_inputs', f'{object_name}', 'masks', '%05d.png' % int(frame_id)), mask_frame,
               check_contrast=False)
        pass
    return

if __name__ == "__main__":

    scene_dir = "/local/home/vincentv/code/motion_segment2/data/dayfour2"
    xmem_dir = "/local/home/vincentv/code/motion_segment2/data/dayfour2/process_dir/xmem/xmem_input_dayfour2_object0"
    prepare_for_bundlesdf(scene_dir, xmem_dir)