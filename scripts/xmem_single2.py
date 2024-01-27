import argparse
from argparse import Namespace
import yaml
import os
from tqdm import tqdm
import shutil
from skimage.io import imsave

from new_utils.utils import *
from main.step08_xmem_tracking import perform_xmem_tracking_on_dir
from new_utils.dataloaders import IPhoneTracks, KinectPreprocessedData, IPhoneData, IPhoneTracks
from segmentation.segment import Segmenter


def xmem_single(scene_dir, tracks, global_stride=10):

    intermediaries = get_scene_intermediaries(scene_dir)
    object_tracks = intermediaries["object_tracks"]
    scene_name = scene_dir.split('/')[-1]
    dataset_name = scene_name
    process_dir = os.path.join(scene_dir, 'process_dir')

    track = tracks[0]
    track_id = tracks[0]

    # prepare_xmem_tracking(scene_dir, config)
    # xmem_paths = os.listdir(os.path.join(scene_dir, "process_dir", "xmem"))
    # xmem_paths = [os.path.join(scene_dir, 'process_dir', 'xmem', fname) for fname in xmem_paths if "xmem" in fname]
    # model_path = yaml.safe_load(open(os.path.join(scene_dir, "config.yaml"), 'r'))["xmem_model_path"]
    #
    # xmem_dir = [path for path in xmem_paths if f'object{track}' in path][0]

    scene_data = IPhoneData(scene_dir)
    # scene_data.load_poses()
    segmenter = Segmenter(scene_data, distance_treshold=0.02, n_comparisons=5)

    #tracks = yaml.safe_load(open(os.path.join(scene_dir, 'config.yaml'), 'r'))["output"]["object_tracks"]

    selected_frame_id_for_xmem = object_tracks[track_id]["best_frame"]
    frame_id = selected_frame_id_for_xmem

    object_mask = segmenter.scene_loader.get_single_object_candidate_mask(selected_frame_id_for_xmem)

    # Make XMEM folder structure
    if not os.path.isdir(
            os.path.join(process_dir, "xmem", f"test_xmem_input_{dataset_name}_object{track_id}", "JPEGImages",
                         "video1")):
        os.makedirs(os.path.join(process_dir, "xmem", f"test_xmem_input_{dataset_name}_object{track_id}", "JPEGImages",
                                 "video1"),
                    exist_ok=True)

    if not os.path.isdir(
            os.path.join(process_dir, "xmem", f"test_xmem_input_{dataset_name}_object{track_id}", "Annotations",
                         "video1")):
        os.makedirs(os.path.join(process_dir, "xmem", f"test_xmem_input_{dataset_name}_object{track_id}", "Annotations",
                                 "video1"),
                    exist_ok=True)

    # Save mask
    imsave(
        os.path.join(process_dir, "xmem", f"test_xmem_input_{dataset_name}_object{track_id}", "Annotations", "video1",
                     '%05d.png' % frame_id),
        object_mask.astype(np.uint8))

    # Save images as .jpg
    print("Copying frames to xmem folder hierarchy.")
    for frame in tqdm(scene_data.frames):
        img = scene_data.get_single_color_frame(frame)
        imsave(
            os.path.join(process_dir, "xmem", f"test_xmem_input_{dataset_name}_object{track_id}", "JPEGImages", "video1",
                         '%05d.jpg' % int(frame)), img)

    # Save to-be-tracked best frame
    best_frame_vis_path = [path for path in glob.glob(os.path.join(process_dir, 'best_frame_visualizations', '*'))
                           if f"object{track_id}" in os.path.basename(path)][0]
    shutil.copy(best_frame_vis_path,
                os.path.join(process_dir, 'xmem', f'test_xmem_input_{dataset_name}_object{track_id}',
                             f'object{track_id}_bestframe.png'))

    xmem_dir = os.path.join(process_dir, "xmem", f"test_xmem_input_{dataset_name}_object{track_id}")
    model_path = yaml.safe_load(open(os.path.join(scene_dir, "config.yaml"), 'r'))["xmem_model_path"]
    perform_xmem_tracking_on_dir(xmem_dir, xmem_dir, model_path)

    # tracks = yaml.safe_load(open(os.path.join(scene_dir, 'config.yaml'), 'r'))["output"]["object_tracks"]
    dataset_name = scene_dir.split('/')[-1]


    # Save gif of tracking results in track period
    # if config.save_active_gif_xmem:

    # Save active gif
    active_imgs = [os.path.join(scene_dir, 'rgb', f'{i}.png') for i in range(*object_tracks[track]['period'])]
    active_masks = [
        os.path.join(scene_dir, 'process_dir', 'xmem', f'test_xmem_input_{dataset_name}_object{track}', 'video1',
                     '%05d.png' % i)
        for i in range(*object_tracks[track]['period'])]
    overlay_gif(active_imgs, active_masks,
                os.path.join(scene_dir, 'process_dir', 'xmem', f'test_xmem_input_{scene_name}_object{track}', f'overlay_active{track}.gif'))

    # Save global gif
    global_imgs = glob.glob(os.path.join(scene_dir, 'rgb', "*.png"))
    global_masks = glob.glob(
        os.path.join(scene_dir, 'process_dir', 'xmem', f'test_xmem_input_{scene_name}_object{track}', 'video1',
                     "*.png"))
    global_imgs.sort(key=lambda filename: int(filename.split("/")[-1][:-4]))
    global_masks.sort(key=lambda filename: int(filename.split("/")[-1][:-4]))
    global_imgs = global_imgs[::global_stride]
    global_masks = global_masks[::global_stride]
    overlay_gif(global_imgs, global_masks,
                os.path.join(scene_dir, 'process_dir', 'xmem', f'test_xmem_input_{scene_name}_object{track}', f'overlay_global{track}.gif'))
    return

    #     # Make active gif
    # #active_imgs = [os.path.join(scene_dir, 'rgb', f'{i}.png') for i in range(*track['period'])]
    # #active_masks = [
    # #    os.path.join(scene_dir, 'process_dir', 'xmem', f'xmem_input_{scene_name}_object{id}', 'video1', '%05d.png' % i)
    # #    for i in range(*track['period'])]
    # #overlay_gif(active_imgs, active_masks, os.path.join(scene_dir, 'process_dir', 'xmem', f'overlay_active{id}.gif'))
    # for track in tracks:
    #     # Make global gif
    #     global_imgs = glob.glob(os.path.join(scene_dir, 'rgb', "*.png"))
    #     global_masks = glob.glob(
    #         os.path.join(scene_dir, 'process_dir', 'xmem', f'xmem_input_{scene_name}_object{track}', 'video1', "*.png"))
    #     global_imgs.sort(key=lambda filename: int(filename.split("/")[-1][:-4]))
    #     global_masks.sort(key=lambda filename: int(filename.split("/")[-1][:-4]))
    #     global_imgs = global_imgs[::stride]
    #     global_masks = global_masks[::stride]
    #     overlay_gif(global_imgs, global_masks,
    #                 os.path.join(scene_dir, 'process_dir', 'xmem', f'overlay_global{track}.gif'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    parser.add_argument('-t', '--tracks', nargs='+', type=int)
    parser.add_argument('-s', '--stride', default=10, type=int)

    args = parser.parse_args()

    config = load_config(args)

    xmem_single(config.scene_dir, config.tracks, config.stride)

