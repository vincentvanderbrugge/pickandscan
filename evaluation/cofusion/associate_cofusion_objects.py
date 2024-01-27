import os
import argparse
import glob

import imageio
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import open3d as o3d
import re
from imageio import imsave
from new_utils.utils import *
from new_utils.dataloaders import IPhoneData
from distinctipy import distinctipy
# from main.step08_xmem_tracking import prepare_xmem_tracking
from evaluation.segmentation_baseline import perform_xmem_tracking_on_dir

PATH_TO_INDEX_MAP = lambda path: int(re.findall('\d+', os.path.basename(path))[0])


def get_colors(n_labels):
    colors = distinctipy.get_colors(n_labels)
    colors = [(np.array(c)*255).astype("uint8") for c in colors]
    colors = [tuple(c) for c in colors]
    return colors


def create_masked_frame(image, mask):
    color = (255, 0, 0)

    #mask[:,:,1:] = 0
    if mask.dtype == np.dtype('bool'):
        logical_mask = mask
    else:
        logical_mask = mask == 1
    mask_img = np.zeros_like(image)
    mask_img[logical_mask] = color
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.array([gray, gray, gray]).transpose([1, 2, 0])
    try:
        gray[logical_mask] = cv2.addWeighted(gray, 0.5, mask_img, 0.5, 1)[logical_mask]
    except:
        pass

    return gray

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

        mask_img[logical_mask] = colors[mask_id]
    any_mask = mask != 0
    gray[any_mask] = cv2.addWeighted(gray, 0.5, mask_img, 0.5, 1)[any_mask]

    return gray


def prepare_xmem_tracking(scene_dir, mask, object_id, frame_id):
    process_dir = os.path.join(scene_dir, 'cofusion_baseline')
    dataset_name = scene_dir.split('/')[-1]

    scene_data = IPhoneData(scene_dir)
    scene_data.load_poses()
    # segmenter = Segmenter(scene_data, distance_treshold=0.02, n_comparisons=5)

    track_id = object_id

    # tracks = yaml.safe_load(open(os.path.join(scene_dir, 'config.yaml'), 'r'))["output"]["object_tracks"]

    # selected_frame_id_for_xmem = track["best_frame"]
    # frame_id = selected_frame_id_for_xmem

    object_mask = mask

    xmem_dir = os.path.join(process_dir, "xmem", f"xmem_input_{dataset_name}_object{track_id}")

    # Make XMEM folder structure
    if not os.path.isdir(os.path.join(xmem_dir, "JPEGImages", "video1")):
        os.makedirs(os.path.join(xmem_dir, "JPEGImages", "video1"),
                    exist_ok=True)

    if not os.path.isdir(os.path.join(xmem_dir, "Annotations", "video1")):
        os.makedirs(os.path.join(xmem_dir, "Annotations", "video1"),
                    exist_ok=True)

    # Save mask
    imsave(
        os.path.join(xmem_dir, "Annotations", "video1", '%05d.png' % frame_id),
        object_mask.astype(np.uint8))

    # Save images as .jpg
    print("Copying frames to xmem folder hierarchy.")
    for frame in tqdm(scene_data.frames):
        img = scene_data.get_single_color_frame(frame)
        imsave(os.path.join(xmem_dir, "JPEGImages", "video1",
                            '%05d.jpg' % int(frame)), img)

    # # Save to-be-tracked best frame
    # best_frame_vis_path = [path for path in glob.glob(os.path.join(process_dir, 'best_frame_visualizations', '*'))
    #                        if f"object{track_id}" in os.path.basename(path)][0]
    # shutil.copy(best_frame_vis_path,
    #             os.path.join(process_dir, 'xmem', f'xmem_input_{dataset_name}_object{track_id}',
    #                          f'object{track_id}_bestframe.png'))

    return xmem_dir


def overlay_gif(img_paths, mask_paths, num_labels, out_path, nth_frame=1, slow_down=1.0):
    fig, ax = plt.subplots()
    # fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    colors = get_colors(num_labels)

    # Set the initial image
    try:
        im = ax.imshow(create_multi_masked_frame(load_image(img_paths[0]), load_image(mask_paths[0]), colors), animated=True)
    except:
        pass

    def update(i):
        image = create_multi_masked_frame(load_image(img_paths[i]), load_image(mask_paths[i]), colors)
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


def overlay_gif_multi_mask_to_single(img_paths, mask_paths, label, out_path, nth_frame=1, slow_down=1.0):
    fig, ax = plt.subplots()
    # fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    # Set the initial image
    try:
        im = ax.imshow(create_masked_frame(load_image(img_paths[0]), load_image(mask_paths[0]) == label), animated=True)
    except:
        pass

    def update(i):
        image = create_masked_frame(load_image(img_paths[i]), load_image(mask_paths[i]) == label)
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

def write_state(path_to_yaml, state):
    config_dict = yaml.safe_load(open(path_to_yaml, 'r'))
    config_dict["state"] = state
    yaml.safe_dump(config_dict, open(path_to_yaml, 'w'), sort_keys=False)
    return


def read_state(path_to_yaml):
    state_dict = yaml.safe_load(open(path_to_yaml, 'r'))
    return state_dict["state"]


def find_occurences(all_mask_paths):

    object_occurrences = {}

    pattern = re.compile(r'\d+')

    for mask_path in all_mask_paths:
        match = re.search(pattern, os.path.basename(mask_path))
        frame_id = int(match.group())

        mask = np.array(Image.open(mask_path))
        labels = np.unique(mask)

        for label in labels:
            if label not in list(object_occurrences.keys()):
                object_occurrences[label] = [frame_id]
            else:
                object_occurrences[label].append(frame_id)

    return object_occurrences


def find_nested_minimum(list_of_lists):
    # Flatten the list of lists to a single list
    flattened_list = [item for sublist in list_of_lists for item in sublist]

    # Find the minimum value in the flattened list
    min_value = min(flattened_list)

    return min_value


def find_nested_maximum(list_of_lists):
    # Flatten the list of lists to a single list
    flattened_list = [item for sublist in list_of_lists for item in sublist]

    # Find the minimum value in the flattened list
    max_value = max(flattened_list)

    return max_value


def track_ground_truth_masks(scene_dir, cfn_eval_dir, xmem_model_path):
    ground_truth_mask_files = glob.glob(os.path.join(cfn_eval_dir, "ground_truth*.png"))
    ground_truth_frame_ids = {
        int(re.findall('\d+', os.path.basename(gt_fil))[0]): int(re.findall('\d+', os.path.basename(gt_fil))[1]) for
        gt_fil in ground_truth_mask_files}

    for object_id, gt_frame in ground_truth_frame_ids.items():
        path_to_mask = [gt_file for gt_file in ground_truth_mask_files if
                        int(re.findall('\d+', os.path.basename(gt_file))[0]) == object_id][0]
        gt_mask = load_image(path_to_mask) == 1
        xmem_dir = prepare_xmem_tracking(scene_dir, gt_mask, object_id, gt_frame)
        perform_xmem_tracking_on_dir(xmem_dir, xmem_dir, xmem_model_path)


def make_detection_gifs(all_mask_paths, img_dir, cfn_eval_dir):
    object_occurrences = find_occurences(all_mask_paths)
    for label in list(object_occurrences.keys()):
        if label == 0:
            continue
        img_paths = []
        # for frame_id in object_occurrences[label]:
        #     img_paths.append(os.path.join(img_dir))
        img_paths = [img_path for img_path in glob.glob(os.path.join(img_dir, "*")) if
                     int(re.findall('\d+', os.path.basename(img_path))[0]) in object_occurrences[label]]
        mask_paths = [mask_path for mask_path in all_mask_paths if
                      int(re.findall('\d+', os.path.basename(mask_path))[0]) in object_occurrences[label]]

        img_paths.sort(key=lambda path: int(re.findall('\d+', os.path.basename(path))[0]))
        mask_paths.sort(key=lambda path: int(re.findall('\d+', os.path.basename(path))[0]))

        out_path = os.path.join(cfn_eval_dir, f"overlay_object{label}.gif")

        print(f"Creating overlay_object{label}.gif")
        overlay_gif_multi_mask_to_single(img_paths, mask_paths, label, out_path)
    return object_occurrences


def associate(cfn_out_path, scene_dir, xmem_model_path):
    all_mask_paths = glob.glob(os.path.join(cfn_out_path, "outSeg*"))
    all_mask_paths.sort(key= lambda path: int(re.findall('\d+', os.path.basename(path))[0]))


    img_dir = os.path.join(scene_dir, "rgb")
    cfn_eval_dir = os.path.join(scene_dir, "cofusion_baseline")
    
    if not os.path.exists(cfn_eval_dir):
        os.makedirs(cfn_eval_dir, exist_ok=True)
    
    cfn_eval_yaml = os.path.join(cfn_eval_dir, "cofusion_eval_result.yaml")
    
    if not os.path.exists(cfn_eval_yaml):
        with open(cfn_eval_yaml, 'w') as file:
            yaml.safe_dump({'state': 0}, file, sort_keys=False)


    # Track all ground truth masks with XMem
    if read_state(cfn_eval_yaml) < 1:
        track_ground_truth_masks(cfn_eval_yaml)

        write_state(cfn_eval_yaml, 1)




    # Make gifs of all tracked detections for debugging
    object_occurrences = find_occurences(all_mask_paths)

    if read_state(cfn_eval_yaml) < 2:
        make_detection_gifs(all_mask_paths, img_dir, cfn_eval_dir)
        write_state(cfn_eval_yaml, 2)

    # Make gifs of all ground truth tracks for debugging
    if read_state(cfn_eval_yaml) < 3:
        for label in list(object_occurrences.keys()):
            if label == 0:
                continue
            img_paths = []
            # for frame_id in object_occurrences[label]:
            #     img_paths.append(os.path.join(img_dir))
            img_paths = [img_path for img_path in glob.glob(os.path.join(img_dir, "*")) if int(re.findall('\d+', os.path.basename(img_path))[0]) in object_occurrences[label]]
            mask_paths = [mask_path for mask_path in all_mask_paths if int(re.findall('\d+', os.path.basename(mask_path))[0]) in object_occurrences[label]]

            img_paths.sort(key=PATH_TO_INDEX_MAP)
            mask_paths.sort(key=PATH_TO_INDEX_MAP)

            out_path = os.path.join(cfn_eval_dir, f"overlay_object{label}.gif")

            print(f"Creating overlay_object{label}.gif")
            overlay_gif_multi_mask_to_single(img_paths, mask_paths, label, out_path)
        write_state(cfn_eval_yaml, 3)

    # Calculate all-frame ious between ground truth objects and detections
    start_frame_id = find_nested_minimum(object_occurrences.values())
    end_frame_id = find_nested_maximum(object_occurrences.values())

    num_objects = len(ground_truth_mask_files)
    num_detections = len(object_occurrences)
    object_ids = ground_truth_frame_ids.keys()
    ious = {detection_id: {object_id: [] for object_id in object_ids} for detection_id in object_occurrences.keys()}

    for frame_id in tqdm(range(start_frame_id, end_frame_id)):
        cfn_mask_path = all_mask_paths[frame_id - start_frame_id]
        cofusion_mask = load_image(cfn_mask_path) == cfn_mask_path
        for detection_id in object_occurrences.keys():
            for i, object_id in enumerate(ground_truth_frame_ids.keys()):
                if not frame_id in object_occurrences[detection_id]:
                    continue
                path_to_mask = [gt_file for gt_file in ground_truth_mask_files if
                                int(re.findall('\d+', os.path.basename(gt_file))[0]) == object_id][0]
                gt_mask = load_image(path_to_mask) == 1
                iou = calculate_iou(gt_mask, cofusion_mask == detection_id)
                ious[detection_id][object_id].append(iou)
                pass

    all_time_ious = np.zeros((num_detections, num_objects)) * np.nan

    for detection_id in object_occurrences.keys():
        for i, object_id in enumerate(ground_truth_frame_ids.keys()):
            all_time_ious[detection_id, i] = np.mean(ious[detection_id][object_id])

    frame_ids = [PATH_TO_INDEX_MAP(path) for path in all_mask_paths]
    img_paths = [path for path in img_paths if PATH_TO_INDEX_MAP(path) in frame_ids]

    overlay_gif(img_paths, all_mask_paths, len(list(object_occurrences.keys())), out_path)

    if not os.path.exists(overlay_dir):
        os.makedirs(overlay_dir, exist_ok=True)

    for label in list(object_occurrences.keys()):
        if label == 0:
            continue
        img_paths = []
        # for frame_id in object_occurrences[label]:
        #     img_paths.append(os.path.join(img_dir))
        img_paths = [img_path for img_path in glob.glob(os.path.join(img_dir, "*")) if int(re.findall('\d+', os.path.basename(img_path))[0]) in object_occurrences[label]]
        mask_paths = [mask_path for mask_path in all_mask_paths if int(re.findall('\d+', os.path.basename(mask_path))[0]) in object_occurrences[label]]

        img_paths.sort(key=lambda path: int(re.findall('\d+', os.path.basename(path))[0]))
        mask_paths.sort(key=lambda path: int(re.findall('\d+', os.path.basename(path))[0]))

        out_path = os.path.join(overlay_dir, f"overlay_object{label}.gif")

        print(f"Creating overlay_object{label}.gif")
        overlay_gif(img_paths, mask_paths, label, out_path)
    pass


class CoFusionEvaluator:

    def __init__(self, scene_dir, cofusion_output_dir, xmem_model_path):
        self.scene_dir = scene_dir
        self.dataset_name = scene_dir.split('/')[-1]
        self.cofusion_output_dir = cofusion_output_dir
        self.xmem_model_path = xmem_model_path

        # Dir shortcuts
        self.img_dir = os.path.join(scene_dir, "rgb")
        self.cfn_eval_dir = os.path.join(scene_dir, "cofusion_baseline")

        # Make dir structure
        if not os.path.exists(self.cfn_eval_dir):
            os.makedirs(self.cfn_eval_dir, exist_ok=True)

        # Make yaml
        self.cfn_eval_yaml = os.path.join(self.cfn_eval_dir, "cofusion_eval_result.yaml")

        if not os.path.exists(self.cfn_eval_yaml):
            with open(self.cfn_eval_yaml, 'w') as file:
                yaml.safe_dump({'state': 0}, file, sort_keys=False)

        # Get detection mask paths
        self.all_mask_paths = glob.glob(os.path.join(self.cofusion_output_dir, "outSeg*"))
        self.all_mask_paths.sort(key=lambda path: int(re.findall('\d+', os.path.basename(path))[0]))

        self.object_occurrences = {}

        # Ground truth ids
        self.ground_truth_mask_files = glob.glob(os.path.join(self.cfn_eval_dir, "ground_truth*.png"))
        self.ground_truth_object_ids = [int(re.findall('\d+', os.path.basename(gt_fil))[0]) for gt_fil in
                                        self.ground_truth_mask_files]
        self.ground_truth_object_ids.sort()

    def match_detections_to_ground_truth(self):

        # Track all ground truth masks with XMem
        if read_state(self.cfn_eval_yaml) < 1:
            self.track_ground_truth_masks()
            write_state(self.cfn_eval_yaml, 1)

        # Make gifs of all tracked detections for debugging
        if read_state(self.cfn_eval_yaml) < 2:
            self.object_occurrences = self.make_detection_gifs()
            write_state(self.cfn_eval_yaml, 2)

        # Make gifs of all tracked ground truth objects for debugging
        if read_state(self.cfn_eval_yaml) < 3:
            self.make_ground_truth_gifs()
            write_state(self.cfn_eval_yaml, 3)

        # Calculate mean IoU between detections and ground truth
        if read_state(self.cfn_eval_yaml) < 4:
            self.calculate_iou_detections_to_ground_truth()
            write_state(self.cfn_eval_yaml, 4)

        # Associate detections with gt objects using ious
        if read_state(self.cfn_eval_yaml) < 5:
            self.association_from_ious()
            # write_state(self.cfn_eval_yaml, 5)

        pass

    def association_from_ious(self):
        self.load_ious()
        if len(self.object_occurrences) == 0:
            self.object_occurrences = find_occurences(self.all_mask_paths)
        longest_iou_threshold = 0.3
        fp_iou_threshold = 0.3
        # Find best matches
        best_matches = {}

        for i, object_id in enumerate(self.ground_truth_object_ids):

            matches = np.where(self.ious[:, i] > longest_iou_threshold)[0]
            if len(matches) > 0:
                best_match = max(matches, key= lambda match: len(self.object_occurrences[match]))


            else:
                best_match = np.argmax(self.ious[:, i], axis=0)
            if self.ious[best_match, i] < fp_iou_threshold:
                best_matches[i] = None
            else:
                best_matches[i] = {"detection_id": int(best_match), "start": min(self.object_occurrences[best_match]), "end": max(self.object_occurrences[best_match])}

        n_true_positives = len([key for key in best_matches.keys() if not key is None])
        n_false_positives = len(self.ious) - n_true_positives
        n_false_negatives = len(self.ground_truth_object_ids) - n_true_positives

        recall = n_true_positives / (n_true_positives + n_false_negatives)
        precision = n_true_positives / (n_true_positives + n_false_positives)

        config_dict = yaml.safe_load(open(self.cfn_eval_yaml, 'r'))
        config_dict["best_detections"] = best_matches
        config_dict["precision"] = precision
        config_dict["recall"] = recall
        config_dict["n_positives"] = n_false_positives + n_true_positives
        config_dict["n_false_positives"] = n_false_positives
        config_dict["n_false_negatives"] = n_false_negatives
        config_dict["above_threshold_detections0"] = int(np.sum(self.ious[:, 0] > fp_iou_threshold))
        config_dict["above_threshold_detections1"] = int(np.sum(self.ious[:, 1] > fp_iou_threshold))
        config_dict["above_threshold_detections2"] = int(np.sum(self.ious[:, 2] > fp_iou_threshold))
        yaml.dump(config_dict, open(self.cfn_eval_yaml, 'w'), sort_keys=False)

        pass

    def load_ious(self):
        self.ious = np.load(os.path.join(self.cfn_eval_dir, "ious.npy"))

    def calculate_iou_detections_to_ground_truth(self):
        # Calculate all-frame ious between ground truth objects and detections
        if len(self.object_occurrences) == 0:
            self.object_occurrences = find_occurences(self.all_mask_paths)

        start_frame_id = find_nested_minimum(self.object_occurrences.values())
        end_frame_id = find_nested_maximum(self.object_occurrences.values())

        ground_truth_mask_files = glob.glob(os.path.join(self.cfn_eval_dir, "ground_truth*.png"))
        ground_truth_frame_ids = {
            int(re.findall('\d+', os.path.basename(gt_fil))[0]): int(re.findall('\d+', os.path.basename(gt_fil))[1]) for
            gt_fil in ground_truth_mask_files}

        num_objects = len(ground_truth_mask_files)
        num_detections = len(self.object_occurrences)
        object_ids = list(ground_truth_frame_ids.keys())
        object_ids.sort()

        ious = {detection_id: {object_id: [] for object_id in object_ids} for detection_id in self.object_occurrences.keys()}

        print("Calculating IoUs...")

        for frame_id in tqdm(range(start_frame_id, end_frame_id+1)):
            cfn_mask_path = self.all_mask_paths[frame_id - start_frame_id]
            cofusion_mask = load_image(cfn_mask_path)
            for detection_id in self.object_occurrences.keys():
                for i, object_id in enumerate(self.ground_truth_object_ids):
                    if not frame_id in self.object_occurrences[detection_id]:
                        continue
                    path_to_gt_mask = os.path.join(self.cfn_eval_dir, "xmem", f"xmem_input_{self.dataset_name}_object{object_id}", "video1", "%05d.png" % frame_id)
                    gt_mask = load_image(path_to_gt_mask) != 0
                    detection_mask = cofusion_mask == detection_id
                    iou = calculate_iou(gt_mask, detection_mask)
                    ious[detection_id][object_id].append(iou)
                    pass

        self.all_time_ious = np.zeros((num_detections, num_objects)) * np.nan

        for detection_id in self.object_occurrences.keys():
            for i, object_id in enumerate(self.ground_truth_object_ids):
                self.all_time_ious[detection_id, i] = np.mean(ious[detection_id][object_id])

        np.save(os.path.join(self.cfn_eval_dir, "ious.npy"), self.all_time_ious)

    def make_ground_truth_gifs(self, stride=1):
        scene_name = self.scene_dir.split('/')[-1]

        ground_truth_mask_files = glob.glob(os.path.join(self.cfn_eval_dir, "ground_truth*.png"))
        ground_truth_object_ids = [int(re.findall('\d+', os.path.basename(gt_fil))[0]) for gt_fil in ground_truth_mask_files]

        for track in ground_truth_object_ids:
            # Make global gif
            print(f"Creating overlay for ground truth {track}")
            global_imgs = glob.glob(os.path.join(self.img_dir, "*.png"))
            global_masks = glob.glob(
                os.path.join(self.cfn_eval_dir, 'xmem', f'xmem_input_{scene_name}_object{track}', 'video1',
                             "*.png"))
            global_imgs.sort(key=lambda filename: int(filename.split("/")[-1][:-4]))
            global_masks.sort(key=lambda filename: int(filename.split("/")[-1][:-4]))
            global_imgs = global_imgs[::stride]
            global_masks = global_masks[::stride]
            overlay_gif(global_imgs, global_masks, num_labels=2, out_path=os.path.join(self.cfn_eval_dir, f'overlay_gt{track}.gif'))

    def track_ground_truth_masks(self):
        ground_truth_mask_files = glob.glob(os.path.join(self.cfn_eval_dir, "ground_truth*.png"))
        ground_truth_frame_ids = {
            int(re.findall('\d+', os.path.basename(gt_fil))[0]): int(re.findall('\d+', os.path.basename(gt_fil))[1]) for
            gt_fil in ground_truth_mask_files}

        for object_id, gt_frame in ground_truth_frame_ids.items():
            path_to_mask = [gt_file for gt_file in ground_truth_mask_files if
                            int(re.findall('\d+', os.path.basename(gt_file))[0]) == object_id][0]
            gt_mask = load_image(path_to_mask) != 0

            xmem_dir = prepare_xmem_tracking(self.scene_dir, gt_mask, object_id, gt_frame)
            perform_xmem_tracking_on_dir(xmem_dir, xmem_dir, self.xmem_model_path)

    def get_state(self):
        state_dict = yaml.safe_load(open(self.cfn_eval_yaml, 'r'))
        return state_dict["state"]

    def write_state(self, state):
        config_dict = yaml.safe_load(open(self.cfn_eval_yaml, 'r'))
        config_dict["state"] = state
        yaml.safe_dump(config_dict, open(self.cfn_eval_yaml, 'w'), sort_keys=False)
        return

    @staticmethod
    def find_occurences(all_mask_paths):

        object_occurrences = {}

        pattern = re.compile(r'\d+')

        for mask_path in all_mask_paths:
            match = re.search(pattern, os.path.basename(mask_path))
            frame_id = int(match.group())

            mask = np.array(Image.open(mask_path))
            labels = np.unique(mask)

            for label in labels:
                if label not in list(object_occurrences.keys()):
                    object_occurrences[label] = [frame_id]
                else:
                    object_occurrences[label].append(frame_id)

        return object_occurrences

    def make_detection_gifs(self):

        object_occurrences = find_occurences(self.all_mask_paths)

        for label in list(object_occurrences.keys()):

            if label == 0:
                continue

            img_paths = [img_path for img_path in glob.glob(os.path.join(self.img_dir, "*")) if
                         PATH_TO_INDEX_MAP(img_path) in object_occurrences[label]]

            mask_paths = [mask_path for mask_path in self.all_mask_paths if
                          PATH_TO_INDEX_MAP(mask_path) in object_occurrences[label]]

            img_paths.sort(key=PATH_TO_INDEX_MAP)
            mask_paths.sort(key=PATH_TO_INDEX_MAP)

            out_path = os.path.join(self.cfn_eval_dir, f"overlay_object{label}.gif")

            print(f"Creating overlay_object{label}.gif")
            overlay_gif_multi_mask_to_single(img_paths, mask_paths, label, out_path)

        return object_occurrences





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control conversion cofusion to cofusion.")
    parser.add_argument("--cfn_out", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--xmem_path", required=True)
    args = parser.parse_args()

    cofusion_evaluator = CoFusionEvaluator(args.scene, args.cfn_out, args.xmem_path)
    cofusion_evaluator.match_detections_to_ground_truth()

    # associate(args.scene, args.cfn_out, args.xmem_path)