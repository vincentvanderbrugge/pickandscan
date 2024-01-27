import argparse
import os
import cv2
import torch
import yaml
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import supervision as sv
import distinctipy
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import sam_model_registry, SamPredictor

from new_utils.dataloaders import IPhoneData


def create_multi_masked_frame(image, mask):
    color = (255, 0, 0)
    n_nonzero_masks = len(np.unique(mask)) - 1
    colors = distinctipy.get_colors(n_nonzero_masks)
    colors = [(255 * np.array(color)).astype(np.uint8).tolist() for color in colors]

    # mask[:,:,1:] = 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.array([gray, gray, gray]).transpose([1, 2, 0])
    mask_img = np.zeros_like(image)
    for i, mask_id in enumerate(np.unique(mask).tolist()[1:]):
        # if mask_id == 0:
        #     continue
        logical_mask = mask == mask_id

        mask_img[logical_mask] = colors[i]
    any_mask = mask != 0
    gray[any_mask] = cv2.addWeighted(gray, 0.5, mask_img, 0.5, 1)[any_mask]

    return gray


def display_mask(mask):
    plt.imshow(mask.astype(np.uint8)*255)
    plt.show()


def scale_mask(mask, shape):
    scaled_mask = np.zeros(shape)
    scale_factor_x = shape[0] / mask.shape[0]
    scale_factor_y = shape[1] / mask.shape[1]
    for y in range(shape[0]):
        for x in range(shape[1]):
            try:
                scaled_mask[y, x] = mask[int(y//scale_factor_y), int(x//scale_factor_x)]
            except:
                pass
    return scaled_mask == 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_dir')
    parser.add_argument('--frame_id', type=int)
    args = parser.parse_args()

    checkpoint_path = "/local/home/vincentv/code/motion_segment2/checkpoints/sam_vit_h_4b8939.pth"
    output_dir = os.path.join(args.scene_dir, "process_dir", "segmentation_comparison")


    # TODO make dirs
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    image_path = os.path.join(args.scene_dir, "rgbd", f"{args.frame_id}.jpg")

    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # TODO make and save masks with motion seg
    tracks = yaml.safe_load(open(os.path.join(args.scene_dir, 'config.yaml'), 'r'))["output"]["object_tracks"]
    #tracks = [(i, track) for i, track in enumerate(tracks) if track["not_nms_suppressed"]]

    masks = []

    scene_loader = IPhoneData(args.scene_dir)

    scene_name = args.scene_dir.split('/')[-1]

    for i, track in enumerate(tracks):
        if not track["not_nms_suppressed"]:
            continue
        global_masks = glob.glob(
            os.path.join(args.scene_dir, 'process_dir', 'xmem', f'xmem_input_{scene_name}_object{i}', 'video1', "*.png"))
        global_masks.sort(key=lambda filename: int(filename.split("/")[-1][:-4]))
        mask_path = global_masks[args.frame_id]
        mask = cv2.imread(mask_path)
        mask = np.any(mask, axis=2)
        mask = scale_mask(mask, image_bgr.shape[:2])
        masks.append(mask)

    obj_masks = np.zeros_like(masks[0], dtype=np.uint32)
    for i in range(len(masks)):
        obj_masks[masks[i]] = i + 1
    vis = create_multi_masked_frame(image_rgb, obj_masks)



    # TODO make and save masks with instance seg
    CHECKPOINT_PATH = checkpoint_path
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)
    mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)

    bigger_side = 256

    os.makedirs(output_dir, exist_ok=True)

    factor = bigger_side / max(image_rgb.shape[:2])
    new_shape = [int(dim * factor) for dim in image_rgb.shape[:2]]
    new_shape.reverse()
    # resized_img = cv2.resize(image_rgb, new_shape)
    resized_img = image_rgb

    result = mask_generator.generate(resized_img)
    detections = sv.Detections.from_sam(result)
    annotated_image = mask_annotator.annotate(resized_img, detections)
    side_by_side = np.hstack([image_bgr, annotated_image, vis])

    cv2.imwrite(os.path.join(output_dir, "sam_based.png"), annotated_image)
    cv2.imwrite(os.path.join(output_dir, "motion_based.png"), vis)
    cv2.imwrite(os.path.join(output_dir, "rgb.png"), image_bgr)
    cv2.imwrite(os.path.join(output_dir, "side_by_side.png"), side_by_side)