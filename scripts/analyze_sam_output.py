from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
from PIL import Image
import os
import glob
import torch
import supervision as sv
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from new_utils.utils import *
from skimage.io import imsave
import skimage.transform as st

def load_image(path):
    image = Image.open(path)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    return image_array


def resize_mask(mask, shape):
    a = mask.astype(np.uint8)
    return st.resize(a, shape, order=0, preserve_range=True, anti_aliasing=False) == 1


def calculate_iou(mask_a, mask_b):
    intersection = np.sum(np.logical_and(mask_a, mask_b))
    union = np.sum(np.logical_or(mask_a, mask_b))
    return intersection / union


model_loaded = False

CHECKPOINT_PATH = "/local/home/vincentv/code/motion_segment2/checkpoints/sam_vit_h_4b8939.pth"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir')
    args = parser.parse_args()

    bigger_side = 1024

    output_dir = os.path.join(os.path.dirname(args.dir), os.path.basename(args.dir) + '_out')

    os.makedirs(output_dir, exist_ok=True)

    image_paths = glob.glob(os.path.join(args.dir, "*.jpg"))
    assert len(image_paths) == 1
    image_path = image_paths[0]
    mask_paths = glob.glob(os.path.join(args.dir, "*.png"))
    mask_paths.sort()

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)
    mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)

    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    factor = bigger_side / max(image_rgb.shape[:2])
    new_shape = [int(dim * factor) for dim in image_rgb.shape[:2]]
    new_shape.reverse()
    resized_img = cv2.resize(image_rgb, new_shape)

    result = mask_generator.generate(resized_img)
    detections = sv.Detections.from_sam(result)

    # annotated_image = mask_annotator.annotate(resized_img, detections)
    # side_by_side = np.hstack([cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR), annotated_image])

    annotated_image = mask_annotator.annotate(resized_img, detections)
    side_by_side = np.hstack([cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR), annotated_image])

    out_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, side_by_side)



    gt_masks = [np.any(cv2.imread(p), axis=2) for p in mask_paths]

    detected_masks = [resize_mask(detected_mask, gt_masks[0].shape) for detected_mask in detections.mask]

    # for i, gt_mask in enumerate(gt_masks):
    #
    #
    #
    #     ious = []
    #     for j, detected_mask in enumerate(detected_masks):
    #         # detected_mask = resize_mask(detected_mask, gt_mask.shape)
    #         iou = calculate_iou(gt_mask, detected_mask)
    #         ious.append(iou)
    #
    #     best_mask = detected_masks[np.argmax(ious)]
    #     # best_mask = np.resize(best_mask, image_rgb.shape[:2])
    #
    #     out_path = os.path.join(output_dir, os.path.basename(image_path))
    #     # cv2.imwrite(out_path, side_by_side)
    #     overlay_img = create_masked_frame(image_rgb, mask=best_mask)
    #     out_path = os.path.join(output_dir, f'{i}.png')
    #     imsave(out_path, overlay_img.astype(np.uint8), check_contrast=False)

    for j, detected_mask in enumerate(detected_masks):
        overlay_img = create_masked_frame(image_rgb, mask=detected_mask)
        out_path = os.path.join(output_dir, f'{j}.png')
        imsave(out_path, overlay_img.astype(np.uint8), check_contrast=False)