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

def load_image(path):
    image = Image.open(path)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    return image_array


# sam_checkpoint = "C://Users//Vincent//code//segment-anything//models//sam_vit_h_4b8939.pth"
# model_type = "vit_h"
#
# device = "cuda"
#
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)
#
# predictor = SamPredictor(sam)

model_loaded = False

# image_path = "C://Users//Vincent//code//segment-anything//data//0000.png"
# input_dir = 'C://Users//Vincent//Downloads//masters thesis'
CHECKPOINT_PATH = "/local/home/vincentv/code/motion_segment2/checkpoints/sam_vit_h_4b8939.pth"

# output_dir = os.path.join(input_dir, "output")
# output_dir = 'C://Users//Vincent//Downloads//masters thesis//sam_segmentation'
# if not os.path.isdir(output_dir):
#     os.makedirs(output_dir, exist_ok=True)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

# sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
# sam.to(device=DEVICE)
# mask_generator = SamAutomaticMaskGenerator(sam)
# mask_annotator = sv.MaskAnnotator()

# bigger_side = 1024
#
# os.makedirs(output_dir, exist_ok=True)
#
# image_paths = glob.glob(os.path.join(input_dir, "*"))
# # image_paths.sort(key=lambda path: int(os.path.basename(path)[:-4]))
# for image_path in tqdm(image_paths):
#     image_bgr = cv2.imread(image_path)
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#
#     factor = bigger_side / max(image_rgb.shape[:2])
#     new_shape = [int(dim * factor) for dim in image_rgb.shape[:2]]
#     new_shape.reverse()
#     resized_img = cv2.resize(image_rgb, new_shape)
#
#     result = mask_generator.generate(resized_img)
#     detections = sv.Detections.from_sam(result)
#     annotated_image = mask_annotator.annotate(resized_img, detections)
#     side_by_side = np.hstack([cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR), annotated_image])
#
#     out_path = os.path.join(output_dir, os.path.basename(image_path))
#     cv2.imwrite(out_path, side_by_side)
#     pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir')
    args = parser.parse_args()

    bigger_side = 1024

    output_dir = os.path.join(os.path.dirname(args.dir), os.path.basename(args.dir) + '_out')

    os.makedirs(output_dir, exist_ok=True)

    image_paths = glob.glob(os.path.join(args.dir, "*"))

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)
    mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)

    for image_path in tqdm(image_paths):
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        factor = bigger_side / max(image_rgb.shape[:2])
        new_shape = [int(dim * factor) for dim in image_rgb.shape[:2]]
        new_shape.reverse()
        resized_img = cv2.resize(image_rgb, new_shape)

        result = mask_generator.generate(resized_img)
        detections = sv.Detections.from_sam(result)
        annotated_image = mask_annotator.annotate(resized_img, detections)
        side_by_side = np.hstack([cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR), annotated_image])

        out_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(out_path, side_by_side)
