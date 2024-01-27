import copy

from masking.mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import glob
import os
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np
from skimage.io import imsave
import pdb
import matplotlib.pyplot as plt
from argparse import Namespace
import cv2


def create_masked_frame(image, mask):
    color = (255, 0, 0)

    #mask[:,:,1:] = 0
    logical_mask = mask
    mask_img = np.zeros_like(image)
    mask_img[logical_mask] = color
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.array([gray, gray, gray]).transpose([1, 2, 0])
    try:
        gray[logical_mask] = cv2.addWeighted(gray, 0.5, mask_img, 0.5, 1)[logical_mask]
    except:
        pass

    return gray


def segment_hands(input_path):
    config = {'img_dir': '..//..//data//multi2//rgb',
              'pred_seg_dir': 'C://Users//Vincent//code//motionsegment//data//multi2//process_dir',
              'nth_frame': 1}

    # parser = argparse.ArgumentParser(description="")
    # parser.add_argument("--config_file",
    #                     default='./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K.py',
    #                     type=str)
    # parser.add_argument("--checkpoint_file",
    #                     default='./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/best_mIoU_iter_42000.pth',
    #                     type=str)
    # parser.add_argument("--img_dir", default='../data/train/image', type=str)
    # parser.add_argument("--pred_seg_dir",
    #                     default='./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/outputs/train_seg',
    #                     type=str)
    # args = parser.parse_args()
    # final_config = args.__dict__
    # final_config.update(config)
    # args = Namespace(**final_config)

    # args.img_dir = '..//..//data//wiper_combined//sequence0//rgb'
    # args.pred_seg_dir = 'C://Users//Vincent//code//motionsegment//data//wiper_combined//process_dir'
    pred_seg_dir = os.path.join(input_path, "process_dir")
    img_dir = os.path.relpath(os.path.join(input_path, "rgb")) # This path formulation seems to be needed by mmseg
    config_file = '/local/home/vincentv/code/motion_segment2/checkpoints/work_dirs/seg_twohands_ccda/seg_twohands_ccda.py'
    checkpoint_file = '/local/home/vincentv/code/motion_segment2/checkpoints/work_dirs/seg_twohands_ccda/best_mIoU_iter_56000.pth'

    os.makedirs(os.path.join(pred_seg_dir, 'hand_mask_visualizations'), exist_ok=True)
    os.makedirs(os.path.join(pred_seg_dir, 'hand_masks'), exist_ok=True)

    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

    alpha = 0.5

    for file in tqdm(glob.glob(img_dir + '/*')):
        fname = os.path.basename(file).split('.')[0]
        img = np.array(Image.open(os.path.join(img_dir, fname + '.png')))
        seg_result = inference_segmentor(model, file)[0]
        indices = np.unique(seg_result)
        # print(indices)
        mask_visualization = create_masked_frame(img, seg_result != 0)
        mask_img = copy.deepcopy(seg_result)
        mask_img[seg_result != 0] = 1
        imsave(os.path.join(pred_seg_dir, 'hand_mask_visualizations', fname + '.png'),
               mask_visualization.astype(np.uint8))
        imsave(os.path.join(pred_seg_dir, 'hand_masks', fname + '.png'), mask_img.astype(np.uint8), check_contrast=False)

if __name__ == "__main__":
    # Config
    config = {'img_dir': '..//..//data//multi2//rgb',
              'pred_seg_dir': 'C://Users//Vincent//code//motionsegment//data//multi2//process_dir',
              'nth_frame': 1}

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_file", default='./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K.py', type=str)
    parser.add_argument("--checkpoint_file", default='./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/best_mIoU_iter_42000.pth', type=str)
    parser.add_argument("--img_dir", default='../data/train/image', type=str)
    parser.add_argument("--pred_seg_dir", default='./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/outputs/train_seg', type=str)
    args = parser.parse_args()
    final_config = args.__dict__
    final_config.update(config)
    args = Namespace(**final_config)

    # args.img_dir = '..//..//data//wiper_combined//sequence0//rgb'
    # args.pred_seg_dir = 'C://Users//Vincent//code//motionsegment//data//wiper_combined//process_dir'
    args.config_file = 'C://Users//Vincent//code//EgoHOS//checkpoints//work_dirs//seg_twohands_ccda//seg_twohands_ccda.py'
    args.checkpoint_file = 'C://Users//Vincent//code//EgoHOS//checkpoints//work_dirs//seg_twohands_ccda//best_mIoU_iter_56000.pth'


    os.makedirs(os.path.join(args.pred_seg_dir, 'hand_mask_visualizations'), exist_ok = True)
    os.makedirs(os.path.join(args.pred_seg_dir, 'hand_masks'), exist_ok = True)

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config_file, args.checkpoint_file, device='cuda:0')

    alpha = 0.5

    # fname = os.path.basename(file).split('.')[0]
    # img = np.array(Image.open(os.path.join(args.img_dir, fname + '.jpg')))
    # seg_result = inference_segmentor(model, file)[0]
    # indices = np.unique(seg_result)
    # print(indices)
    # mask_visualization = create_masked_frame(img, seg_result != 0)
    # imsave(os.path.join(args.pred_seg_dir, fname + '.png'), mask_visualization.astype(np.uint8))


    for file in tqdm(glob.glob(args.img_dir + '/*')):
        fname = os.path.basename(file).split('.')[0]
        img = np.array(Image.open(os.path.join(args.img_dir, fname + '.png')))
        seg_result = inference_segmentor(model, file)[0]
        indices = np.unique(seg_result)
        # print(indices)
        mask_visualization = create_masked_frame(img, seg_result != 0)
        mask_img = copy.deepcopy(seg_result)
        mask_img[seg_result != 0] = 1
        imsave(os.path.join(args.pred_seg_dir, 'hand_mask_visualizations', fname + '.png'), mask_visualization.astype(np.uint8))
        imsave(os.path.join(args.pred_seg_dir, 'hand_masks', fname + '.png'), mask_img.astype(np.uint8), check_contrast=False)

    print('Done.')