import argparse
import torch
from skimage.io import imsave
from tqdm import tqdm
import shutil

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import sam_model_registry, SamPredictor
import supervision as sv

from new_utils.utils import *
from new_utils.dataloaders import IPhoneData
from segmentation.segment import Segmenter
from main.step08_xmem_tracking import perform_xmem_tracking_on_dir


def segment_sam(img, checkpoint_path):
    CHECKPOINT_PATH = checkpoint_path
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

    bigger_side = 256

    # os.makedirs(output_dir, exist_ok=True)

    factor = bigger_side / max(img.shape[:2])
    new_shape = [int(dim * factor) for dim in img.shape[:2]]
    new_shape.reverse()
    # resized_img = cv2.resize(img, new_shape)
    resized_img = img

    result = mask_generator.generate(resized_img)
    detections = sv.Detections.from_sam(result)
    annotated_image = mask_annotator.annotate(resized_img, detections)

    return detections, result, annotated_image

def calculate_iou(mask_a, mask_b):
    intersection = np.sum(np.logical_and(mask_a, mask_b))
    union = np.sum(np.logical_or(mask_a, mask_b))
    return intersection / union


def prepare_xmem_tracking(scene_dir, mask, object_id, frame_id):
    process_dir = os.path.join(scene_dir, 'divide_experiment')
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
        # img = scene_data.get_single_color_frame(frame)
        img = cv2.imread(os.path.join(args.scene_dir, "rgbd", f"{frame}.jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imsave(os.path.join(xmem_dir, "JPEGImages", "video1",
                            '%05d.jpg' % int(frame)), img)

    # # Save to-be-tracked best frame
    # best_frame_vis_path = [path for path in glob.glob(os.path.join(process_dir, 'best_frame_visualizations', '*'))
    #                        if f"object{track_id}" in os.path.basename(path)][0]
    # shutil.copy(best_frame_vis_path,
    #             os.path.join(process_dir, 'xmem', f'xmem_input_{dataset_name}_object{track_id}',
    #                          f'object{track_id}_bestframe.png'))

    return xmem_dir


def create_overlay_gif(scene_dir, mask_dir, output_path, stride=1):
    global_imgs = glob.glob(os.path.join(scene_dir, 'rgbd', "*.jpg"))
    global_masks = glob.glob(os.path.join(mask_dir, "*.png"))
    global_imgs.sort(key=lambda filename: int(filename.split("/")[-1][:-4]))
    global_masks.sort(key=lambda filename: int(filename.split("/")[-1][:-4]))
    global_imgs = global_imgs[::stride]
    global_masks = global_masks[::stride]
    overlay_gif(global_imgs, global_masks, output_path)


def bundleprep(scene_dir, xmem_dir, output_path):

    scene_data = IPhoneData(scene_dir)
    dataset_name = scene_dir.split('/')[-1]

    mask_dir = os.path.join(xmem_dir, 'video1')

    frame_ids = [int(filename[:5]) for filename in os.listdir(mask_dir)]

    bundlesdf_input_dir = os.path.join(output_path)

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')

    args = parser.parse_args()

    config = load_config(args)

    if not os.path.exists(os.path.join(config.scene_dir, 'divide_experiment')):
        os.makedirs(os.path.join(config.scene_dir, 'divide_experiment'), exist_ok=True)

    gt_path = config.ground_truth_object_mask
    gt_mask = cv2.imread(gt_path)
    gt_mask = np.any(gt_mask, axis=2)

    # Segment with SAM
    image_path = os.path.join(args.scene_dir, "rgbd", f"{config.segmentation_frame}.jpg")
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # detections, result, annotated_image = segment_sam(image_rgb, config.sam_model_path)
    #
    # # Choose mask with highest IoU
    # ious = []
    # for mask in detections.mask:
    #     iou = calculate_iou(gt_mask, mask)
    #     ious.append(iou)

    # best_mask = detections.mask[np.argmax(ious)]
    best_mask = gt_mask

    # Visualize chosen frame with mask overlay
    overlay_img = create_masked_frame(image_rgb, mask=best_mask)
    imsave(os.path.join(config.scene_dir, 'divide_experiment', f"target_mask_obj{config.object_id}_{config.segmentation_frame}.png"),
           overlay_img.astype(np.uint8), check_contrast=False)

    # XMem tracking
    xmem_dir = prepare_xmem_tracking(config.scene_dir, best_mask, config.object_id, config.segmentation_frame)
    perform_xmem_tracking_on_dir(xmem_dir, xmem_dir, config.xmem_model_path)

    # xmem_dir = '/local/home/vincentv/code/motion_segment2/data/data0112_1048/divide_experiment/xmem/xmem_input_data0112_1048_objecta'


    # Visualize XMem tracking results
    create_overlay_gif(scene_dir=config.scene_dir,
                       mask_dir=os.path.join(xmem_dir, 'video1'),
                       output_path=os.path.join(config.scene_dir, 'divide_experiment', f'overlay_global{config.object_id}.gif'))


    # # Bundleprep
    # dataset_name = config.scene_dir.split('/')[-1]
    # bundleprep(scene_dir=config.scene_dir, xmem_dir=xmem_dir,
    #            output_path=os.path.join(config.scene_dir, 'divide_experiment',f'bundlesdf_input_{dataset_name}_segbaseline_object{config.object_id}'))


    pass