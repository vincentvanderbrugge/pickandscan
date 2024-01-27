import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation
import numpy as np
import matplotlib.animation as animation
import glob
import yaml
from argparse import Namespace

# from utils.data.dataloaders import IPhoneData

def inverse_transform(T):
    T_inv = np.eye(4)
    R_inv = np.linalg.inv(T[:3, :3])
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = np.matmul(-R_inv, T[:3, 3])
    return T_inv


def get_intrinsics(path):
    return read_matrix_from_txt_file(os.path.join(path, 'cam_K.txt'))


def get_image(frame, path):
    image = cv2.imread(os.path.join(path, 'color', f'{frame}.png'))
    return image[:, :, [2,1,0]]


def read_matrix_from_txt_file(path_to_txt):
    # Read the text file and split the contents into rows
    with open(path_to_txt, 'r') as file:
        rows = file.readlines()

    # Remove whitespace characters and split each row into columns
    matrix_data = [row.strip().split() for row in rows]

    # Convert the matrix data to a NumPy array
    matrix_array = np.array(matrix_data, dtype=float)

    return matrix_array


def colmap_pose_to_transformation_matrix(colmap_pose):

    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.array(colmap_pose[:4])).as_matrix()
    T[:3, 3] = colmap_pose[-3:]
    return T


def load_image(path):
    image = Image.open(path)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    return image_array


def decode_depth(depth):
    depthn = depth[:, :, 0] * 0
    for i in range(3):
        depthn = depthn + depth[:, :, i] * 2 ** (8 * i)

    return depthn


def load_depth_from_bundle(frame, path_to_bundle):
    depth = load_image(os.path.join(path_to_bundle, 'depth', f'{frame}'))
    # depth = decode_depth(depth)
    # depth = depth.astype(np.int32)
    return depth


def load_image_from_bundle(frame, path_to_bundle):
    image = load_image(os.path.join(path_to_bundle, 'rgb', f'{frame}'))
    # depth = decode_depth(depth)
    # depth = depth.astype(np.int32)
    return image


def get_point_cloud2(depth, color, intrinsics):

    depthimg = o3d.geometry.Image((depth / 1).astype(np.uint16))
    colorimg = o3d.geometry.Image(color.astype(np.uint8))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(colorimg, depthimg, convert_rgb_to_intensity=False)
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=depth.shape[1],
                                                                 height=depth.shape[0],
                                                                 fx=intrinsics[0,0],
                                                                 fy=intrinsics[1,1],
                                                                 cx=intrinsics[0,2],
                                                                 cy=intrinsics[1,2])
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
    # pcd2 = o3d.geometry.PointCloud.create_from_depth_image(depthimg, pinhole_camera_intrinsic, depth_scale=1.0)
    defined_mask = np.asarray(rgbd.depth) != 0

    # o3d.visualization.draw_geometries([pcd])
    return pcd, defined_mask


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


def get_scene_intermediaries(scene_dir):

    if not os.path.exists(scene_dir):
        return -1

    try:
        yaml.safe_load(open(os.path.join(scene_dir, "config.yaml"), 'r'))
    except FileNotFoundError:
        raise FileNotFoundError("Scene dir was created but without config yaml.")

    config_dict = yaml.safe_load(open(os.path.join(scene_dir, "config.yaml"), 'r'))

    try:
        return config_dict["output"]
    except KeyError:
        return -1


def overlay_gif(imgs, masks, output_file, nth_frame=1, slow_down=1.0):
    fig, ax = plt.subplots()
    # fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    # Set the initial image

    im = ax.imshow(create_masked_frame(load_image(imgs[0]), load_image(masks[0])), animated=True)

    def update(i):
        image = create_masked_frame(load_image(imgs[i]), load_image(masks[i]))
        im.set_array(image)
        plt.title(f'Frame {i}')
        return im,

    print("Started creating gif.")

    # Create the animation object
    animation_fig = animation.FuncAnimation(fig, update, frames=len(imgs) // nth_frame,
                                            interval=int(30 * nth_frame * slow_down), blit=True,
                                            repeat_delay=10, )

    animation_fig.save(output_file)

    print("Finished creating gif.")
    pass


def write_state(scene_dir, state):
    config_dict = yaml.safe_load(open(os.path.join(scene_dir, "config.yaml"), 'r'))
    config_dict["state"] = state
    yaml.safe_dump(config_dict, open(os.path.join(scene_dir, "config.yaml"), 'w'), sort_keys=False)
    return


def get_state(scene_dir):

    if not os.path.exists(scene_dir):
        return -1

    try:
        yaml.safe_load(open(os.path.join(scene_dir, "config.yaml"), 'r'))
    except FileNotFoundError:
        raise FileNotFoundError("Scene dir was created but without config yaml.")

    config_dict = yaml.safe_load(open(os.path.join(scene_dir, "config.yaml"), 'r'))

    try:
        return config_dict["state"]
    except KeyError:
        return -1


def write_output_to_config(scene_dir, key, value):
    config_dict = yaml.safe_load(open(os.path.join(scene_dir, "config.yaml"), 'r'))
    config_dict["output"][key] = value
    yaml.safe_dump(config_dict, open(os.path.join(scene_dir, "config.yaml"), 'w'), sort_keys=False)
    return


def load_config(args):
    d = yaml.safe_load(open(args.config, 'r'))
    d['config'] = args.config
    args = vars(args)
    args.update(d)
    args = Namespace(**args)
    return args


def load_or_initialize_scene_dir(config):
    if config.step == 0:
        state = get_state(config.scene_dir)
    else:
        state = config.step - 1

    if not os.path.exists(config.scene_dir):
        os.makedirs(config.scene_dir, exist_ok=True)

    if not os.path.exists(os.path.join(config.scene_dir, 'config.yaml')):
        config_dict = yaml.safe_load(open(config.config, 'r'))
        with open(os.path.join(config.scene_dir, "config.yaml"), 'w') as file:
            yaml.safe_dump(config_dict, file, sort_keys=False)

    return state


def calculate_iou(mask_a, mask_b):
    intersection = np.sum(np.logical_and(mask_a, mask_b))
    union = np.sum(np.logical_or(mask_a, mask_b))
    return intersection / union


def show(img):
    plt.imshow(img)
    plt.show()
    return
