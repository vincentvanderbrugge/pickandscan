import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import glob

scene_dir = '/local/home/vincentv/code/motion_segment2/data/refactor_test_data1226_1432'
bundlesdf_dir = '/local/home/vincentv/code/motion_segment2/data/data1226_1432_object0'
out_path = '/local/home/vincentv/code/motion_segment2/data/misc/pose_video.avi'

root_color = (66, 245, 155)
axis_color = {'x': (255,0,0),
              'y': (0,255,0),
              'z': (0,0,255)}


def project_pose_axis(image, pose, intrinsics):

    # Get root pixel
    root_u, root_v = project_onto_image_plane(pose[:3, 3], intrinsics)

    # Get coordinate axis tip pixels
    x = transform_point(np.array((0.1, 0, 0)), pose)
    x_axis_u, x_axis_v = project_onto_image_plane(x, intrinsics)
    y = transform_point(np.array((0, 0.1, 0)), pose)
    y_axis_u, y_axis_v = project_onto_image_plane(y, intrinsics)
    z = transform_point(np.array((0, 0, 0.1)), pose)
    z_axis_u, z_axis_v = project_onto_image_plane(z, intrinsics)

    # Draw root
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    radius = 5
    draw.ellipse((root_u - radius, root_v - radius, root_u + radius, root_v + radius), fill=root_color, outline=root_color)

    # Draw coordinate axis
    draw.line((root_u, root_v, x_axis_u, x_axis_v), fill=axis_color["x"], width=4)
    draw.line((root_u, root_v, y_axis_u, y_axis_v), fill=axis_color["y"], width=4)
    draw.line((root_u, root_v, z_axis_u, z_axis_v), fill=axis_color["z"], width=4)

    img = np.array(img_pil)

    return img


def project_onto_image_plane(point3d, intrinsics):
    pixel = np.matmul(intrinsics, point3d)
    pixel = pixel[:2] / pixel[2]
    x, y = pixel.astype(np.int32).tolist()

    return x, y


def transform_point(point, transform):
    point_copy = np.ones((4,))
    point_copy[:3] = point

    transformed_point = np.matmul(transform, point_copy)[:3]

    return transformed_point


def get_image(frame, path):

    image = cv2.imread(os.path.join(path, 'color', f'{frame}.png'))
    return image[:, :, [2,1,0]]


def get_image2(frame, path):

    image = cv2.imread(os.path.join(path, 'rgb', f'{frame}.png'))
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


def get_pose(frame, path):
    return read_matrix_from_txt_file(os.path.join(path, 'ob_in_cam', f'{frame}.txt'))


def get_pose2(frame, path):
    return read_matrix_from_txt_file(os.path.join(path, 'poses', f'{frame}.txt'))


def get_intrinsics(path):
    return read_matrix_from_txt_file(os.path.join(path, 'cam_K.txt'))


def create_pose_axis_video(scene_dir, bundlesdf_dir, out_path, fps=5):

    frames = [name[:-4] for name in os.listdir(os.path.join(bundlesdf_dir, 'ob_in_cam'))]
    frames.sort()
    intrinsics = get_intrinsics(bundlesdf_dir)

    height, width, layers = get_image(frames[0], bundlesdf_dir).shape
    video = cv2.VideoWriter(filename=out_path,  # Provide a file to write the video to
                            fourcc=cv2.VideoWriter_fourcc(*'XVID'), frameSize=(width, height), fps=fps)


    for frame in frames:

        image = get_image(frame, bundlesdf_dir)
        pose = get_pose(frame, bundlesdf_dir)
        pose_axis_visualization_frame = project_pose_axis(image, pose, intrinsics)

        #pose_axis_visualization_frame = pose_axis_visualization_frame[:,:,[2,1,0]]


        if int(frame) > 20:
            show(pose_axis_visualization_frame)

        video.write(pose_axis_visualization_frame)

    cv2.destroyAllWindows()
    video.release()

def show(img):
    plt.imshow(img)
    plt.show()


def create_pose_axis_video2(scene_dir, bundlesdf_dir, out_path, fps=5):

    frames = [name[:-4] for name in os.listdir(os.path.join(path, 'poses'))]
    frames.sort()
    intrinsics = get_intrinsics(path)

    height, width, layers = get_image2(frames[0], path).shape
    video = cv2.VideoWriter(filename=out_path,  # Provide a file to write the video to
                            fourcc=cv2.VideoWriter_fourcc(*'XVID'), frameSize=(width, height), fps=fps)


    for frame in tqdm(frames):

        image = get_image2(frame, path)
        pose = get_pose2(frame, path)
        pose_axis_visualization_frame = project_pose_axis(image, pose, intrinsics)

        pose_axis_visualization_frame = pose_axis_visualization_frame[:,:,[2,1,0]]

        video.write(pose_axis_visualization_frame)

    cv2.destroyAllWindows()
    video.release()


def read_poses(bundlesdf_dir):
    pose_files = glob.glob(os.path.join(bundlesdf_dir, 'ob_in_cam', '*'))
    pose_files.sort()

    return [read_matrix_from_txt_file(pose_file) for pose_file in pose_files]


if __name__ == "__main__":
    # poses = read_poses(bundlesdf_dir)
    create_pose_axis_video(scene_dir, bundlesdf_dir, out_path)
