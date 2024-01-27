import json

from utils.data.iphone_recordings.datasets_common import *

from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.rgbdimages import RGBDImages

if __name__ == "__main__":

    scene_dir = 'C://Users//Vincent//code//motionsegment//data//krishayush1'
    process_dir = os.path.join(scene_dir, 'process_dir')

    # output_dir = 'C://Users//Vincent//code//motionsegment//data//wiper_combined//process_dir'
    output_dir = process_dir

    cfg = load_dataset_config(
        os.path.join(scene_dir, "dataconfig.yaml")
    )
    dataset = Record3DDataset(
        config_dict=cfg,
        basedir=scene_dir,
        sequence=None,
        start=0,
        end=-1,
        stride=1,
        # desired_height=680,
        # desired_width=1200,
        desired_height=240,
        desired_width=320,
    )

    colors, depths, poses = [], [], []
    intrinsics = None
    for idx in range(len(dataset)):
        _color, _depth, intrinsics, _pose = dataset[idx]
        colors.append(_color)
        depths.append(_depth)
        poses.append(_pose)
    colors = torch.stack(colors)
    depths = torch.stack(depths)
    poses = torch.stack(poses)
    colors = colors.unsqueeze(0)
    depths = depths.unsqueeze(0)
    intrinsics = intrinsics.unsqueeze(0).unsqueeze(0)
    poses = poses.unsqueeze(0)
    colors = colors.float()
    depths = depths.float()
    intrinsics = intrinsics.float()
    poses = poses.float()

    # create rgbdimages object
    rgbdimages = RGBDImages(
        colors,
        depths,
        intrinsics,
        poses,
        channels_first=False,
        has_embeddings=False,  # KM
    )

    # SLAM
    slam = PointFusion(odom="gt", dsratio=1, device="cuda:0", use_embeddings=False)
    pointclouds, recovered_poses = slam(rgbdimages)

    import open3d as o3d

    print(pointclouds.colors_padded.shape)
    pcd = pointclouds.open3d(0)
    recovered_poses = np.asarray(recovered_poses.cpu())[0]
    arr_list = recovered_poses.tolist()

    # Save the array as a JSON file
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'poses.json'), 'w') as file:
        json.dump(arr_list, file, indent=4)
    o3d.visualization.draw_geometries([pcd])