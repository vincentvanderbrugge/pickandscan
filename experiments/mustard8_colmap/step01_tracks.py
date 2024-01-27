from utils.data.dataloaders import KinectPreprocessedData
from tracking.trackers import CoTracker
import os


scene_dir = 'C://Users//Vincent//code//motionsegment//data//mustard8'

scene_data = KinectPreprocessedData(scene_dir, downsample_ratio=4)

tracker = CoTracker(path_to_checkpoint='C://Users//Vincent//code//co-tracker//checkpoints//cotracker_stride_4_wind_8.pth',
                    output_dir=os.path.join(scene_data.process_dir, 'tracks'),
                    video=scene_data.get_video(),
                    depth=scene_data.get_depth(),
                    intrinsics=scene_data.get_intrinsics(),
                    nth_frame_to_sample=100,
                    n_frames_considered=800,
                    n_sample_points_per_frame=100,
                    downsample_ratio=scene_data.downsample_ratio)

tracker.get_tracks()
tracker.save_tracks(save_tracking_visualization=True)
pass