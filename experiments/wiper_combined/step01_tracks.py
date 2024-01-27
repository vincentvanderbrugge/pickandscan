from utils.data.dataloaders import IPhoneData
from tracking.trackers import CoTracker
import os

output_dir = 'C://Users//Vincent//code//motionsegment//data//wiper_combined//process_dir'
scene_data = IPhoneData('C://Users//Vincent//code//motionsegment//data//wiper_combined')
tracker = CoTracker(path_to_checkpoint='C://Users//Vincent//code//co-tracker//checkpoints//cotracker_stride_4_wind_8.pth',
                    output_dir=os.path.join(scene_data.process_dir, 'tracks'),
                    video=scene_data.get_video(),
                    depth=scene_data.get_depth(),
                    intrinsics=scene_data.get_intrinsics(),
                    nth_frame_to_sample=50,
                    n_frames_considered=150,
                    n_sample_points_per_frame=100)

tracker.get_tracks()
tracker.save_tracks(save_tracking_visualization=True)
print("Done.")
pass