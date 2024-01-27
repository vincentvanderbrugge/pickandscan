from utils.data.dataloaders import IPhoneTracks, KinectPreprocessedData, IPhoneData, IPhoneTracks
from segmentation.segment import Segmenter
import matplotlib.pyplot as plt
from utils.utils import create_masked_frame
from matplotlib.animation import FuncAnimation
from IPython import display
from skimage.io import imsave
import os
import numpy as np
from tqdm import tqdm
import shutil
import argparse
from argparse import Namespace
import json
import open3d

def custom_draw_geometry_with_rotation(pcd):

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False

    open3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                              rotate_view)

if __name__ == '__main__':

    # Configure inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    args = parser.parse_args()

    config = {'bundlesdf_outdir': 'C://Users//Vincent//Downloads//multi2',
              'model_dir': 'C://Users//Vincent//Downloads//00743//nerf'
              }
    final_config = args.__dict__
    final_config.update(config)
    args = Namespace(**final_config)

    model = open3d.io.read_triangle_mesh(os.path.join(args.model_dir, "mesh_real_world.obj"), True)
    custom_draw_geometry_with_rotation(model)
    # open3d.visualization.draw_geometries([model])

    pass