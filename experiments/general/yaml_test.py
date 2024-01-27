import argparse
from argparse import Namespace
import yaml

from step0_preprocess import preprocess
from step00_gradslam import perform_gradslam
from step01_handsegmentation import perform_handsegmentation
from step02_initial_pointcloud import generate_initial_pointcloud


def write_state(scene_dir, state):

    pass



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-c', '--config')
    parser.add_argument('-o', '--output')

    # path_to_zip = "C://Users//Vincent//Downloads//2023-09-07--18-27-49.r3d"
    # output_dir = "C://Users//Vincent//code//motionsegment//data//krishayush1"

    args = parser.parse_args()

    d = yaml.safe_load(open(args.config, 'r'))
    args = Namespace(**d)

    path_to_zip = args.input
    scene_dir = args.scene_dir

    write_state(scene_dir, 0)
    state_yaml_path = os.path.join(scene_dir, "config.yaml")
    yaml_string = None
    # Step0: Preprocess (unpack directories, make gif of RGB images) - gradslam5;0o/pL&MUY<*i9o0p/'py
    print("Step0a: Preprocessing started.")
    preprocess(path_to_zip, scene_dir)

    #Step00: Gradslam (unpacks the poses from the ARKit - gradslam5
    print("Step0b: Gradslam started.")
    perform_gradslam(scene_dir)

    #Step01: Hand segmentation (segments out hand in each image, xmem_saves masks & visualizations)
    print("Step1: Hand segmentation started.")
    perform_handsegmentation(scene_dir)

    #Step02: Generate & save initial pointcloud (pointcloud before anything is moved; for movement detection
    print("Step2: Initial point cloud extraction started.")
    generate_initial_pointcloud(scene_dir)

    #Step03: Heuristic moving mask estimation
    # print("Step3: Heuristic moving mask started.")




