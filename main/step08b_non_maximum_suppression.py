from tqdm import tqdm
import yaml
import os
from new_utils.utils import *
from copy import deepcopy


def non_maximum_suppression(scene_dir, config):

    # Parameters

    # Calculate contains matrix
    tracks = yaml.safe_load(open(os.path.join(scene_dir, 'config.yaml'), 'r'))["output"]["object_tracks"]
    contains_matrix = np.zeros((len(tracks), len(tracks)), dtype=np.float32)

    if os.path.exists(os.path.join(scene_dir, "process_dir", "step8b_nms_contains_matrix.npy")):
        contains_matrix = np.load(os.path.join(scene_dir, "process_dir", "step8b_nms_contains_matrix.npy"))
    else:
        for i in tqdm(range(len(tracks))):
            for j in tqdm(range(len(tracks))):
                if i == j:
                    continue

                contains_matrix[i, j] = calculate_contains(scene_dir, i, j, config.non_maximum_suppression["stride"])

        np.save(os.path.join(scene_dir, "process_dir", "step8b_nms_contains_matrix.npy"),
                contains_matrix)

    # Eliminate tracks until no above-threshold pairs remain
    working_contains_matrix = deepcopy(contains_matrix)

    active_tracks = [i for i in range(len(tracks))]
    while np.max(working_contains_matrix) > config.non_maximum_suppression["threshold"]:
        try:
            max_pair = np.unravel_index(np.argmax(working_contains_matrix), working_contains_matrix.shape)
            remove_track = max_pair[1]
            active_tracks.remove(remove_track)
            working_contains_matrix[:, remove_track] = 0
            working_contains_matrix[remove_track, :] = 0
        except:
            pass

    for i, track in enumerate(tracks):
        if i in active_tracks:
            track["not_nms_suppressed"] = True
        else:
            track["not_nms_suppressed"] = False

    write_output_to_config(scene_dir, "object_tracks", tracks)


def calculate_contains(scene_dir, track_id_a, track_id_b, stride=10):
    dataset_name = scene_dir.split('/')[-1]
    mask_dir_a = os.path.join(scene_dir, "process_dir", "xmem", f'xmem_input_{dataset_name}_object{track_id_a}', 'video1')
    mask_dir_b = os.path.join(scene_dir, "process_dir", "xmem", f'xmem_input_{dataset_name}_object{track_id_b}', 'video1')
    mask_files_a = sorted(glob.glob(os.path.join(mask_dir_a, "*")))
    mask_files_b = sorted(glob.glob(os.path.join(mask_dir_b, "*")))
    contains = []
    for i in range(0, len(mask_files_a), stride):
    # for mask_file in mask_files_a:
        mask_file = mask_files_a[i]
        mask_a = np.any(cv2.imread(mask_file), axis=2)
        mask_b = np.any(cv2.imread(os.path.join(mask_dir_b, os.path.basename(mask_file))), axis=2)
        if np.sum(mask_b) > 0:
            contains_fraction = np.sum(np.logical_and(mask_a, mask_b)) / np.sum(mask_b)
            contains.append(contains_fraction)

    return np.mean(contains)
