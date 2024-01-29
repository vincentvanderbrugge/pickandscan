# PickScan: Discovery and 3D reconstruction of rigid objects from handheld interactions.

Master thesis by Vincent van der Brugge.

<p float="left">
  <img src="/local/home/vincentv/code/test2/media/video5.gif" width="200" />
  <img src="/local/home/vincentv/code/test2/media/overlay_all_global-cropped3.gif" width="200" /> 
  <img src="/local/home/vincentv/code/test2/media/outputs_data1404.png" width="156" />
</p>

# Setup
1. Have CUDA 11.8 installed.
2. Create env & install pytorch
```
conda create -n motionseg python=3.10
conda activate motionseg
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pyliblzfse
```
3. Install gradslam
```
git clone https://github.com/gradslam/gradslam.git
cd gradslam
pip install -e .[dev]
```
4. Install EgoHos
```
git clone https://github.com/owenzlz/EgoHOS
cd EgoHos
# Change requirements.txt to exclude torch, torchvision
pip install -r requirements.txt
pip install -U openmim
#Install mmcv from source
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
# In mmcv's setup.py, change all c++14 compiler flags to c++17 
# (e.g. replace all 'c++14' strings with 'c++17'
pip install -e . -v
cd ../EgoHos/mmsegmentation
pip install -v -e .
```
5. Install XMem. See: https://github.com/hkchengrex/XMem.
6. Install other dependencies
```
pip install open3d pyliblzfse pypng natsort tyro imageio
```
7. Add this project to system path, for example:
```
conda develop path/to/motion_segment2

```
8. To reconstruct the discovered objects, using the intermediary outputs provided by our method, BundleSDF's docker container is required, see: https://github.com/NVlabs/BundleSDF. 

9. To run the SAM* baseline, install Segment Anything, see https://github.com/facebookresearch/segment-anything.

# Usage

### Processing

Running our method (on a .r3d scan, captured by Record3D):

```
python scripts/reconstruct_factorized.py --config path/to/run_ours_config.yaml
```

Running the SAM* baseline on a given object A:

```
python evaluation/segmentation_baseline.py --config path/to/run_samstar_objectA_config.yaml
```

### Evaluation

Converting a data directory in our format to CoFusion's input directory format:

```
python scripts/convert_ours_to_cfsn.py --input path/to/input_dir/ --output path/to/output_dir
```

Evaluating an object reconstruction for an object A by our method / SAM* (chamfer distance):

```
python evaluation/error_calculation.py --config path/to/eval_objectA_config.yaml
```

Evaluating an object reconstruction for an object A by CoFusion (chamfer distance):

```
python evaluation/error_calculation.py --config path/to/eval_cofusion_objectA_config.yaml
```

### Notes

* the core of this project is the code in ```main/```
* see ```example_configs/``` for examples of config yaml files. 
* to convert a directory hierarchy in the CoFusion input format to the actual .klg input file used by CoFusion, see https://github.com/martinruenz/dataset-tools.
# References

This project makes use of the following projects and their codebases:

[1] B. Wen, J. Tremblay, V. Blukis, S. Tyree, T. Muller, A. Evans, D. Fox,
J. Kautz, and S. Birchfield, “Bundlesdf: Neural 6-dof tracking and
3d reconstruction of unknown objects,” 2023 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pp. 606–617, 2023. 
Github: https://github.com/NVlabs/BundleSDF

[2] H. K. Cheng and A. G. Schwing, “Xmem: Long-term video object
segmentation with an atkinson-shiffrin memory model,” in European
Conference on Computer Vision, 2022. 
Github: https://github.com/hkchengrex/XMem

[3] L. Zhang, S. Zhou, S. Stent, and J. Shi, “Fine-grained egocentric hand-
object segmentation: Dataset, model, and applications,” in European
Conference on Computer Vision, 2022.
Github: https://github.com/owenzlz/EgoHOS

For evaluation, we also use:

[4] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao,
S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollár, and R. Girshick, “Segment
anything,” arXiv:2304.02643, 2023. Github: https://github.com/facebookresearch/segment-anything

[5] M. Rünz and L. de Agapito, “Co-fusion: Real-time segmentation, track-
ing and fusion of multiple objects,” 2017 IEEE International Conference on
Robotics and Automation (ICRA), pp. 4471–4478, 2017. Github: https://github.com/martinruenz/co-fusion