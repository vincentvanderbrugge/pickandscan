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
5. Install other dependencies
```
pip install open3d pyliblzfse pypng natsort tyro imageio
```
6. Add this project to system path, for example:
```
conda develop path/to/motion_segment2

```