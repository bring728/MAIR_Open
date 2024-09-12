
# MAIR: Multi-view Attention Inverse Rendering with 3D Spatially-Varying Lighting Estimation

This repository contains the implementation of the paper **"MAIR: Multi-view Attention Inverse Rendering with 3D Spatially-Varying Lighting Estimation"** presented at **CVPR 2023**.

---

## Timelog

- **2024-06-03**: MAIR initial commit (test code only)
- **2024-09-12**: Object insertion script update

## TODO(MAIR++)
We are currently awaiting the review of our extended work, [MAIR++](https://arxiv.org/abs/2408.06707). Stay tuned for its release, as the code for MAIR++ will be made available soon.

## Installation

This code has been verified to work with **CUDA 11.8**, but the specific versions of PyTorch and CUDA are not strictly required.

1. **Create and activate the conda environment**:
   ```bash
   conda create -n MAIR python=3.9
   conda activate MAIR
   ```

2. **Install PyTorch and CUDA**:
   ```bash
   conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

3. **Install additional Python packages**:
   ```bash
   pip install tqdm termcolor scikit-image imageio nvidia-ml-py3 h5py wandb opencv-python trimesh[easy] einops
   ```

## Object Insertion

To insert objects into the scene, modify the input and output directories in the provided scripts. We include example real-world data in [Examples](https://github.com/bring728/MAIR_Open/tree/master/Examples/input) from [IBRNet](https://github.com/googleinterns/IBRNet?tab=readme-ov-file).  

### 1. **Find Camera Poses**
   Run the following script to get the camera poses:
   ```bash
   python cds-mvsnet/img2poses.py
   ```

   > **Note**: You need to install **COLMAP** and **ImageMagick** to extract camera poses and resize images.

### 2. **Predict Multi-view Depth and Confidence Maps**
   Run the following script to predict depth and confidence maps:
   ```bash
   python cds-mvsnet/img2depth.py
   ```

   > We use [CDS-MVSNet](https://github.com/TruongKhang/cds-mvsnet). Special thanks to the authors for sharing the code.

   Download the [pretrained model](https://github.com/TruongKhang/cds-mvsnet/tree/main/pretrained/fine_tuning_on_blended) and place it in `cds-mvsnet/pretrained/`.
   

### 3. **Inverse rendering**
   Run the following script to predict geometry, material and 3D spatially-varying lighting volume.
   ```bash
   python test.py
   ```

   Download the [pretrained model](https://drive.google.com/drive/folders/1cwm8qZdOJJziTjxJOG1bgqN0Mt-f60-V?usp=sharing) and place it in `pretrained/MAIR/`.
   

### 4. **Object Insertion**
   Run the following script to insert object in scene.
   ```bash
   python ObjectInsertion/oi_main.py
   ```

---
