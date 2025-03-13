# MAIR & MAIR++: Multi-view Attention Inverse Rendering

This repository contains the official implementation of two related works:
- **"MAIR: Multi-view Attention Inverse Rendering with 3D Spatially-Varying Lighting Estimation"**  
  - Presented at **CVPR 2023**.
- **"MAIR++: Improving Multi-view Attention Inverse Rendering with Implicit Lighting Representation"**  
  - Accepted at **TPAMI**.

## Introduction

This repository provides the source code and pre-trained models for **MAIR** and its enhanced version **MAIR++**, which focus on multi-view inverse rendering with advanced 3D spatially-varying lighting estimation and implicit lighting representation. These methods are designed to improve the accuracy and efficiency of 3D scene reconstruction and rendering from multi-view inputs. The code is implemented in Python using PyTorch, making it accessible for researchers and developers.

- **MAIR (CVPR 2023)**: Introduces a novel attention-based approach for inverse rendering.
- **MAIR++ (TPAMI)**: Enhances MAIR with implicit lighting representation for better performance.

For more details, refer to the original papers:
- [MAIR Paper](https://arxiv.org/abs/2303.12368)
- [MAIR++ Paper](https://arxiv.org/abs/2408.06707)
- [Dataset](https://github.com/bring728/OpenRooms_FF)
---

## Timelog
- **2024-06-03**: MAIR initial commit (test code only)
- **2024-09-12**: Object insertion script update
- **2025-03-13**: MAIR++ update

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

   Download the [our pretrained model(mair, mair++)](https://drive.google.com/drive/folders/1cwm8qZdOJJziTjxJOG1bgqN0Mt-f60-V?usp=sharing) and place it in `pretrained/`.
   

### 4. **Object Insertion**
   Run the following script to insert object in scene.
   ```bash
   python ObjectInsertion/oi_main.py
   ```

---
