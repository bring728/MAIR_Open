This repository implements the paper "MAIR: Multi-view Attention Inverse Rendering with 3D Spatially-Varying Lighting Estimation" in CVPR 2023. Currently, we only include test code.


## installation

Verified in cuda 11.8, but torch and cuda versions do not matter.

```
conda create -n MAIR python=3.9
conda activate MAIR
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install tqdm termcolor scikit-image imageio nvidia-ml-py3 h5py wandb opencv-python trimesh[easy] einops
```

## Pretrained model
[Google drive](https://drive.google.com/drive/folders/1cwm8qZdOJJziTjxJOG1bgqN0Mt-f60-V?usp=sharing)
