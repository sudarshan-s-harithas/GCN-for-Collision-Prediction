# Semantic Graph Convolutional Networks for 3D Human Pose Regression (CVPR 2019)

This repository holds the Pytorch implementation of [Semantic Graph Convolutional Networks for 3D Human Pose Regression](https://arxiv.org/abs/1904.03345) by Long Zhao, Xi Peng, Yu Tian, Mubbasir Kapadia and Dimitris N. Metaxas. If you find our code useful in your research, please consider citing:

```
@inproceedings{zhaoCVPR19semantic,
  author    = {Zhao, Long and Peng, Xi and Tian, Yu and Kapadia, Mubbasir and Metaxas, Dimitris N.},
  title     = {Semantic Graph Convolutional Networks for 3D Human Pose Regression},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {3425--3435},
  year      = {2019}
}
```

<p align="center"><img src="example.gif" width="70%" alt="" /></p>

## Introduction

We propose Semantic Graph Convolutional Networks (SemGCN), a novel graph convolutional network architecture that operates on regression tasks with graph-structured data. The code of training and evaluating our approach for 3D human pose estimation on the [Human3.6M Dataset](http://vision.imar.ro/human3.6m/) is provided in this repository.

In this repository, 3D human poses are predicted according to **Configuration #1** in [our paper](https://arxiv.org/pdf/1904.03345.pdf): we only leverage 2D joints of the human pose as inputs. We utilize the method described in Pavllo et al. [2] to normalize 2D and 3D poses in the dataset, which is different from the original implementation in our paper. To be specific, 2D poses are scaled according to the image resolution and normalized to [-1, 1]; 3D poses are aligned with respect to the root joint. Please refer to the corresponding part in Pavllo et al. [2] for more details. We predict 16 joints (as the skeleton in Martinez et al. [1] without the 'Neck/Nose' joint). We also provide the results of Martinez et al. [1] in the same setting for comparison.


### References

[1] Martinez et al. [A simple yet effective baseline for 3d human pose estimation](https://arxiv.org/pdf/1705.03098.pdf). ICCV 2017.

[2] Pavllo et al. [3D human pose estimation in video with temporal convolutions and semi-supervised training](https://arxiv.org/pdf/1811.11742.pdf). CVPR 2019.

## Quick start

This repository is build upon Python v2.7 and Pytorch v1.1.0 on Ubuntu 16.04. NVIDIA GPUs are needed to train and test. See [`requirements.txt`](requirements.txt) for other dependencies. We recommend installing Python v2.7 from [Anaconda](https://www.anaconda.com/), and installing Pytorch (>= 1.1.0) following guide on the [official instructions](https://pytorch.org/) according to your specific CUDA version. Then you can install dependencies with the following commands.

```
git clone git@github.com:garyzhao/SemGCN.git
cd SemGCN
pip install -r requirements.txt
```

### Dataset setup
You can find the instructions for setting up the Human3.6M and results of 2D detections in [`data/README.md`](data/README.md). The code for data preparation is borrowed from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D).

### TO run the visulization  

python viz.py --architecture gcn --non_local --evaluate <CKPT_Path> --viz_subject S11 --viz_action Walking --viz_camera 0 --viz_output output.gif --viz_size 3 --viz_downsample 2 --viz_limit 60


### TO run the Inference


python main_gcn.py --evaluate <CKPT_Path>


 
 



