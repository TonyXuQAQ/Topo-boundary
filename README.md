# Topo-boundary
This is the official github repo of paper [Topo-boundary: A Benchmark Dataset on Topological Road-boundary Detection Using Aerial Images for Autonomous Driving](https://arxiv.org/abs/2103.17119). 

[Project page](https://tonyxuqaq.github.io/Topo-boundary/).

Topo-boundary is a publicly available benchmark dataset for topological road-boundary detection in aerial images. With an aerial image as the input, the evaluated method should predict the topological structure of road boundaries in the form of a graph.

This dataset is based on [NYC Planimetric Database](https://github.com/CityOfNewYork/nyc-planimetrics/blob/master/Capture_Rules.md). Topo-boundary consists of 25,297 4-channel aerial images, and each aerial image has eight labels for different deep-learning tasks. More details about the dataset structure can be found in our paper. Follow the steps in the ```./dataset``` to prepare the dataset.

We also provide the implementation code (including training and inference) based on *PyTorch* of 9 methods. Go to the Implementation section for details.

<img src=./dataset/pic.png width="100%" height="100%">

## Update
* May/27/2021 The code of **Enhanced-iCurb** has been cleaned and released.
* May/22/2021 *Topo_boundary* is released. More time is needed to prepare **ConvBoundary**, **DAGMapper** and **Enhanced-iCurb**, thus currently these models are not open-sourced.

## Platform information
Hardware info
```
GPU: one RTX3090 and one GTX1080Ti
CPU: i7-8700K
RAM: 32G
SSD: 256G + 1T
```

Software info
```
Ubuntu 18.04
CUDA 11.2
Docker 20.10.1
Nvidia-driver 460.73.01
```
Make sure you have Docker installed.

## Environment and Docker
Docker is used to set up the environment. If you are not familiar with Docker, refer to [install Docker](https://docs.docker.com/engine/install/ubuntu/) and [Docker beginner tutorial](https://docker-curriculum.com/) for more information.

To build the Docker image, run:
```
# go to the directory
cd ./docker
# optional
chmod +x ./build_image.sh
# build the Docker image
./build_image.sh
```
The image is based on the Docker image ```pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime``` provided on Docker hub. You may use other base image at [Pytorch_Docker_hub](https://hub.docker.com/r/pytorch/pytorch/tags?page=1&ordering=last_updated) based on your own preference.

To build Docker containers, check ```./build_container.bash``` under the directory of each baseline.
 
## File structure
```
Topo-Boundary
|
├── dataset
|   ├── data_split.json
|   ├── config_dir.yml
|   ├── get_data.bash
|   ├── get_checkpoints.bash
│   ├── cropped_tiff
│   ├── labels
|   ├── pretrain_checkpoints
│   └── scripts
|   
├── docker 
|
├── graph_based_baselines
|   ├── ConvBoundary
|   ├── DAGMApper
|   ├── Enhanced-iCurb
|   ├── iCurb
|   ├── RoadTracer
|   └── VecRoad 
|
├── segmentation_based_baselines
|   ├── DeepRoadMapper
|   ├── OrientationRefine
|   └── naive_baseline
|
```

## Data and pretrain checkpoints preparation
Follow the steps in ```./dataset``` to prepare the dataset and checkpoints trained by us.

## Implementations 
We provide the implementation code of 9 methods, including 3 segmentation-based baseline models, 5 graph-based baseline models, and an improved method based on our previous work [iCurb](https://tonyxuqaq.github.io/iCurb/). All methods are implemented with *PyTorch* by ourselves. 

Note that the evaluation results of baselines may change after some modifications being made.

## Evaluation metrics
We evaluate our implementations by 3 relaxed-pixel-level metrics, the self-defined Entropy Connectivity Metric (ECM), naive connectivity metric (proposed in ConvBoundary) and Average Path Length Similarity ([APLS](https://medium.com/the-downlinq/spacenet-road-detection-and-routing-challenge-part-i-d4f59d55bfce)). For more details, refer to the [supplementary document](./dataset/topoboundary_supplementary.pdf).

## Related topics
Other research topics about line-shaped object detection could be inspiring to our task. **Line-shaped object** indicts target objects that have long but thin shapes, and the topology correctness of them also matters a lot. They usually have an irregular shape. E.g., road-network detection, road-lane detection, road-curb detection, line-segment detection, etc. The method to detect one line-shaped object could be adapted to another category without much modification.

## To do
- [ ] Acceleration
- [ ] Fix bugs

## Contact
For any questions, please send email to zxubg at connect dot ust dot hk.

## Citation
```
@article{xu2021topo,
  title={Topo-boundary: A Benchmark Dataset on Topological Road-boundary Detection Using Aerial Images for Autonomous Driving},
  author={Xu, Zhenhua and Sun, Yuxiang and Liu, Ming},
  journal={arXiv preprint arXiv:2103.17119},
  year={2021}
}

@article{xu2021icurb,
  title={iCurb: Imitation Learning-Based Detection of Road Curbs Using Aerial Images for Autonomous Driving},
  author={Xu, Zhenhua and Sun, Yuxiang and Liu, Ming},
  journal={IEEE Robotics and Automation Letters},
  volume={6},
  number={2},
  pages={1097--1104},
  year={2021},
  publisher={IEEE}
}
```
<!-- ## License -->

