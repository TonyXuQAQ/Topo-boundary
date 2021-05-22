<head>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.css" integrity="sha384-yFRtMMDnQtDRO8rLpMIKrtPCD5jdktao2TV19YiZYWMDkUR5GQZR/NOVTdquEx1j" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.js" integrity="sha384-9Nhn55MVVN0/4OFx7EE5kpFBPsEMZxKTCnA+4fqDmg12eCTqGi6+BB2LjY8brQxJ" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
<style>
.katex-display > .katex {
  display: inline-block;
  white-space: nowrap;
  max-width: 100%;
  overflow-x: scroll;
  text-align: initial;
}
.katex {
  font: normal 1.21em KaTeX_Main, Times New Roman, serif;
  line-height: 1.2;
  white-space: normal;
  text-indent: 0;
}
</style>
</head>

## Welcome to Topo-boundary project page

This is the project page of paper "Topo-boundary: A Benchmark Dataset on Topological Road-boundary Detection Using Aerial Images for Autonomous Driving" ([Arxiv](https://arxiv.org/abs/2103.17119))

Authors: Zhenhua Xu, Yuxiang Sun, Ming Liu

## Abstract 
Road-boundary detection is important for autonomous driving. For example, it can be used to constrain vehicles running on road areas, which ensures driving safety. Compared with online road-boundary detection using on-vehicle cameras/Lidars, offline detection using aerial images could alleviate the severe occlusion issue. Moreover, the offline detection results can be directly used to annotate high-definition (HD) maps. In recent years, deep-learning technologies have been used in offline detection. But there is still lacking a publicly available dataset for this task, which hinders the research progress in this area. So in this paper, we propose a new benchmark dataset, named _Topo-boundary_, for offline topological road-boundary detection. The dataset contains 25,295 $$1000\times1000$$-sized 4-channel aerial images. Each image is provided with 8 training labels for different sub-tasks. We also design a new entropy-based metric for connectivity evaluation, which could better handle noises or outliers. We implement and evaluate 3 segmentation-based baselines and 5 graph-based baselines using the dataset. We also propose a new imitation-learning-based baseline which is enhanced from our previous work. The superiority of our enhancement is demonstrated from the comparison.

### Dataset description

This dataset is for topological road-boundary detection in aerial images for autonomous driving purposes. We provide 25,295 high-resolutin aerial images and each image have multiple labels for different learining tasks. We also provide 9 baseline models.

This dataset is based on the GIS database [NYC Planimetric Database](https://github.com/CityOfNewYork/nyc-planimetrics/blob/master/Capture_Rules.md).

### Supplementary
[Supplementary document](https://github.com/TonyXuQAQ/Topo-boundary/blob/master/dataset/topoboundary_supplementary.pdf) provides more details about the evaluation metrics and data structure.

### Code and data
Please check our [github repo](https://github.com/TonyXuQAQ/Topo-boundary) and follow the steps.


### Contact
For any questions, please send email to zxubg at connect dot ust dot hk.

### Citation
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
