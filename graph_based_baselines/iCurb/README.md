# iCurb 

iCurb is the previous work of our team, which aims to solve the problem of road curb detection in aerial images by iterative graph generation. Due to the similarity of road curbs and road boundaries, we provide both the training and inference code in this benchmark. Since the quality of some road-curb annotations in 
[nyc-planimetrics database](https://github.com/CityOfNewYork/nyc-planimetrics) is not as good as expected, we only show the process of how iCurb solves the problem of road boundary detection.

Different from past works, iCurb analyzes the detection problem from the perspective of imitation learning, and proposes a new pipeline as well as a DAgger-based training strategy. The comparison results demonstrate the superiority of our work.

## Related links
[iCurb paper](https://arxiv.org/abs/2103.17118) (RAL2021).

[iCurb project page](https://tonyxuqaq.github.io/iCurb/). A supplementary document is provided in the link.

If you are interested in this work, please cite our paper
```
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

## Experiment environment
Run 
```
./build_container.bash
``` 
to create the Docker container.

## Try with saved checkpoint
Run 
```
./run_eval.bash
```

Inference would be done with saved checkpoints. 

Binary maps of the generated graph are saved in ```./records/test/skeleton```; generated vertices are saved in ```./records/test/vertices_record```. After inference, a script is triggered to calculate the evaluation results of the inference.

## Train your own model

### Train FPN
For better training efficiency and convergence performance, we train the FPN and the agent network separately. A saved checkpoint of FPN is provided, which could be directly used for later tasks. FPN is trained with the pretrain set.
Run 
```
./run_train_seg.bash
```


### Train the agent network
Run 
```
./run_train.bash
```

All the training parameters could be modified in the ```config.yml```. You can choose the number of rounds of the restricted exploration (```r_exp```) and the free exploration (```f_exp```) in this file, but it is suggested to set ```r_exp=1``` and ```f_exp=1``` for the trade-off between efficiency and effectiveness. More rounds of free explorations will not effectively enhance the final performance due to more time consumption.

If your need the visualization of the trajectories generated during the training period, set ```visualization``` in ```config.yml``` to **True**, and check the visualizations in folder ```./records/train/vis```. 

### Tensorboard
Open another terminal and run 
```
docker exec -it topo_iCurb bash
``` 
Then run 
```
./run_tensorboard_seg.bash # if you train segmentation net
./run_tensorboard.bash # if you train agent net
``` 
The port number of above two bashes is **5008**. 

### Inference
Run 
```
./run_eval.bash
```

Inference would be done with saved checkpoints. 

Binary maps of the generated graph are saved in ```./records/test/skeleton```; generated vertices are saved in ```./records/test/vertices_record```. After inference, a script is triggered to calculate the evaluation results of the inference.

## Visualization in aerial images
If you want to project the generated road-boundary graph onto the aerial images for better visualization, run
```
python utils/visualize.py
```
Generated visualizations are saved in ```./records/test/final_vis```.





