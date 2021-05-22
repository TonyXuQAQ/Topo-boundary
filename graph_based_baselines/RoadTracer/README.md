# RoadTracer 

RoadTracer is believed to be the first method that iteratively generates the graph of the road network, and most after works follow a similar core idea with it. 

## Related links
[RoadTracer paper](https://arxiv.org/abs/1802.03680) (CVPR2018).

[RoadTracer supplementary](https://roadmaps.csail.mit.edu/roadtracer/supplementary.pdf).

[Official implementation of RoadTracer](https://github.com/mitroadmaps/roadtracer). But it is implemented by *TensorFlow 1.0*. Due to the difference between our tasks, we make necessary modifications to it while the core idea is not altered.


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
The inference would be done with the saved checkpoint. Binary maps of the generated graph are saved in ```./records/test/skeleton```; generated vertices are saved in ```./records/test/vertices_record```. After inference, a script is triggered to calculate the evaluation results of the inference.

## Train your own model

### Pretrain by teacher-forcing
We provide a pretraining checkpoint, which is trained by teacher-forcing for 1 epoch. It takes a long time to train it. If you want to pretrain your own checkpoint, set ```pretrain``` in ```config.yml``` to **True**, then run 
```
./run_train.bash
```
RoadTracer would be trained by teacher-forcing for 1 epoch. The teacher-forcing takes a long time to train since the generated boundary graphs are usually very long.

With the pretraining checkpoint, we train RoadTracer freely (without any corrections, use the predicted vertex to update the graph) for 3 epochs by default. 

### Training
Set ```pretrain``` in ```config.yml``` to **False**. Run 
```
./run_train.bash
```
All the training parameters could be modified in the ```config.yml```.

If your need the visualization of the trajectories generated during the training period set ```visualization``` in ```config.yml``` to **True**, and check the visualizations in folder ```./records/train/vis```. 



### Tensorboard
Open another terminal and run 
```
docker exec -it topo_roadTracer bash
``` 
Then run 
```
./run_tensorboard.bash
``` 
The port number is **5004**.

### Inference
Run ```
./run_eval.bash
```, the inference would be done with the saved checkpoint. 

Binary maps of the generated graph are saved in ```./records/test/skeleton```; generated vertices are saved in ```./records/test/vertices_record```. After inference, a script is triggered to calculate the evaluation results of the inference.

## Visualization in aerial images
If you want to project the generated road-boundary graph onto the aerial images for better visualization, run
```
python utils/visualize.py
```
Generated visualizations are saved in ```./records/test/final_vis```.

Some examples are shown below. Cyan lines are the ground truth and green lines are the prediction.

<img src=./img/gt_000180_31.png width="25%" height="25%"><img src=./img/gt_000190_21.png width="25%" height="25%"><img src=./img/gt_000230_21.png width="25%" height="25%"><img src=./img/gt_005165_43.png width="25%" height="25%">

<img src=./img/000180_31.png width="25%" height="25%"><img src=./img/000190_21.png width="25%" height="25%"><img src=./img/000230_21.png width="25%" height="25%"><img src=./img/005165_43.png width="25%" height="25%">