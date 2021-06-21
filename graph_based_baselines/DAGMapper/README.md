# DAGMApper

## Related links
[DAGMapper paper](https://arxiv.org/abs/2012.12377) (ICCV2019).

[DAGMapper supplementary](https://nhoma.github.io/papers/dagmapper_iccv19_supp.pdf).

## Modifications
Currently, this work publishes neither its code nor its data. We re-implement this work. But due to the lack of details, it is hard to let the model converge with the original idea in this paper, which may be caused by different scenarios (they detect lane lines in simple high way scenarios with point cloud maps) or some tricks. 

To make this baseline work, we make the following modifications to their original methods:
* Remove the direction head, and directly predict the probability map of the next vertex, which is similar to VecRoad.
* Remove RNN.
* Use our own FPN network as the position head.


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

### Train FPN
Set ```pretrain``` in ```./config.yml``` to ```True```. Run 

```
./run_train.bash
```
Only the FPN and the distance transform head are trained. The network will be trained for 10 epochs. This step is for better convergence.


### Train agent net
Set ```pretrain``` in ```./config.yml``` to ```False```. Run 

```
./run_train.bash
```

All the training parameters could be modified in the ```config.yml```.


If your need the visualization of the trajectories generated during the training period set ```visualization``` in ```config.yml``` to **True**, and check the visualizations in folder ```./records/train/vis```. 

### Tensorboard
Open another terminal and run 
```
docker exec -it topo_DAGMapper bash
``` 
Then run 
```
./run_tensorboard.bash
``` 
The port number is **5006**.

### Inference
Run 

```
./run_eval.bash
```

Binary maps of the generated graph are saved in ```./records/test/skeleton```; generated vertices are saved in ```./records/test/vertices_record```. After inference, a script is triggered to calculate the evaluation results of the inference.

## Visualization in aerial images
If you want to project the generated road-boundary graph onto the aerial images for better visualization, run
```
python utils/visualize.py
```
Generated visualizations are saved in ```./records/test/final_vis```.




Some examples are shown below. Cyan lines are the ground truth. In the predicted road-boundary graph, yellow nodes are vertices and orange lines are edges.


<img src=./img/gt_000167_41.png width="25%" height="25%"><img src=./img/gt_000197_22.png width="25%" height="25%"><img src=./img/gt_000160_01.png width="25%" height="25%"><img src=./img/gt_000217_43.png width="25%" height="25%">

<img src=./img/000167_41.png width="25%" height="25%"><img src=./img/000197_22.png width="25%" height="25%"><img src=./img/000160_01.png width="25%" height="25%"><img src=./img/000217_43.png width="25%" height="25%">
