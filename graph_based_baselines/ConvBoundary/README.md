# ConvBoundary

## Related links
[ConvBoundary paper](https://nhoma.github.io/papers/road_cvpr19.pdf) (CVPR2019).

[ConvBoundary supplementary](https://nhoma.github.io/papers/road_cvpr19_supp.pdf).

## Modifications
Currently, this work publishes neither its code nor its data. We re-implement this work. But due to the lack of details, it is hard to let the model converge with the original idea in this paper, which may be caused by different scenarios (they detect road boundaries in simple high way scenarios) or some tricks. 

To make this baseline work, we make the following modifications to their original methods:
* Instead of calculating the direction from the predicted direction map, we directly predict the probability map of the next vertex, because the precited direction map may have incorrectly predicted pixels, especially near the boundary. Then the pipeline is similar to VecRoad.


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

### Generate direction map
Follow the step in ```../../dataset/scripts``` to generate labels for this baseline.

### Train segmentation FPN
Run 

```
./run_train_seg.bash
```
The network will be trained for 15 epochs. The output of the segmentation FPN is a 3 channel image. The first channel is the predicted inverse distance map, and the rest two channels are the predicted direction map.


### Train agent net
Run 
```
./run_train.bash
```

All the training parameters could be modified in the ```config.yml```.


If your need the visualization of the trajectories generated during the training period set ```visualization``` in ```config.yml``` to **True**, and check the visualizations in folder ```./records/train/vis```. 


### Tensorboard
Open another terminal and run 
```
docker exec -it topo_convBoundary bash
``` 
Then run 
```
./run_tensorboard_seg.bash # if you train segmentation net
./run_tensorboard.bash # if you train agent net
``` 
The port number of above two bashes are **5007**.

### Inference
Run 

```
./run_eval.bash
```

Binary maps of the generated graph are saved in ```./records/test/skeleton```; generated vertices are saved in ```./records/test/vertices_record```. After inference, a script is triggered to calculate the evaluation results of the inference.

## Efficiency
Since ConvBoundary does not have a head to stop the agent as other graph-based baseline models, the stop action is only triggered when the agent reaches the image edge. If the prediction is not accurate enough, the agent may fail to stop correctly, causing the agent to hang around and greatly degrade the efficiency.

## Visualization in aerial images
If you want to project the generated road-boundary graph onto the aerial images for better visualization, run
```
python utils/visualize.py
```
Generated visualizations are saved in ```./records/test/final_vis```.


Some examples are shown below. Cyan lines are the ground truth. In the predicted road-boundary graph, yellow nodes are vertices and orange lines are edges.


<img src=./img/gt_000167_41.png width="25%" height="25%"><img src=./img/gt_000197_22.png width="25%" height="25%"><img src=./img/gt_000160_01.png width="25%" height="25%"><img src=./img/gt_000217_43.png width="25%" height="25%">

<img src=./img/000167_41.png width="25%" height="25%"><img src=./img/000197_22.png width="25%" height="25%"><img src=./img/000160_01.png width="25%" height="25%"><img src=./img/000217_43.png width="25%" height="25%">


