# DeepRoadMapper 

DeepRoadMapper is one early work of the segmentation-based method for road-network detection. It first predicts the segmentation map of the road network, then extracts the topology structure (graph) of it by skeletonization and some other processing. Some candidate connection hypotheses are generated to bridge disconnections existing in the obtained graph. An extra network is trained to reason the proposed connections, to verify whether they should be added to the obtained graph. In this way, error topology could be corrected to some extent.

## Related links
[DeepRoadMapper paper](http://www.cs.toronto.edu/~wenjie/papers/iccv17/mattyus_etal_iccv17.pdf).

There is no available official repo. But DeepRoadMapper is implemented by RoadTracer as a baseline [non-official implementation](https://github.com/mitroadmaps/roadtracer/tree/master/deeproadmapper). The implementation is based on *Tensorflow 1.0*.

## Experiment environment
Run 
```
./build_container.bash
``` 
to create the Docker container.

## Try with saved checkpoint
Run 
```
./run_eval_refine.bash
```
The generated road-boundary graphs are saved in ```./records/reason/test/vis```. After inference, a script is triggered to calculate the evaluation results of the inference.


## Train your own model
### Train the segmentation network
Run 
```
./run_train_seg.bash
```
Note we only train the segmentation network with the pretrain set, and the training set is for reasoning network training. The generated checkpoints are saved in ```./checkpoints```. The segmentation network is UNet.

### Generate candidate connections
Run 
```
./generate_candidate.bash
```
Then samples to train the reasoning network are saved in ```./records```. The sample generation process may take a long time.

### Train the reasoning network
The network is trained to decide whether a proposed connection should be accepted. Run
```
./run_train_reason.bash
```
The checkpoints are saved in ```./checkpoints```.

### Tensorboard
Open another terminal and run 
```
docker exec -it topo_DeepRoadMapper bash
``` 
Then run 
```
./run_tensorboard_seg.bash # if you train segmentation net
./run_tensorboard_reason.bash # if you train reasoning net
``` 
The port number of above two bashes is **5002**. 

### Inference with the reasoning network
Select used checkpoint by ```reason_net.load_checkpoint``` in the config file. Run 
```
./run_eval_reason.bash
```

The inference would be done with saved checkpoints. The generated road-boundary graphs with candidate correction are saved in ```./records/reason/test/vis```.

## Visualization in aerial images
If you want to visualize the detected road boundaries in aerial images, after the inference, run 
```
python ./utils/visualize.py
```
Then road boundaries will be visualized in aerial images (only have R,G,B channels for better visualization), and images are saved in ```./records/test/final_vis```.

Some examples are shown below. Cyan lines are the ground truth and green lines are the prediction.

<img src=./img/gt_000180_31.png width="25%" height="25%"><img src=./img/gt_000190_21.png width="25%" height="25%"><img src=./img/gt_000230_21.png width="25%" height="25%"><img src=./img/gt_005165_43.png width="25%" height="25%">

<img src=./img/000180_31.png width="25%" height="25%"><img src=./img/000190_21.png width="25%" height="25%"><img src=./img/000230_21.png width="25%" height="25%"><img src=./img/005165_43.png width="25%" height="25%">





