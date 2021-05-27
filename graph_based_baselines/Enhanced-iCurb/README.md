# Enhanced-iCurb

Enhanced-iCurb is an improved version of our previous work iCurb. It proposes a more reasonable and effective algorithm to generate expert demonstrations to train the agent network. The above modification improves the final evaluation results and the quality of the qualitative visualizations.

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
### Training

Enhanced-iCurb has exactly the same training process as iCurb. First, run 
```
./run_train.bash
```

All the training parameters could be modified in the ```config.yml```.


If your need the visualization of the trajectories generated during the training period set ```visualization``` in ```config.yml``` to **True**, and check the visualizations in folder ```./records/train/vis```. 

### Tensorboard
Open another terminal and run 
```
docker exec -it topo_enhanced_iCurb bash
``` 
Then run 
```
./run_tensorboard.bash
``` 
The port number is **5009**.

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

Some examples are shown below. Cyan lines are the ground truth. In the predicted road-boundary graph, yellow nodes are vertices and orange lines are edges. We can see good results, and the agent can produce more accurate corners (by using smaller step size).


<img src=./img/gt_000167_41.png width="25%" height="25%"><img src=./img/gt_000197_22.png width="25%" height="25%"><img src=./img/gt_000160_01.png width="25%" height="25%"><img src=./img/gt_000217_43.png width="25%" height="25%">

<img src=./img/ei_000167_41.png width="25%" height="25%"><img src=./img/ei_000197_22.png width="25%" height="25%"><img src=./img/ei_000160_01.png width="25%" height="25%"><img src=./img/ei_000217_43.png width="25%" height="25%">