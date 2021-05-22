# OrientationRefine

OrientationRefine is a strong segmentation-based baseline. It first predicts the segmentation map with the extra supervision of the orientation map, which greatly improves the topology correctness of the segmentation map. Then, after obtaining the segmentation map, this work conducts an iterative refinement process called "connectivity refinement" to correct false-predicted pixels of the segmentation map. Connectivity refinement can effectively bridge disconnections in the segmentation map. 

## Related links
[OrientationRefine paper](https://anilbatra2185.github.io/papers/RoadConnectivityCVPR2019.pdf) (CVPR2019).

[OrientationRefine supplementary](https://anilbatra2185.github.io/papers/RoadConnectivity_CVPR_Supplementary.pdf).

[Official implementation of OrientationRefine](https://github.com/anilbatra2185/road_connectivity). Due to the difference between our works, necessary modifications are made to the original work. The core idea is not altered.

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
The generated road-boundary graphs are saved in ```./records/refine/test/skeleton```. After inference, a script is triggered to calculate the evaluation results of the inference.

## Train your own model
### Train the segmentation network
Run 
```
./run_train_seg.bash
```
Note we only train the segmentation network with images in the pretrain set, since the training images are for refinement network training. The generated checkpoints are saved in ```./checkpoints```.

<!-- ## Train the refinement network
Training the refinement network requires corrupted ground-truth masks and predicted semantic segmentation maps, thus before start training, we should first generate the required images. -->

### Generate corrupted ground-truth masks
Run 
```
python ./utils/generate_corrupted_mask.py
```
Then corrupted masks of the training, validation, and testing set are generated into ```./records/corrupted_mask```. The corrupted mask will be used to pretrain the refinement network.

### Generate segmentation maps
Generate segmentation maps with the trained segmentation network in the previous section. Run 
```
./run_eval_seg.bash
``` 
The predicted segmentation map of the training, validation and testing set are saved into ```./records/segmentation``` 

### Pretrain the refinement network
Set ```refine.pretrain``` to ```True```. Run 
```
./run_train_refine.bash
```
The generated pretrain checkpoint will be saved in ```./checkpoints```.

### Train the refinement network
Set ```refine.pretrain``` to ```False```.Run 
```
./run_train_refine.bash
```
The generated pretrain checkpoint will be saved in ```./checkpoints```.

### Tensorboard
Open another terminal and run 
```
docker exec -it topo_ImproveConnectivity bash
``` 
Then run 
```
./run_tensorboard_seg.bash # if you train segmentation net
./run_tensorboard_refine.bash # if you train refinement net
``` 
The port number of above two bashes is **5003**. 

### Inference with the refinement network
Select used checkpoint by ```refine.load_checkpoint``` in the config file. Run 
```
./run_eval_refine.bash
```
The generated road-boundary graphs are saved in ```./records/refine/test/skeleton```.

## Visualization in aerial images
If you want to visualize the detected road boundaries in aerial images, after the inference, run 
```
python ./utils/visualize.py
```
Road boundaries will be visualized in aerial images (only have R,G,B channels for better visualization), and images are saved in ```./records/test/final_vis```.

Some examples are shown below. Cyan lines are the ground truth and green lines are the prediction.

<img src=./img/gt_000180_31.png width="25%" height="25%"><img src=./img/gt_000190_21.png width="25%" height="25%"><img src=./img/gt_000230_21.png width="25%" height="25%"><img src=./img/gt_005165_43.png width="25%" height="25%">

<img src=./img/000180_31.png width="25%" height="25%"><img src=./img/000190_21.png width="25%" height="25%"><img src=./img/000230_21.png width="25%" height="25%"><img src=./img/005165_43.png width="25%" height="25%">





