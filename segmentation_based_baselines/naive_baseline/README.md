# Naive baseline

The naive baseline first predicts the semantic segmentation map of road boundaries and then runs post-processing steps to extract the graph of road boundaries by binarization, skeletonization, thresholding, etc. UNet is utilized as the semantic segmentation network.

## Related links
The implementation of the naive baseline is based on the open-sourced repo: [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet). 

## Experiment environment
Run 
```
./build_container.bash
``` 
to create the Docker container.

## Try with saved checkpoint
After downloading the saved checkpoint, select checkpoint by ```load_checkpoint``` in ```config.yml```. Run 
```
./run_eval.bash
```
After inference, a script is triggered to calculate the evaluation results of the inference.

## Train your own model
### Training
Since no pretraining is needed, the naive baseline is trained with data in both the training set and the pretraining set.

Run 
```
./run_train.bash
```

All the training parameters could be modified in the ```config.yml```. The generated checkpoints are saved in ```./checkpoints```.

### Tensorboard
Open another terminal and run 
```
docker exec -it topo_naive bash
``` 
Then run 
```
./run_tensorboard.bash
``` 
The port number is **5001**. 

### Inference
Select used checkpoint by ```load_checkpoint``` in the config file. Run 
```
./run_eval.bash
```

The inference would be done with saved checkpoints. Then post-processing as well as evaluation will be triggered. Obtained road-boundary graphs are saved in ```./records/test/skeleton```. It takes around 1 hour for inference and evaluation.

## Visualization in aerial images
If you want to visualize the detected road boundaries in aerial images, after the inference, run 
```
python ./utils/visualize.py
```
Road boundaries will be visualized in aerial images (only have R,G,B channels for better visualization), and images are saved in ```./records/test/final_vis```.

Some examples are shown below. Cyan lines are the ground truth and green lines are the prediction.

<img src=./img/gt_000180_31.png width="25%" height="25%"><img src=./img/gt_000190_21.png width="25%" height="25%"><img src=./img/gt_000230_21.png width="25%" height="25%"><img src=./img/gt_005165_43.png width="25%" height="25%">

<img src=./img/000180_31.png width="25%" height="25%"><img src=./img/000190_21.png width="25%" height="25%"><img src=./img/000230_21.png width="25%" height="25%"><img src=./img/005165_43.png width="25%" height="25%">






