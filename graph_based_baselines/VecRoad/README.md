# VecRoad 

## Related links
[VecRoad paper](http://mftp.mmcheng.net/Papers/20CvprVecRoad.pdf) (CVPR2020).

[VecRoad supplementary](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Tan_VecRoad_Point-Based_Iterative_CVPR_2020_supplemental.pdf).

[Official implementation of VecRoad](https://github.com/tansor/VecRoad). But it only releases the inference script so far.

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
We provide a pretraining checkpoint, which is trained by teacher-forcing with the first 1000 images in the 1st epoch. It takes a long time to train it. If you want to pretrain your own checkpoint, set ```teacher_forcing_number``` to a proper value (1000 is suggested) and run
```
./run_train.bash
```
```teacher_forcing_number``` controls the number of images used to train VecRoad by teacher-forcing. 

Because of the huge resource consumption and low efficiency, with the pretraining checkpoint, we only train VecRoad freely (without any corrections, use the predicted vertex to update the graph) for 2 epochs by default.

### Training
Set ```teacher_forcing_number``` in ```config.yml``` to ```-1```.Run 

```
./run_train.bash
```

If your need the visualization of the trajectories generated during the training period set ```visualization``` in ```config.yml``` to **True**, and check the visualizations in folder ```./records/train/vis```. 




### Tensorboard
Open another terminal and run 
```
docker exec -it topo_vecRoad bash
``` 
Then run 
```
./run_tensorboard.bash
``` 
The port number is **5005**.

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




