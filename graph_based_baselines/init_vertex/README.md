# New feature: predict initial vertices

In the past, for fair comparison, we directly use ground-truth initial vertices to start graph-based methods. Now we release the code for initial vertices prediction code. The predicted initial vertices could be used to test Enhanced-iCurb. For other models, they do not support predicted initial vertices, yet. 

We first predict the endpoint map by FPN, then conduct algorithms to extract initial vertex coordinates.

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

Then the endpoint map is predicted and initial vertices are extracted into ```./records/endpoint/vertices```. Then the vertices are copied to ```./dataset/init_vertex``` to be used by other baseline models.

## Train your own model

### Train FPN
You can train your own FPN model for endpoint map prediction.
Run 
```
./run_train_seg.bash
```

### Tensorboard
Open another terminal and run 
```
docker exec -it topo_iCurb bash
``` 
Then run 
```
./run_tensorboard.bash
``` 
The port number of above two bashes is **5012**. 

### Initial vertex extraction
Run 
```
./run_eval.bash
```

## Evaluation
After we obtain the initial vertices, we directly use the Enhanced-iCurb checkpoints to run the test. The results are inferior to its results with GT initial vertices. One possible explanation could be that we did not fine tune and retrain models. The results could be further improved by using more powerful initial vertex segmentation backbone or add human users in the loop.

<img src=./utils/results.png width="50%" height="50%">