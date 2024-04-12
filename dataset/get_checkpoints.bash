gdown https://drive.google.com/uc?id=1ijgnesWfvx5SfcuD68T5s8Lbr4ZclZ0R

unzip -d ./ pretrain_checkpoints.zip
rm -rf y./pretrain_checkpoints.zip

mkdir -p ../segmentation_based_baselines/naive_baseline/checkpoints
cp ./pretrain_checkpoints/naive_baseline/* ../segmentation_based_baselines/naive_baseline/checkpoints/

mkdir -p ../segmentation_based_baselines/OrientationRefine/checkpoints
cp ./pretrain_checkpoints/OrientationRefine/* ../segmentation_based_baselines/OrientationRefine/checkpoints/

mkdir -p ../segmentation_based_baselines/DeepRoadMapper/checkpoints
cp ./pretrain_checkpoints/DeepRoadMapper/* ../segmentation_based_baselines/DeepRoadMapper/checkpoints/

mkdir -p ../graph_based_baselines/RoadTracer/checkpoints
cp ./pretrain_checkpoints/RoadTracer/* ../graph_based_baselines/RoadTracer/checkpoints/

mkdir -p ../graph_based_baselines/VecRoad/checkpoints
cp ./pretrain_checkpoints/VecRoad/* ../graph_based_baselines/VecRoad/checkpoints/

mkdir -p ../graph_based_baselines/iCurb/checkpoints
cp ./pretrain_checkpoints/iCurb/* ../graph_based_baselines/iCurb/checkpoints/

mkdir -p ../graph_based_baselines/DAGMapper/checkpoints
cp ./pretrain_checkpoints/DAGMapper/* ../graph_based_baselines/DAGMapper/checkpoints/

mkdir -p ../graph_based_baselines/ConvBoundary/checkpoints
cp ./pretrain_checkpoints/ConvBoundary/* ../graph_based_baselines/ConvBoundary/checkpoints/

mkdir -p ../graph_based_baselines/Enhanced-iCurb/checkpoints
cp ./pretrain_checkpoints/Enhanced-iCurb/* ../graph_based_baselines/Enhanced-iCurb/checkpoints/

mkdir -p ../graph_based_baselines/init_vertex/checkpoints
cp ./pretrain_checkpoints/initial_vertex/* ../graph_based_baselines/init_vertex/checkpoints/
