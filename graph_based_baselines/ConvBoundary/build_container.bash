#!/bin/bash

CMD=$*

if [ -z "$CMD"];
then 
	CMD=/bin/bash
fi

home_dir=$(cat ../../dataset/config_dir.yml | grep "docker_home_dir" | cut -d\   -f2)
sub_dir=$(cat ./config.yml | grep "docker_sub_dir" | cut -d\   -f2)
dataset_dir=$(cat ../../dataset/config_dir.yml | grep "docker_dataset_dir" | cut -d\   -f2)
container_name=$(cat ./config.yml | grep "docker_container_name" | cut -d\   -f2)
image_name=$(cat ../../dataset/config_dir.yml | grep "docker_image_name" | cut -d\   -f2)
port_number=$(cat ./config.yml | grep "docker_port_number" | cut -d\   -f2)

docker run \
	-v $home_dir$sub_dir:/topoBoundary\
	-v $dataset_dir:/topoBoundary/dataset\
	--name=$container_name\
	--gpus all\
	-p $port_number:6006\
	--rm -it $image_name $CMD

