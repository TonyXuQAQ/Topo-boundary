# r_exp=$(cat ./config.yml | grep "r_exp" | cut -d\   -f2)
# f_exp=$(cat ./config.yml | grep "f_exp" | cut -d\   -f2)
tensorboard --logdir ./records/tensorboard --port 6006 --bind_all
