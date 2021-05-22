import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from main_sampler import  run_explore
import torch.nn.functional as F
from scipy.spatial import cKDTree

def run_train(env,data,i):
    
    env.training_step += 1
    train_len = env.network.train_len()
    args = env.args
    
    # explore the current image
    seq, seq_lens, tiff, mask, name, init_points, end_points = data
    tiff = tiff.to(env.args.device)
    seq, seq_lens, mask, init_points, end_points = seq[0], seq_lens[0], mask[0], init_points[0], end_points[0]
    name = name[0][-14:-5]
    print('##################################################################' )
    print('VecRoad Image: {}'.format(name))
    # sampling and training
    loss_coord_ave, loss_flag_ave, loss_seg_ave = run_explore(env,seq,tiff, seq_lens, mask,name, init_points, end_points,i)
    
    # time usage
    time_now = time.time()
    time_used = time_now - env.time_start
    time_remained = time_used / env.training_step * (env.training_image_number - env.training_step)
    speed = time_used / env.training_step

    print('Time usage: S {}s/im || Ut {}h || Rt {}h'.format(round(speed,2)
                ,round(time_used/3600,2),round(time_remained/3600,2)))

    print('Epoch: {}/{} | Image: {}/{} | Loss: coord {}/seg {}'.format(
            env.epoch_counter,args.epochs,i,train_len,round(loss_coord_ave,3),round(loss_seg_ave,3)))
    env.network.writer.add_scalar('Train/grow_loss',loss_coord_ave,env.training_step)
    env.network.writer.add_scalar('Train/seg_loss',loss_seg_ave,env.training_step)