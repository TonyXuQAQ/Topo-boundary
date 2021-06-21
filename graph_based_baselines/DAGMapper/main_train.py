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
    network = env.network
    # explore the current image
    seq, seq_lens, tiff, mask, name, init_points, end_points = data
    tiff = tiff.to(env.args.device)
    # predict distance map
    global_feature_map = network.DAGMapperEncoder(tiff)
    distance_map = network.DAGMapperDTH(global_feature_map)
    cat_feature_map = torch.cat([global_feature_map,distance_map],dim=1)
    cat_feature_map = F.interpolate(cat_feature_map,size=(1000,1000), mode='bilinear', align_corners=True)
    distance_map = F.interpolate(distance_map,size=(1000,1000), mode='bilinear', align_corners=True)
    loss_segmentation = network.criterions['bce'](distance_map,torch.FloatTensor(mask).to(args.device).unsqueeze(0))
    network.bp(loss_segmentation,seg=True)
    if not args.pretrain:
        # 
        seq, seq_lens, mask, init_points, end_points = seq[0], seq_lens[0], mask[0], init_points[0], end_points[0]
        name = name[0][-14:-5]
        print('##################################################################' )
        print('DAGMapper Image: {}'.format(name))
        # sampling and training
        loss_coord_ave, loss_direction_ave,loss_state_ave = run_explore(env,seq, cat_feature_map, seq_lens, mask,name, init_points, end_points,i)
        # time usage
        time_now = time.time()
        time_used = time_now - env.time_start
        time_remained = time_used / env.training_step * (env.training_image_number - env.training_step)
        speed = time_used / env.training_step

        print('Time usage: S {}s/im || Ut {}h || Rt {}h'.format(round(speed,2)
                        ,round(time_used/3600,2),round(time_remained/3600,2)))

        print('Epoch: {}/{} | Image: {}/{} | Loss: coord {}/flag {}/seg {}'.format(
                env.epoch_counter,args.epochs,i,train_len,round(loss_coord_ave,3),round(loss_state_ave,3),round(loss_segmentation.item(),3)))
        env.network.writer.add_scalar('Train/grow_loss',loss_coord_ave,env.training_step)
        env.network.writer.add_scalar('Train/flag_loss',loss_state_ave,env.training_step)
        env.network.writer.add_scalar('Train/seg_loss',loss_segmentation.item(),env.training_step)
        # env.network.writer.add_scalar('Train/direction_loss',loss_direction_ave,env.training_step)
    else:
        print('##################################################################' )
        name = name[0][-14:-5]
        print('DAGMapper pretrain Image: {}'.format(name))
        print('Epoch: {}/{} | Image: {}/{} | Loss: seg {}'.format(
                env.epoch_counter,args.epochs,i,train_len,round(loss_segmentation.item(),3)))
        env.network.writer.add_scalar('Train/seg_loss',loss_segmentation.item(),env.training_step)
