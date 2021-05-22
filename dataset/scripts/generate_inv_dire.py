import argparse
import logging
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from skimage import measure
from PIL import Image, ImageDraw
from torch.autograd import Variable
from scipy.spatial import cKDTree
import scipy
import scipy.io as sio
from skimage.morphology import skeletonize
import ctypes
import time
import math
import pickle
import cv2


xx, yy = np.meshgrid(range(1000),range(1000))
grid = np.array((xx.ravel(), yy.ravel())).T
device = 'cuda:0'
so = ctypes.cdll.LoadLibrary 
lib = so("./cuda.so")  

def cuda_label(init,end,query):
    query = np.array(query)
    query_x = query[:,0].astype(np.float32)
    query_y = query[:,1].astype(np.float32)
    init_x = init[:,0].astype(np.float32)
    init_y = init[:,1].astype(np.float32)
    end_x = end[:,0].astype(np.float32)
    end_y = end[:,1].astype(np.float32)
    arr_flag = ctypes.c_int
    query_length = len(query_x)
    input_length = len(init_x)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_int_p = ctypes.POINTER(ctypes.c_int)
    #
    _init_x = init_x.ctypes.data_as(c_float_p)
    _init_y = init_y.ctypes.data_as(c_float_p)
    _end_x = end_x.ctypes.data_as(c_float_p)
    _end_y = end_y.ctypes.data_as(c_float_p)
    _query_x = query_x.ctypes.data_as(c_float_p)
    _query_y = query_y.ctypes.data_as(c_float_p)
    #
    distance = np.zeros((query_length,1)).astype(np.float32)
    _distance = distance.ctypes.data_as(c_float_p)
    _query_length = arr_flag(query_length)
    _input_length = arr_flag(input_length)
    lib.python_read_data(_init_x,_init_y,_end_x,_end_y,_query_x,_query_y,_distance,
        _input_length,_query_length)
    dds = np.ctypeslib.as_array(_distance,shape=(query_length,))
    return dds

seq_path = "./labels/annotation_seq"
seq_list = os.listdir(seq_path)
time0 = time.time()
length = len(seq_list)
for ii, seq in enumerate(seq_list):
    with open(os.path.join(seq_path,seq)) as json_file:
        data_json = json.load(json_file)
    init_whole = []
    end_whole = []
    for instance in data_json:
        for jj, v in enumerate(instance['seq'][1:]):
            init_whole.append(instance['seq'][jj])
            end_whole.append(v)

    dds = cuda_label(np.array(init_whole),np.array(end_whole),grid)
    dis_map = dds.reshape(1000,1000).T
    # print(dis_map)
    # inv_dis_map = np.array(1 / dis_map)
    # Image.fromarray(inv_dis_map*255).convert('RGB').save('./labels/inverse_distance_map/{}.png'.format(seq[:-5]))
    #
    dis_map = cv2.GaussianBlur(dis_map,(5,5),0)
    x = cv2.Sobel(dis_map,cv2.CV_16S,1,0,ksize=7)
    y = cv2.Sobel(dis_map,cv2.CV_16S,0,1,ksize=7)
    norm = np.array([x,y])
    norm = np.linalg.norm(norm,axis=0)
    norm = norm + (norm==0)
    #
    x = x / norm
    y = y / norm
    direction_matrix = [x,y]
    with open('./labels/direction_map/{}.pickle'.format(seq[:-5]), 'wb') as handle:
        pickle.dump(direction_matrix, handle)
    time_used = time.time() - time0
    time_remained = time_used / (ii+1) * (length-ii)
    print('Remained time: ',round(time_remained/3600,3),'h || ',ii,'/',length)

    # visualization
    # vis = np.ones((1000,1000,3))
    # vis[:,:,0] = x
    # vis[:,:,1] = y
    # vis[:,:,2] = (x+y)/2
    # Image.fromarray((vis*255).astype(np.uint8)).convert('RGB').save('./direciton_vis.png')
    
    # break
