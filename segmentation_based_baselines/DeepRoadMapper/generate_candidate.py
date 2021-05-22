import imageio
from PIL import Image, ImageDraw
import numpy as np
from multiprocessing import Pool
import subprocess
from scipy.spatial import cKDTree
import os
from skimage import measure
import pickle
import random
import json
import time
from arguments import get_parser, update_dir_candidate_train

parser = get_parser()
args = parser.parse_args()
update_dir_candidate_train(args)

tiff_dir = args.image_dir
gt_instance_mask_dir = args.instance_mask_dir
gt_mask_dir = args.mask_dir
skeleton_dir = './records/skeleton/train'

with open('./dataset/data_split.json','r') as jf:
    images = json.load(jf)['train']


class Graph():
    def __init__(self):
        self.curbs = []
        self.all_vertices = []
        self.cross_edges = []  

    def init(self,num):
        for i in range(num):
            self.curbs.append(Curb(i))

class Curb():
    def __init__(self,id):
        self.edges = []
        self.vertices = []
        self.num_e = 0
        self.num_v = 0
        self.index = id
        self.curb_num = 0

    def add_v(self,v):
        for v_in in self.vertices:
            if v_in.if_same(v):
                break
        self.vertices.append(v)
        v.id = self.num_v
        self.num_v += 1

    def add_e(self,e):
        self.edges.append(e)
        self.num_e += 1
        # self.add_v(e.src)
        # self.add_v(e.dst)
        e.src.edges.append(e)
        e.dst.edges.append(e)
        e.src.neighbor.append(e.dst)
        e.dst.neighbor.append(e.src)
        

class Vertex():
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.edges = []
        self.neighbor = []
        self.id = None
    
    def if_same(self,u):
            if (self.x == u.x) and (self.y == u.y):
                return True
            return False

    def compare(self,u):
        if self.if_same(u) or (u in self.neighbor):
            return True
        return False

class Edge():
    def __init__(self,v_1,v_2):
        self.src = v_1
        self.dst = v_2
        self.pre_flag = 0
        self.cost = 0
    def show(self):
        print('-------Edge------')
        print('src',[self.src.x,self.src.y])
        print('dst',[self.dst.x,self.dst.y])
        print('-------End----------')

def process(image):
    
    sat = Image.open('{}/{}.tiff'.format(tiff_dir, image))
    # instancemap
    gt_label = Image.open('{}/{}.png'.format(gt_instance_mask_dir, image))[:,:,0]
    # binary map
    gt_mask = Image.open('{}/{}.png'.format(gt_mask_dir, image)).astype('float32')[:,:,0] / 255.0
    # obtained skeleton (predicted graph)
    skel_mask = Image.open('{}/{}.png'.format(skeleton_dir, image)).astype('float32')
    skel_mask = skel_mask.copy()[:,:,0] / 255.0
    all_vertices = np.where(gt_mask>0)
    all_vertices = [[all_vertices[0][x],all_vertices[1][x]] for x in range(len(all_vertices[0]))]
    tree = cKDTree(all_vertices)

    graph = Graph()
    graph.init(np.max(gt_label))

    # propose connections
    labeled_array = measure.label(skel_mask,connectivity=2)
    for i_idx in range(np.max(labeled_array)):
        # processing a line segment in the obtained skeleton map
        points = np.where(labeled_array==(i_idx+1))
        points = [[points[0][x],points[1][x]] for x in range(len(points[0]))]

for i,image in enumerate(images):
    print('Generating connection candidates for training image... {}/{}'.format(i,len(images)-1))
    process(image)
