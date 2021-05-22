import sys
sys.path.append('./lib')

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
from tqdm import tqdm
from arguments import get_parser, update_dir_candidate_test

parser = get_parser()
args = parser.parse_args()
update_dir_candidate_test(args)

tiff_dir = args.image_dir
gt_instance_mask_dir = args.instance_mask_dir
gt_mask_dir = args.mask_dir
skeleton_dir = './records/skeleton/test'

with open('./dataset/data_split.json','r') as jf:
    images = json.load(jf)['test']


class Graph():
    def __init__(self):
        self.instances = []
        self.all_vertices = []
        self.cross_edges = []  

    def init(self,num):
        for i in range(num):
            self.instances.append(Instance(i))

class Instance():
    def __init__(self,id):
        self.edges = []
        self.vertices = []
        self.num_e = 0
        self.num_v = 0
        self.index = id
        self.instance_num = 0

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
    
    def distance(self,u):
        return pow(pow(self.x-u.x,2) + pow(self.y-u.y,2),0.5)

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
    
    def find_nearest_gt(point):
        dd,ii = tree.query(point,k=[1])
        if dd < 30:
            return gt_instance_label[all_vertices[ii[0]][0],all_vertices[ii[0]][1]]
        else:
            return 0

    def find_extreme_points(mask,points):
        def mask_value(u,v):
            if u<0 or u>999 or v <0 or v> 999:
                return 0
            else:
                return mask[u,v]
        def check_leafpoint(point):
            u,v = point
            if mask_value(u-1,v-1) + mask_value(u,v-1)\
                 + mask_value(u+1,v-1) + mask_value(u-1,v)\
                 + mask_value(u+1,v) + mask_value(u-1,v+1)\
                 + mask_value(u,v+1) + mask_value(u+1,v+1) == 1:
                return True
            else:
                return False
        leaf_points = []
        for point in points:
            if check_leafpoint(point):
                leaf_points.append(point)
        if len(leaf_points) == 2:
            return leaf_points
        else:
            return None

    def update_graph(e):
        start_vertex = np.array([int(e.src.x),int(e.src.y)])
        end_vertex = np.array([int(e.dst.x),int(e.dst.y)])
        all_points = []
        p = start_vertex
        d = end_vertex - start_vertex
        N = np.max(np.abs(d))
        mask_raw[start_vertex[0],start_vertex[1],:] = [255,0,0]
        mask_raw[end_vertex[0],end_vertex[1],:] = [255,0,0]
        if N:
            s = d / (N)
            for i in range(0,N):
                p = p + s
                mask_raw[int(p[0]),int(p[1]),:] = [255,0,0]
    # load data
    tiff_image = np.array(Image.open(('{}/{}.tiff'.format(tiff_dir, image))))
    # instancemap
    gt_instance_label = np.array(Image.open(('{}/{}.png'.format(gt_instance_mask_dir, image))))[:,:,0]
    # binary map
    gt_binary_label = np.array(Image.open(('{}/{}.png'.format(gt_mask_dir, image)))).astype('float32')[:,:,0] / 255.0
    # obtained skeleton (predicted graph)
    skel_mask = np.array(Image.open(('{}/{}.png'.format(skeleton_dir, image)))).astype('float32')
    skel_mask = skel_mask.copy()[:,:,0] / 255.0
    # all gt pixels
    all_vertices = np.where(gt_binary_label>0)
    all_vertices = [[all_vertices[0][x],all_vertices[1][x]] for x in range(len(all_vertices[0]))]
    tree = cKDTree(all_vertices)
    # create graph
    graph = Graph()
    graph.init(np.max(gt_instance_label))

    ###################### Main algorithm ###################################
    counter = 0
    bad_counter = 0
    # candidate connections
    connections = []
    # get all predicted instances
    labeled_array = measure.label(skel_mask,connectivity=2)
    for i_idx in range(np.max(labeled_array)):
        # processing a line instance in the obtained skeleton map
        points = np.where(labeled_array==(i_idx+1))
        points = [[points[0][x],points[1][x]] for x in range(len(points[0]))]
        # find the leaf point of this line instance
        leaf_points = find_extreme_points(skel_mask,points)
        if leaf_points is not None:
            v_1 = Vertex(leaf_points[0][0],leaf_points[0][1])
            # determine which gt instance the leaf point belongs to
            v_1_index = find_nearest_gt([v_1.x,v_1.y])
            v_2 = Vertex(leaf_points[1][0],leaf_points[1][1])
            v_2_index = find_nearest_gt([v_2.x,v_2.y])
            v_1.instance_num = v_1_index
            v_2.instance_num = v_2_index
            # if two leaf points belong to the same curb instance, it's test for later processing
            if (v_1_index == v_2_index) and (v_1_index or v_2_index):
                graph.instances[0].add_v(v_1)
                graph.instances[0].add_v(v_2)
                graph.instances[0].add_e(Edge(v_1,v_2))
            # all leaf vertices
            graph.all_vertices.append(v_1)
            graph.all_vertices.append(v_2)

    # find n(n-1) possible connections, and the cost of each connection (edge) is the Euclidean distance
    for v in graph.instances[0].vertices:
        for u in graph.instances[0].vertices:
            distance_uv = v.distance(u)
            if not v.compare(u):
                e = Edge(u,v)
                e.cost = distance_uv
                # only consider candidates shorter than 200 pixels
                if distance_uv < 200 and distance_uv > 1:
                    # pre_flag means edge e is a candidate edge
                    e.pre_flag = 1
                    connections.append(e)
                graph.instances[0].add_e(e)

    for connection in connections:
        min_edge = connection
        if min_edge.pre_flag:
            counter += 1
            center = [(min_edge.src.x+min_edge.dst.x)//2,(min_edge.src.y+min_edge.dst.y)//2]
            candidate = Image.fromarray((skel_mask*255).astype(np.uint8)).convert('RGB')
            draw_good = ImageDraw.Draw(candidate)
            draw_good.line((min_edge.src.y,min_edge.src.x, min_edge.dst.y,min_edge.dst.x), fill='red')
            candidate = np.array(candidate)
            # crop a 320*320 sized patch for connection candidate training
            d = min(999,max(0,center[0]-160))
            u = min(999,max(0,center[0]+160))
            l = min(999,max(0,center[1]-160))
            r = min(999,max(0,center[1]+160))
            cropped_tiff_save = np.pad(tiff_image[d:u,l:r,:],((160-center[0]+d,160+center[0]-u),(160-center[1]+l,160+center[1]-r),(0,0)),mode='constant')
            candidate = np.pad(candidate[d:u,l:r,:],((160-center[0]+d,160+center[0]-u),(160-center[1]+l,160+center[1]-r),(0,0)),mode='constant')
            Image.fromarray(candidate).convert('RGB').save('./records/candidate_test/reason/{}_{}.png'.format(image,counter))
            Image.fromarray(cropped_tiff_save).convert('RGB').save('./records/candidate_test/rgb/{}_{}.png'.format(image,counter))
            # record this crop info into json for testing and testation
            connection_info = {'src':[int(min_edge.src.x),int(min_edge.src.y)],'dst':[int(min_edge.dst.x),int(min_edge.dst.y)],'label':1}
            with open('./records/candidate_test/json/{}_{}.json'.format(image,counter),'w') as jf:
                json.dump(connection_info,jf)

    return counter

counter = 0
print('Generating connection candidates for testing image...')
with tqdm(total=len(images), unit='img') as pbar:
    for i,image in enumerate(images):
        counter += process(image)
        # print('Rate: ',round(i/len(images)*100,3),'% | Samples: ',counter)

        pbar.set_postfix(**{'Sample': counter})
        pbar.update()