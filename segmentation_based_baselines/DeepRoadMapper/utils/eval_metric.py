import argparse
import logging
import os
import sys

import skimage.graph
import numpy as np
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
import torchvision
import torchvision.transforms.functional as FT
# import spconv
from torch.autograd import Function
from skimage.morphology import skeletonize
# import cv2
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from skimage import measure
import json
from tqdm import tqdm
import matplotlib
sys.path.append(os.path.abspath('.'))
from arguments import get_parser
from scipy.sparse.csgraph import dijkstra

parser = get_parser()
args = parser.parse_args()

predicted_skel_dir = './records/reason/test/vis'
predicted_graph_dir = './records/reason/test/graph'
baseline_name = 'DeepRoadMapper'

def simplify_graph():
    print('Start simplifying graph...')
    with open('./dataset/data_split.json','r') as jf:
        json_data = json.load(jf)['test']
    skel_list = [x+'.png' for x in json_data]
    with tqdm(total=len(skel_list), unit='img') as pbar:
        for i,skel_name in enumerate(skel_list):
            skel = np.array(Image.open(os.path.join(predicted_skel_dir,skel_name)))[:,:,0]
            generate_graph(skel,skel_name)
            pbar.update()
        # break
    print('Finish simplifying graph...')

class Vertex():
    def __init__(self,v):
        self.coord = v
        self.index = v[0] * 1000 + v[1]
        self.neighbors = []
        self.unprocessed_neighbors = []
        self.processed_neighbors = []
        self.sampled_neighbors = []
        self.key_vertex = False
    def compare(self,v):
        if self.coord[0] == v[0] and self.coord[1] == v[1]:
            return True
        return False
    def next(self,previous):
        neighbors = self.neighbors.copy()
        neighbors.remove(previous)
        return neighbors[0]
    def distance(self,v):
        return pow(pow(self.coord[0]-v.coord[0],2)+pow(self.coord[1]-v.coord[1],2),0.5)

class Graph():
    def __init__(self):
        self.vertices = []
        self.key_vertices = []
        self.sampled_vertices = []
    def find_vertex(self,index):
        for v in self.vertices:
            if index == v.index:
                return v
        return None
    def add_v(self,v,neighbors):
        self.vertices.append(v)
        for n in neighbors:
            index = n[0] * 1000 + n[1]
            u = self.find_vertex(index)
            if u is not None:
                u.neighbors.append(v)
                v.neighbors.append(u)
                u.unprocessed_neighbors.append(v)
                v.unprocessed_neighbors.append(u)
    def find_key_vertices(self):
        for v in self.vertices:
            if len(v.neighbors)!=2:
                v.key_vertex = True
                self.key_vertices.append(v)
                self.sampled_vertices.append(v)


def generate_graph(skeleton,file_name):
    # Simplify skeletons into graphs with less vertices. Directly use skeleton (every foreground pixel is a vertex)
    # requires too much time and resource consumption.
    def find_neighbors(v,img,remove=False):
        output_v = []
        def get_pixel_value(u):
            if max(u) > 999 or min(u) < 0:
                return
            if img[u[0],u[1]]:
                output_v.append(u)

        get_pixel_value([v[0]+1,v[1]])
        get_pixel_value([v[0]-1,v[1]])
        get_pixel_value([v[0],v[1]-1])
        get_pixel_value([v[0],v[1]+1])
        get_pixel_value([v[0]+1,v[1]-1])
        get_pixel_value([v[0]+1,v[1]+1])
        get_pixel_value([v[0]-1,v[1]-1])
        get_pixel_value([v[0]-1,v[1]+1])
        if remove:
            img[v[0],v[1]] = 0
        return output_v

    graph = Graph()
    img = skeleton
    pre_points = np.where(img!=0)
    pre_points = [[pre_points[0][i],pre_points[1][i]] for i in range(len(pre_points[0]))]
    for point in pre_points:
        v = Vertex(point)
        graph.add_v(v,find_neighbors(point,img))
    graph.find_key_vertices()
    for key_vertex in graph.key_vertices:
        if len(key_vertex.unprocessed_neighbors):
            for neighbor in key_vertex.unprocessed_neighbors:
                key_vertex.unprocessed_neighbors.remove(neighbor)
                #
                curr_v = neighbor
                pre_v = key_vertex
                sampled_v = key_vertex
                counter = 1
                while(not curr_v.key_vertex):
                    if counter % 30 == 0:
                        sampled_v.sampled_neighbors.append(curr_v)
                        curr_v.sampled_neighbors.append(sampled_v)
                        sampled_v = curr_v
                        if not sampled_v.key_vertex:
                            graph.sampled_vertices.append(sampled_v)
                    next_v = curr_v.next(pre_v)
                    pre_v = curr_v
                    curr_v = next_v
                    counter += 1
                sampled_v.sampled_neighbors.append(curr_v)
                curr_v.sampled_neighbors.append(sampled_v)
                curr_v.unprocessed_neighbors.remove(pre_v)
    
    adjacent = np.ones((len(graph.sampled_vertices),len(graph.sampled_vertices))) * np.inf
    vertices = []
    for ii, v in enumerate(graph.sampled_vertices):
        v.index = ii
        vertices.append([int(v.coord[0]),int(v.coord[1])])
    for v in graph.sampled_vertices:
        for u in v.sampled_neighbors:
            dist = v.distance(u)
            adjacent[v.index,u.index] = dist
            adjacent[u.index,v.index] = dist
    with open(os.path.join(predicted_graph_dir,file_name[:-3]+'pickle'),'wb') as jf:
        pickle.dump({'vertices':vertices,'adj':adjacent},jf)

def thr_eval(name_in):
    def tuple2list(t):
            return [[t[0][x],t[1][x]] for x in range(len(t[0]))]
            
    pre_acc = np.zeros((4))
    pre_recall = np.zeros((4))
    pre_f1 = np.zeros((4))
    counter = 0
    print('Start calculating pixel-level metrics...')
    skel_dir = predicted_skel_dir
    img_list = os.listdir(skel_dir)
    with open('./dataset/data_split.json','r') as jf:
        img_list = json.load(jf)['test']
    img_list = [x+'.png' for x in img_list]
    with tqdm(total=len(img_list), unit='img') as pbar:
        for i,img in enumerate(img_list):
            gt_image = np.array(Image.open(os.path.join(args.mask_dir,img)))[:,:,0]
            pre_image = np.array(Image.open(os.path.join(skel_dir,img)))[:,:,0]
            if len(np.where(pre_image!=0)[0])==0:
                continue
            gt_points = tuple2list(np.where(gt_image!=0))
            pre_points = tuple2list(np.where(pre_image!=0))
            gt_tree = cKDTree(gt_points)

            for ii,thr in enumerate([2,5,10]):
                if len(pre_points):
                    # recall
                    pre_tree = cKDTree(pre_points)
                    pre_dds,_ = pre_tree.query(gt_points, k=1)
                    recall = len([x for x in pre_dds if x<thr])/len(pre_dds)
                    pre_recall[ii] = (pre_recall[ii] * counter + recall) / (counter+1)
                    # accuracy
                    gt_acc_dds,_ = gt_tree.query(pre_points, k=1)
                    acc = len([x for x in gt_acc_dds if x <thr])/len(gt_acc_dds)
                    pre_acc[ii] = (pre_acc[ii] * counter + acc) / (counter+1)
                    # f1 score
                    if recall*acc:
                        f1 = 2*acc*recall/(acc+recall)
                    else:
                        f1 = 0
                    pre_f1[ii] = (pre_f1[ii] * counter + f1)/(counter+1)
            counter+=1
            pbar.update()
    with open('./{}_thr_eval.json'.format(name_in),'w') as jf:
        json.dump({'pre_acc':pre_acc.tolist(),'pre_recall':pre_recall.tolist()\
            ,'r_f1':pre_f1.tolist()},jf)

def entropy_conn(name_in):
    print('Start calculating ECM...')
    with open('./dataset/data_split.json','r') as jf:
        json_data = json.load(jf)['test']
    img_list = [x+'.png' for x in json_data]
    ECM = 0
    naive = 0
    with tqdm(total=len(img_list), unit='img') as pbar:
        for i,img in enumerate(img_list):
            gt_image = np.array(Image.open(os.path.join(args.mask_dir,img)))[:,:,0]
            pre_image = np.array(Image.open(os.path.join(predicted_skel_dir,img)))[:,:,0]
            # find instances of the gt map
            gt_instance_map = measure.label(gt_image / 255,background=0)
            gt_instance_indexes = np.unique(gt_instance_map)[1:]
            # record length of all predicted instance assigned to a gt instance
            gt_assigned_lengths = [[] for x in range(len(gt_instance_indexes))]
            # record gt-instance length and vertices of this instance
            gt_instance_length = []
            gt_instance_points = []
            # record gt-instance pixels covered by projected predicted instances (measure completion)
            gt_covered = []
            # each gt_index labels is an gt instance
            for index in gt_instance_indexes:
                instance_map = (gt_instance_map == index)
                instance_points = np.where(instance_map==1)
                instance_points = [[instance_points[0][i],instance_points[1][i]] for i in range(len(instance_points[0]))]
                gt_instance_length.append(len(instance_points))
                gt_covered.append(np.zeros((len(instance_points))))
                gt_instance_points.append(instance_points)
            # find instances of the predicted graph map
            pre_instance_map = measure.label(pre_image / 255,background=0)
            pre_instance_indexes = np.unique(pre_instance_map)[1:]
            # all gt pixel points
            gt_points = np.where(gt_image!=0)
            gt_points = [[gt_points[0][i],gt_points[1][i]] for i in range(len(gt_points[0]))]
            tree = cKDTree(gt_points)
            # each pre_index is an predicted instance
            for index in pre_instance_indexes: 
                votes = []
                instance_map = (pre_instance_map == index)
                instance_points = np.where(instance_map==1)
                instance_points = [[instance_points[0][i],instance_points[1][i]] for i in range(len(instance_points[0]))]
                if instance_points:
                    # Each predicted point of the current pre-instance finds its closest gt point and votes
                    # to the gt-instance that the closest gt point belongs to.
                    _, iis = tree.query(instance_points,k=[1])
                    closest_gt_points = [[gt_points[x[0]][0],gt_points[x[0]][1]] for x in iis]
                    votes = [gt_instance_map[x[0],x[1]] for x in closest_gt_points]
                # count the voting results
                votes_summary = np.zeros((len(gt_instance_indexes)))
                for j in range(len(gt_instance_indexes)):
                    # the number of votes made to gt-instance j+1
                    votes_summary[j] = votes.count(j+1) 
                # find the gt-instance winning the most vote and assign the current pre-instance to it
                if np.max(votes_summary):
                    vote_result = np.where(votes_summary==np.max(votes_summary))[0][0]
                    # the length of the pre-instance assigned to corresponding gt-instance 
                    gt_assigned_lengths[vote_result].append(len(instance_points[0]))
                    # calculate projection of the predicted instance to corresponding gt-instance
                    instance_tree = cKDTree(gt_instance_points[vote_result])
                    _, iis = instance_tree.query(instance_points,k=[1])
                    gt_covered[vote_result][np.min(iis):np.max(iis)+1] = 1
            # calculate ECM
            entropy_conn = 0
            naive_conn = 0
            # iterate all gt-instances, calculate connectivity of each of them 
            for j,lengths in enumerate(gt_assigned_lengths):
                # lengths are the length of assigned pre-instances to the current gt-instance
                if len(lengths):
                    lengths = np.array(lengths)
                    # contribution of each assigned pre-instance
                    probs = (lengths / np.sum(lengths)).tolist()
                    C_j = 0
                    for p in probs:
                        C_j += -p*np.log2(p)
                    entropy_conn += np.exp(-C_j) * np.sum(gt_covered[j]) / len(gt_points)
                    naive_conn += 1 / len(lengths)
            if len(gt_assigned_lengths):
                naive_conn = naive_conn / len(gt_assigned_lengths)
            # weighted sum
            ECM = (ECM * i + entropy_conn)/(i+1)
            naive = (naive * i + naive_conn)/(i+1)
            pbar.update()
        output_json = {'ECM':np.array(ECM).tolist(),'naive':naive}
    with open('./{}_connectivity.json'.format(name_in),'w') as jf:
        json.dump(output_json,jf)

def APLS(name_in):
    print('Start calculating APLS')
    APLS = 0
    with open('./dataset/data_split.json','r') as jf:
        json_data = json.load(jf)['test']
    image_list = [x+'.pickle' for x in json_data]
    with tqdm(total=len(image_list), unit='img') as pbar:
        for idx, image_name in enumerate(image_list):
            apls = 0
            apls_counter = 0
            with open(os.path.join(args.sampled_seq_dir,image_name),'rb') as jf:
                gt_data = pickle.load(jf)
            with open(os.path.join(predicted_graph_dir,image_name),'rb') as jf:
                pre_data = pickle.load(jf)
            adjacent = pre_data['adj']
            pre_points = pre_data['vertices']
            if len(pre_points):
                tree = cKDTree(pre_points)
                for gt_data_instance in gt_data:
                    gt_points_instance = gt_data_instance['vertices']
                    gt_adjacent = gt_data_instance['adj']
                    if len(gt_points_instance) > 5:
                        # randomly find some pairs of points for APLS calculation
                        for ii in range(len(gt_points_instance)//3):
                            select_index = np.random.choice(len(gt_points_instance),2,replace=False)
                            init_v = gt_points_instance[select_index[0]]
                            end_v = gt_points_instance[select_index[1]]
                            source = select_index[0]
                            target = select_index[1]
                            dist_matrix = dijkstra(gt_adjacent, directed=False, indices=[source],
                                                    unweighted=False)
                            gt_length = dist_matrix[0,target]
                            # find corresponding pre_vertex by min-Euclidean distance
                            dds,iis = tree.query([init_v,end_v],k=1)
                            source = iis[0]
                            target = iis[1]
                            # vertex too far away is treated as not reachable
                            if np.max(dds) < 50:
                                dist_matrix = dijkstra(adjacent, directed=False, indices=[source],
                                                    unweighted=False)
                                pre_length = dist_matrix[0,target]
                            else:
                                pre_length = 10000
                            apls += min(1,abs(gt_length - pre_length) / (gt_length))
                            apls_counter += 1
            if apls_counter:
                apls = 1 - apls / apls_counter
            APLS = (APLS * idx + apls) / (idx + 1)
            pbar.update()
            # break
    with open('./{}_APLS.json'.format(name_in),'w') as jf:
        json.dump({'APLS':APLS},jf)
    return APLS



def latex(file_name):
    with open('./{}_thr_eval.json'.format(file_name)) as jf:
        json_load = json.load(jf)
    
    with open('./{}_connectivity.json'.format(file_name)) as jf:
        json_load_c = json.load(jf)

    with open('./{}_APLS.json'.format(file_name)) as jf:
        json_load_a = json.load(jf)

    print('===========================================')
    print('Precision 2/5/10: {}/{}/{}'.format(('%.3f'%json_load['pre_acc'][0]),\
                                            ('%.3f'%json_load['pre_acc'][1]),\
                                            ('%.3f'%json_load['pre_acc'][2])))
    print('Recall 2/5/10: {}/{}/{}'.format(('%.3f'%json_load['pre_recall'][0]),\
                                            ('%.3f'%json_load['pre_recall'][1]),\
                                            ('%.3f'%json_load['pre_recall'][2])))
    print('F1-score 2/5/10: {}/{}/{}'.format(('%.3f'%json_load['r_f1'][0]),\
                                            ('%.3f'%json_load['r_f1'][1]),\
                                            ('%.3f'%json_load['r_f1'][2])))
    print('Naive connectivity: {}'.format(('%.3f'%json_load_c['naive'])))
    print('APLS: {}'.format(('%.3f'%json_load_a['APLS'])))
    print('ECM: {}'.format(('%.3f'%json_load_c['ECM'])))
    print('===========================================')

    # ============== latex format 1
    # print(('%.3f'%json_load['pre_acc'][0])
    #     ,'&',('%.3f'%json_load['pre_acc'][1])\
    #     ,'&', ('%.3f'%json_load['pre_acc'][2])\
    # ,'&', ('%.3f'%json_load['pre_recall'][0] )\
    #     ,'&',  ('%.3f'%json_load['pre_recall'][1])\
    #     ,'&', ('%.3f'%json_load['pre_recall'][2] )\
    # ,'&', ('%.3f'%json_load['r_f1'][0])\
    #     ,'&',('%.3f'%json_load['r_f1'][1])\
    #     ,'&', ('%.3f'%json_load['r_f1'][2])\
    # ,'&', ('%.3f'%json_load_c['naive'])\
    # ,'&', ('%.3f'%json_load_a['APLS'])\
    # ,'&', ('%.3f'%json_load_c['ECM']))
    # ============== latex format 2
    # print(('\\textcolor{blue}{%.3f}'%json_load['pre_acc'][0])
    #     ,'&',('\\textcolor{blue}{%.3f}'%json_load['pre_acc'][1])\
    #     ,'&', ('\\textcolor{blue}{%.3f}'%json_load['pre_acc'][2])\
    # ,'&', ('\\textcolor{blue}{%.3f}'%json_load['pre_recall'][0] )\
    #     ,'&',  ('\\textcolor{blue}{%.3f}'%json_load['pre_recall'][1])\
    #     ,'&', ('\\textcolor{blue}{%.3f}'%json_load['pre_recall'][2] )\
    # ,'&', ('\\textcolor{blue}{%.3f}'%json_load['r_f1'][0])\
    #     ,'&',('\\textcolor{blue}{%.3f}'%json_load['r_f1'][1])\
    #     ,'&', ('\\textcolor{blue}{%.3f}'%json_load['r_f1'][2])\
    # ,'&', ('\\textcolor{blue}{%.3f}'%json_load_c['naive'])\
    # ,'&', ('\\textcolor{blue}{%.3f}'%json_load_a['APLS'])\
    # ,'&', ('\\textcolor{blue}{%.3f}'%json_load_c['ECM']))
    

def main():
    print('------------ Starting evaluation... ------------')
    print(baseline_name)
    simplify_graph()
    thr_eval(baseline_name)
    entropy_conn(baseline_name)
    APLS(baseline_name)
    print('Finish evaluation!')
    latex(baseline_name)

if __name__=='__main__':
    main()

