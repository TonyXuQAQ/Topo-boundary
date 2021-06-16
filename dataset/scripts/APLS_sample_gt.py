import numpy as np
import os
import pickle
import json
from PIL import Image, ImageDraw
import time

json_dir = '../labels/dense_seq'
pickle_dir = '../labels/sampled_seq'

class Vertex():
    def __init__(self,v,ii):
        self.coord = v
        self.index = ii
        self.neighbors = []
        self.neighbors = []
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
    def add_v(self,v,ii):
        self.vertices.append(v)
        index = ii-1
        u = self.find_vertex(index)
        if u is not None:
            if v not in u.neighbors:
                u.neighbors.append(v)
            if u not in v.neighbors:
                v.neighbors.append(u)
        index = ii+1
        u = self.find_vertex(index)
        if u is not None:
            if v not in u.neighbors:
                u.neighbors.append(v)
            if u not in v.neighbors:
                v.neighbors.append(u)
            
    def add_key_vertices(self,v):
        v.key_vertex = True
        self.key_vertices.append(v)
        self.sampled_vertices.append(v)

def generate_graph():

    json_list = os.listdir(json_dir)
    length = len(json_list)
    time00 = time.time()
    for j_idx, json_file in enumerate(json_list):
        # json_file='007182_03.json'
        with open(os.path.join(json_dir,json_file),'r') as jf:
            json_data = json.load(jf)
        output_json = []
        # img = Image.open('../../labels/binary_map/{}'.format(json_file[:-4]+'png'))
        # draw = ImageDraw.Draw(img)
        for points in json_data:
            graph = Graph()
            pre_points = points['seq']
            for ii, point in enumerate(pre_points):
                v = Vertex(point,ii)
                graph.add_v(v,ii)
                if not ii or ii == len(pre_points)-1:
                    graph.add_key_vertices(v)
            for key_vertex in graph.key_vertices:
                if len(key_vertex.neighbors):
                    for neighbor in key_vertex.neighbors:
                        key_vertex.neighbors.remove(neighbor)
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
                        curr_v.neighbors.remove(pre_v)
            
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
            output_json.append({'vertices':vertices,'adj':adjacent})
            
            # for point in vertices:
            #     draw.ellipse((point[1]-2,point[0]-2,point[1]+2,point[0]+2),fill='yellow',outline='yellow')
            # for v in graph.sampled_vertices:
            #     for u in v.sampled_neighbors:
            #         draw.line((v.coord[1],v.coord[0],u.coord[1],u.coord[0]),fill='yellow')
        # img.save(os.path.join('./',json_file[:-4]+'png'))
        with open(os.path.join(pickle_dir,json_file[:-4]+'pickle'),'wb') as jf:
            pickle.dump(output_json,jf)
        time_used = time.time() - time00
        time_remained = time_used / (j_idx+1) * (length-j_idx)
        print('Remained time: ',time_remained/3600,'h || ',j_idx,'/',length)

if __name__=='__main__':
    generate_graph()