import numpy as np
import pandas as pd
import time
import json
import warnings
import os
from skimage import measure
from PIL import Image, ImageDraw
import pickle
warnings.filterwarnings("ignore")

def generate_patch():
    intersections = pd.read_csv('./csv/inter_new.csv')
    grid_inters = pd.read_csv('./csv/grid_inter.csv')
    vertices = pd.read_csv('./csv/vertices_new.csv')
    grids = pd.read_csv('./csv/grids_new.csv')
    time00 = time.time()
    length = len(grids)
    for idx, grid in grids.iterrows():
        left,up,right,down = grid['left'],grid['top'],grid['right'],grid['bottom']
        l,u,r,d=left,up,right,down
        vertices_inside = vertices.loc[vertices['id']==grid['id']]
        if len(vertices_inside):
            exception_features = vertices_inside.loc[vertices_inside['FEATURE_CODE']==2270]
            if not len(exception_features):
                inter_inside = intersections.loc[intersections['id']==grid['id']]
                grid_inters_inside = grid_inters.loc[(grid_inters['X']>=l)&(grid_inters['X']<=r)&(grid_inters['Y']>=d)&(grid_inters['Y']<=u)]
                ii = (d%2500)//500
                jj = (l%2500)//500
                v_inside = vertices_inside
                i_inside = inter_inside
                g_inside = grid_inters_inside
                
                v_inside['X'] = v_inside['X'].apply(lambda x: x-l)
                v_inside['Y'] = v_inside['Y'].apply(lambda x: x-d)
                v_inside = v_inside[['X','Y','OBJECTID','SHAPE_Length','vertex_index','distance','angle']]
                i_inside['X'] = i_inside['X'].apply(lambda x: x-l)
                i_inside['Y'] = i_inside['Y'].apply(lambda x: x-d)
                i_inside = i_inside[['X','Y','OBJECTID','OBJECTID_2']]
                g_inside['X'] = g_inside['X'].apply(lambda x: x-l)
                g_inside['Y'] = g_inside['Y'].apply(lambda x: x-d)
                g_inside = g_inside[['X','Y','OBJECTID','SHAPE_Length']]
                
                json_vertices = json.loads(v_inside.to_json(orient='records'))
                json_data = {'info':{'left':int(l),'right':int(r),'up':int(u),'down':int(d)},\
                    'vertices':sorted(json_vertices, key=lambda k: (int(k['OBJECTID']),int(k["vertex_index"])))  ,\
                    'intersections':json.loads(i_inside.to_json(orient='records')),\
                    'grid_intersections':json.loads(g_inside.to_json(orient='records'))}
                    
                with open('./json/{}_{}{}.json'.format(str(int(grid['IMAGE'])).zfill(6),int(4-ii),int(jj)),'w') as jf:
                    json.dump(json_data,jf)
    
                time_used = time.time() - time00
                time_remained = time_used / (idx+1) * (length-idx)
                print('Remained time: ',time_remained/3600,'h || ',idx,'/',length)

# generate_patch()
#################################
# def sort_vertices():
#     json_list = os.listdir('./raw_json')
#     for ii, json_file in enumerate(json_list):
#         with open('./raw_json/{}'.format(json_file),'r') as jf:
#             json_data = json.load(jf)
#         vertices = json_data['vertices']
#         vertices = sorted(vertices, key=lambda k: (int(k['OBJECTID']),int(k["vertex_index"])))  
#         # create patch
#         json_data['vertices'] = vertices
#         with open('./json/{}'.format(json_file),'w') as jf:
#             json.dump(json_data,jf)
#         print(ii,len(json_list))
        
# sort_vertices()
#################################
class Patch():
    def __init__(self):
        self.segments = []
        self.ids = []
        self.lengths = []
        self.output_segments = []
        self.intersections = []

    def len(self):
        return len(self.segments)

    def new_segment(self,id,unique_id,length):
        segment = Segment(id,unique_id,length)
        self.segments.append(segment)
        self.ids.append(id)
        self.lengths.append(length)
        return segment

    def locate_segment_id(self,id):
        for s in self.segments:
            if id in s.ids:
                return s
    
    def duplicate_intersections(self,i):
        if [i['Y'],i['X']] in self.intersections:
            return True
        else:
            return False

    def merge_segments(self,s1,s2,p1,p2):
        if p1 == 'end' and p2 == 'init':
            s1.end_vertex.overlap_vertex = s2.init_vertex
            s2.init_vertex.overlap_vertex = s1.end_vertex
            s1.end_vertex.child = s2.init_vertex.child
            if s2.init_vertex.child is not None:
                s2.init_vertex.child.parent = s1.end_vertex
            s1.ids.extend(s2.ids)
            s1.vertices.extend(s2.vertices[1:])
            s1.init_vertex = s1.vertices[0]
            s1.end_vertex = s1.vertices[-1]
            self.segments.remove(s2)
        elif p1 == 'init' and p2 == 'end':
            s2.end_vertex.overlap_vertex = s1.init_vertex
            s1.init_vertex.overlap_vertex = s2.end_vertex
            s2.end_vertex.child = s1.init_vertex.child
            if s1.init_vertex.child is not None:
                s1.init_vertex.child.parent = s2.end_vertex
            s2.ids.extend(s1.ids)
            s2.vertices.extend(s1.vertices[1:])
            s2.init_vertex = s2.vertices[0]
            s2.end_vertex = s2.vertices[-1]
            self.segments.remove(s1)
        elif p1 == 'init' and p2 == 'init':
            s1.reverse()
            s1.end_vertex.overlap_vertex = s2.init_vertex
            s2.init_vertex.overlap_vertex = s1.end_vertex
            s1.end_vertex.child = s2.init_vertex.child
            if s2.init_vertex.child is not None:
                s2.init_vertex.child.parent = s1.end_vertex
            s1.ids.extend(s2.ids)
            s1.vertices.extend(s2.vertices[1:])
            s1.init_vertex = s1.vertices[0]
            s1.end_vertex = s1.vertices[-1]
            self.segments.remove(s2)
        elif p1 == 'end' and p2 == 'end':
            s2.reverse()
            s1.end_vertex.overlap_vertex = s2.init_vertex
            s2.init_vertex.overlap_vertex = s1.end_vertex
            s1.end_vertex.child = s2.init_vertex.child
            if s2.init_vertex.child is not None:
                s2.init_vertex.child.parent = s1.end_vertex
            s1.ids.extend(s2.ids)
            s1.vertices.extend(s2.vertices[1:])
            s1.init_vertex = s1.vertices[0]
            s1.end_vertex = s1.vertices[-1]
            self.segments.remove(s2)
        else:
            print('Error!!!')


class Segment():
    def __init__(self,id,u_id,length):
        if isinstance(id,int):
            self.ids = [id]
        else:
            self.ids = id
        self.unique_id = u_id
        self.vertices = []
        self.init_vertex = None
        self.end_vertex = None
        self.length = length
        self.split_indexs = []
        self.type = 'unprocessed'
        self.split_indexs = []
        
    def len(self):
        return len(self.vertices)

    def new_vertex(self,v):
        if len(self.vertices):
            vertex = Vertex(v['X'],v['Y'],v['OBJECTID'],v['vertex_index'])
            self.end_vertex.child = vertex
            vertex.parent = self.end_vertex
        else:
            vertex = Vertex(v['X'],v['Y'],v['OBJECTID'],v['vertex_index'])
        self.vertices.append(vertex)
        self.init_vertex = self.vertices[0]
        self.end_vertex = self.vertices[-1]

    def locate_vertex(self,v):
        if v['X'] == self.init_vertex.x and v['Y'] == self.init_vertex.y:
            return 'init'
        elif v['X'] == self.end_vertex.x and v['Y'] == self.end_vertex.y:
            return 'end'
        else:
            for ii, vv in enumerate(self.vertices):
                if v['X'] == vv.x and v['Y'] == vv.y:
                    return ii
        # this causes error            
        return None

    def reverse(self):
        self.vertices = self.vertices[::-1]
        if len(self.vertices)>1:
            self.end_vertex.child = self.vertices[1]
            self.end_vertex.parent = None
            if self.end_vertex.order=='forward':self.end_vertex.order = 'reverse' 
            elif self.end_vertex.order=='reverse':self.end_vertex.order = 'forward' 
            self.init_vertex.parent = self.vertices[-2]
            self.init_vertex.child = None
            if self.init_vertex.order=='forward':self.init_vertex.order = 'reverse' 
            elif self.init_vertex.order=='reverse':self.init_vertex.order = 'forward'

        for ii,v in enumerate(self.vertices[1:-1]):
            if v.order=='forward':v.order = 'reverse'  
            elif v.order=='reverse':v.order = 'forward' 
            v.parent = self.vertices[ii]
            v.child = self.vertices[ii+2]
        self.init_vertex = self.vertices[0]
        self.end_vertex = self.vertices[-1]

class Vertex():
    def __init__(self,x,y,id,index):
        self.x = x
        self.y = y
        self.id = id
        self.index = index
        self.parent = None
        self.child = None
        self.order = 'forward'
        self.overlap_vertex = None
    

def merge_lines():
    vertices_csv = pd.read_csv('./csv/vertices_new.csv')
    json_list = os.listdir('./json')
    time00 = time.time()
    length = len(json_list)
    for j_idx, json_file in enumerate(json_list):
        # json_file = '042167_14.json'
        print(json_file)
        remove_current_patch = False
        with open('./json/{}'.format(json_file),'r') as jf:
            json_data = json.load(jf)
        vertices = json_data['vertices']
        inters = json_data['intersections']
        grid_inters = json_data['grid_intersections'] 
        info = json_data['info']
        # create patch
        patch = Patch()
        curr_segment = Segment(-1,-1,0)
        for v in vertices:
            if v['OBJECTID'] != curr_segment.unique_id:
                curr_segment = patch.new_segment(v['OBJECTID'],v['OBJECTID'],v['SHAPE_Length'])
                curr_segment.new_vertex(v)
            else:
                curr_segment.new_vertex(v)
        for s in patch.segments:
            if len(s.vertices) == 1:
                s.vertices[0].order = 'bidirection'
        # merge segments
        for i in inters:
            if not patch.duplicate_intersections(i):
                patch.intersections.append([i['Y'],i['X']])
                if i['OBJECTID'] in patch.ids and i['OBJECTID_2'] in patch.ids:
                    segment1 = patch.locate_segment_id(i['OBJECTID'])
                    segment2 = patch.locate_segment_id(i['OBJECTID_2'])
                    pos1 = segment1.locate_vertex(i)
                    pos2 = segment2.locate_vertex(i)
                    # circle
                    if segment1.unique_id == segment2.unique_id:
                        segment1.type = 'circle'
                    elif isinstance(pos1, str) and isinstance(pos2, str):
                        patch.merge_segments(segment1,segment2,pos1,pos2)
                    else:
                        remove_current_patch = True
                        break
        if remove_current_patch:
            continue
        ############################# merge finished
        # find edge vertices
        for s in patch.segments:
            if s.type=='unprocessed' or s.type=='circle':
                start_index = 0
                for ii,v in enumerate(s.vertices[1:]):
                    if abs(v.index - v.parent.index) >1 and v.id == v.parent.id:
                        s.split_indexs.append([start_index,ii+1])
                        start_index = ii + 1
                if start_index:
                    s.split_indexs.append([start_index,len(s.vertices)])
                    s.type = 'split'
                else:
                    if not s.type == 'circle':
                        s.type = 'processed'
                    else:
                        s.type = 'processed_circle'
        # split by edge vertices
        for s in patch.segments:
            if s.type=='split':
                for indexes in s.split_indexs:
                    segment = Segment(s.ids,s.unique_id,s.length)
                    segment.vertices = s.vertices[indexes[0]:indexes[1]]
                    segment.init_vertex = segment.vertices[0]
                    segment.end_vertex = segment.vertices[-1]
                    segment.type='processed'
                    patch.output_segments.append(segment)
            else:
                patch.output_segments.append(s)
        # interpolate boundary vertices
        for s in patch.output_segments:
            if s.type=='processed':
                # find previous vertex of the init
                v = s.vertices[0]
                id = v.id
                v_inside = [v.y,v.x]
                if v.order == 'forward':
                    index = v.index - 1
                elif v.order == 'reverse':
                    index = v.index + 1
                else:
                    if v.index == 0:
                        index = 1
                    else:
                        index = v.index - 1
                if (v.overlap_vertex is not None) and v.child is not None:# and ((v.index==0 and v.order != 'forward') or (v.index!=0 and v.order == 'forward')):
                    if v.overlap_vertex.id != v.child.id:
                        id = v.overlap_vertex.id
                        if v.overlap_vertex.order == 'forward':
                            index = v.overlap_vertex.index - 1
                        elif v.overlap_vertex.order == 'reverse':
                            index = v.overlap_vertex.index + 1
                        else:
                            if v.overlap_vertex.index == 0:
                                index = 1
                            else:
                                index = v.overlap_vertex.index - 1
                        v_inside = [v.overlap_vertex.y,v.overlap_vertex.x]
                    
                v_outside = vertices_csv.loc[(vertices_csv['vertex_index']==index)&(vertices_csv['OBJECTID']==id)]
                if len(v_outside):
                    v_outside = [v_outside['Y'].values[0]-info['down'],v_outside['X'].values[0]-info['left']]
                    vector = np.array(v_outside) - np.array(v_inside)
                    if v_outside[1] >= 500:
                        v_boundary_candidate = [vector[0]/vector[1]*(500-v_inside[1])+v_inside[0],500]
                        if v_boundary_candidate[0] >= 0 and v_boundary_candidate[1] >= 0 and v_boundary_candidate[0] <= 500 and v_boundary_candidate[1] <= 500:
                            v_boundary = v_boundary_candidate
                    elif v_outside[1] <= 0:
                        v_boundary_candidate = [vector[0]/vector[1]*(0-v_inside[1])+v_inside[0],0]
                        if v_boundary_candidate[0] >= 0 and v_boundary_candidate[1] >= 0 and v_boundary_candidate[0] <= 500 and v_boundary_candidate[1] <= 500:
                            v_boundary = v_boundary_candidate
                    if v_outside[0] >= 500:    
                        v_boundary_candidate = [500,vector[1]/vector[0]*(500-v_inside[0])+v_inside[1]]
                        if v_boundary_candidate[0] >= 0 and v_boundary_candidate[1] >= 0 and v_boundary_candidate[0] <= 500 and v_boundary_candidate[1] <= 500:
                            v_boundary = v_boundary_candidate
                    elif v_outside[0] <= 0:
                        v_boundary_candidate = [0,vector[1]/vector[0]*(0-v_inside[0])+v_inside[1]]
                        if v_boundary_candidate[0] >= 0 and v_boundary_candidate[1] >= 0 and v_boundary_candidate[0] <= 500 and v_boundary_candidate[1] <= 500:
                            v_boundary = v_boundary_candidate
                    v_boundary = Vertex(v_boundary[1],v_boundary[0],0,0)
                    s.init_vertex = v_boundary
                    s.vertices.append(v_boundary)
                    s.vertices[1:] = s.vertices[:-1]
                    s.vertices[0] = v_boundary
                # ############### find after vertex of the end
                v = s.vertices[-1]
                id = v.id
                v_inside = [v.y,v.x]
                if v.order == 'forward':
                    index = v.index + 1
                elif v.order == 'reverse':
                    index = v.index - 1
                else:
                    if v.index == 0:
                        index = 1
                    else:
                        index = v.index - 1
                if v.overlap_vertex is not None and v.parent is not None:
                    if v.overlap_vertex.id != v.parent.id:
                        id = v.overlap_vertex.id
                        if v.overlap_vertex.order == 'forward':
                            index = v.overlap_vertex.index + 1
                        elif v.overlap_vertex.order == 'reverse':
                            index = v.overlap_vertex.index - 1
                        else:
                            if v.overlap_vertex.index == 0:
                                index = 1
                            else:
                                index = v.overlap_vertex.index - 1
                        v_inside = [v.overlap_vertex.y,v.overlap_vertex.x]
                v_outside = vertices_csv.loc[(vertices_csv['vertex_index']==index)&(vertices_csv['OBJECTID']==id)]
                if len(v_outside):
                    v_outside = [v_outside['Y'].values[0]-info['down'],v_outside['X'].values[0]-info['left']]
                    vector = np.array(v_outside) - np.array(v_inside)
                    if v_outside[1] >= 500:
                        v_boundary_candidate = [vector[0]/vector[1]*(500-v_inside[1])+v_inside[0],500]
                        if v_boundary_candidate[0] >= 0 and v_boundary_candidate[1] >= 0 and v_boundary_candidate[0] <= 500 and v_boundary_candidate[1] <= 500:
                            v_boundary = v_boundary_candidate
                    elif v_outside[1] <= 0:
                        v_boundary_candidate = [vector[0]/vector[1]*(0-v_inside[1])+v_inside[0],0]
                        if v_boundary_candidate[0] >= 0 and v_boundary_candidate[1] >= 0 and v_boundary_candidate[0] <= 500 and v_boundary_candidate[1] <= 500:
                            v_boundary = v_boundary_candidate
                    if v_outside[0] >= 500:    
                        v_boundary_candidate = [500,vector[1]/vector[0]*(500-v_inside[0])+v_inside[1]]
                        if v_boundary_candidate[0] >= 0 and v_boundary_candidate[1] >= 0 and v_boundary_candidate[0] <= 500 and v_boundary_candidate[1] <= 500:
                            v_boundary = v_boundary_candidate
                    elif v_outside[0] <= 0:
                        v_boundary_candidate = [0,vector[1]/vector[0]*(0-v_inside[0])+v_inside[1]]
                        if v_boundary_candidate[0] >= 0 and v_boundary_candidate[1] >= 0 and v_boundary_candidate[0] <= 500 and v_boundary_candidate[1] <= 500:
                            v_boundary = v_boundary_candidate
                    v_boundary = Vertex(v_boundary[1],v_boundary[0],0,0)
                    s.end_vertex = v_boundary
                    s.vertices.append(v_boundary)
        if remove_current_patch:
            continue        
        # 
        image = Image.fromarray(np.zeros((1000,1000,3)).astype(np.uint8)).convert('RGB')
        draw = ImageDraw.Draw(image)
        output_json = []
        for s in patch.output_segments:
            output = []
            if len(s.vertices) > 3:
                output.append([(500-s.vertices[0].y) * 2,s.vertices[0].x*2])
                for ii,v in enumerate(s.vertices[1:]):
                    x = v.x * 2
                    y = (500-v.y) * 2
                    x2 = s.vertices[ii].x*2
                    y2 = (500-s.vertices[ii].y)*2
                    # draw.line((x,y,x2,y2),fill='yellow')
                    output.append([(500-v.y) * 2,v.x*2])
                output_json.append(output)
            s.type='vis'
        with open('./json_merged/{}'.format(json_file),'w') as jf:
            json.dump(output_json,jf)
        time_used = time.time() - time00
        time_remained = time_used / (j_idx+1) * (length-j_idx)
        print('Remained time: ',time_remained/3600,'h || ',j_idx,'/',length)
        # break

# merge_lines()

############################### generate label
def binary_label_seq():
    def update_graph(start_vertex,end_vertex,graph):
        start_vertex = np.array(start_vertex)
        end_vertex = np.array(end_vertex)
        instance_vertices = []
        p = start_vertex
        d = end_vertex - start_vertex
        N = np.max(np.abs(d))
        segment_list = []
        segment_list.append(start_vertex.tolist())
        graph[start_vertex[0],start_vertex[1]] = 1
        graph[end_vertex[0],end_vertex[1]] = 1
        if N:
            s = d / (N)
            for i in range(0,N):
                p = p + s
                graph[max(0,min(999,int(round(p[0])))),max(0,min(999,int(round(p[1]))))] = 1
                segment_list.append([max(0,min(999,int(round(p[0])))),max(0,min(999,int(round(p[1]))))])
        return graph, segment_list, [start_vertex.tolist(),end_vertex.tolist()]


    def update_orientation(start_vertex,end_vertex,graph,value):
        start_vertex = np.array([max(0,min(999,int(round(start_vertex[0])))),max(0,min(999,int(round(start_vertex[1]))))])
        end_vertex = np.array([max(0,min(999,int(round(end_vertex[0])))),max(0,min(999,int(round(end_vertex[1]))))])
        if not (start_vertex[0] == end_vertex[0] and start_vertex[1] == end_vertex[1]):
            instance_vertices = []
            p = start_vertex
            d = end_vertex - start_vertex
            N = np.max(np.abs(d))
            segment_list = []
            segment_list.append(start_vertex.tolist())
            graph[start_vertex[0],start_vertex[1]] = value
            graph[end_vertex[0],end_vertex[1]] = value
            if N:
                s = d / (N)
                for i in range(0,N):
                    p = p + s
                    graph[max(0,min(999,int(round(p[0])))),max(0,min(999,int(round(p[1]))))] = value
        return graph

    json_list = os.listdir('../json_merged')
    time00 = time.time()
    length = len(json_list)
    for j_idx, json_file in enumerate(json_list):
        remove_current_patch = False
        # json_file = '000212_42.json'
        binary_label = np.zeros((1000,1000))
        orientation_map = np.zeros((1000,1000))
        with open('../json_merged/{}'.format(json_file),'r') as jf:
            json_data = json.load(jf)
        dense_seq = []
        segments_seq = []
        for segment in json_data:
            instance_list = []
            for ii, v in enumerate(segment[1:]):
                v1 = [max(0,min(999,int(round(segment[ii][0])))),max(0,min(999,int(round(segment[ii][1]))))]
                v2 = [max(0,min(999,int(round(v[0])))),max(0,min(999,int(round(v[1]))))]
                if not(v1[0]==v2[0] and v1[1]==v2[1]):
                    binary_label, segment_list, segment_json = update_graph(v1,v2,binary_label)
                    instance_list.extend(segment_list)
                    segments_seq.append(segment_json)
                    #
                    vector = np.array(v) - np.array(segment[ii])
                    norm = np.linalg.norm(vector)
                    theta = np.arccos(vector[0]/norm)
                    if vector[1] < 0:
                        theta = 2*np.pi - theta
                    angle = theta // (np.pi/32) + 1
                    if angle > 64 or angle < 0:
                        print('error!!!!!!!!!!!!!')
                    orientation_map = update_orientation(v1,v2,orientation_map,angle)
            if len(instance_list):
                if not(instance_list[0][0]==instance_list[-1][0] and instance_list[0][1]==instance_list[-1][1]):
                    if instance_list[0][0] != 0 and instance_list[0][1] != 0 and instance_list[0][0] != 999 and instance_list[0][1] != 999:
                        remove_current_patch = True
                        break
                    if instance_list[-1][0] != 0 and instance_list[-1][1] != 0 and instance_list[-1][0] != 999 and instance_list[-1][1] != 999:
                        remove_current_patch = True
                        break
                dense_seq.append({'init_vertex':instance_list[0],'end_vertex':instance_list[-1],
                                'seq':instance_list})
        if remove_current_patch:
            continue
        if len(dense_seq):
            # with open('../labels/dense_seq/{}'.format(json_file),'w') as jf:
            #     json.dump(dense_seq,jf)
            # with open('../labels/segments/{}'.format(json_file),'w') as jf:
            #     json.dump(segments_seq,jf)
            # Image.fromarray(binary_label*255).convert('RGB').save('../labels/binary_label/{}png'.format(json_file[:-4]))
            Image.fromarray(orientation_map).convert('RGB').save('../labels/orientation_map/{}png'.format(json_file[:-4]))
        time_used = time.time() - time00
        time_remained = time_used / (j_idx+1) * (length-j_idx)
        # print('Remained time: ',time_remained/3600,'h || ',j_idx,'/',length)
        # break
# binary_label_seq()

########################################################
def split_dataset():
    json_list = os.listdir('../labels/binary_map')
    counter = 0
    no_files = []
    for json_file in json_list:
        if not os.path.exists('../cropped_tiff/{}.tiff'.format(json_file[:-4])):
            no_files.append(json_file[:-4])
    
    json_list = [x[:-4] for x in json_list if x[:-4] not in no_files]
    length = len(json_list)
    train_len = int(length * 0.8)
    valid_len = int(length * 0.07)
    test_len = length - train_len - valid_len
    # train list
    train_index = np.random.choice(length,train_len,replace=False)
    train_list = [json_list[x] for x in train_index]
    remained_list = [x for x in json_list if x not in train_list]
    # valid list
    valid_index = np.random.choice(len(remained_list),valid_len,replace=False)
    valid_list = [remained_list[x] for x in valid_index]
    # test list
    test_list = [x for x in remained_list if x not in valid_list]
    with open('./data_split.json','w') as jf:
        json.dump({'train':train_list,'valid':valid_list,'test':test_list},jf)
    
    print(len(train_list),len(valid_list),len(test_list))
# split_dataset()

############################### generate label
def annotation_seq():

    json_list = os.listdir('../labels/dense_seq')
    time00 = time.time()
    length = len(json_list)
    for j_idx, json_file in enumerate(json_list):
        remove_current_patch = False
        # json_file = '000212_42.json'
        binary_label = np.zeros((1000,1000))
        orientation_map = np.zeros((1000,1000))
        with open('../json_merged/{}'.format(json_file),'r') as jf:
            json_data = json.load(jf)
        annotation_seq = []
        for segment in json_data:
            instance_list = []
            for ii, v in enumerate(segment[1:]):
                v1 = [max(0,min(999,int(round(segment[ii][0])))),max(0,min(999,int(round(segment[ii][1]))))]
                v2 = [max(0,min(999,int(round(v[0])))),max(0,min(999,int(round(v[1]))))]
                if not(v1[0]==v2[0] and v1[1]==v2[1]):
                    instance_list.append([v1,v2])
            if len(instance_list):
                annotation_seq.append({'init_vertex':instance_list[0],'end_vertex':instance_list[-1],
                                'seq':instance_list})

        if len(annotation_seq):
            with open('../labels/annotation_seq/{}'.format(json_file),'w') as jf:
                json.dump(annotation_seq,jf)
        time_used = time.time() - time00
        time_remained = time_used / (j_idx+1) * (length-j_idx)
        print('Remained time: ',time_remained/3600,'h || ',j_idx,'/',length)
        # break
# annotation_seq()


def clean_file():
    with open('../data_split.json','r') as jf:
        data_list = json.load(jf)
    
    data_list = data_list['train'] + data_list['valid'] + data_list['test']
    #
    data_dir = '../labels/annotation_seq'
    for j_idx, data_file in enumerate(os.listdir(data_dir)):
        if data_file[:-5] not in data_list:
            os.remove(os.path.join(data_dir,data_file))

    data_dir = '../labels/dense_seq'
    for j_idx, data_file in enumerate(os.listdir(data_dir)):
        if data_file[:-5] not in data_list:
            os.remove(os.path.join(data_dir,data_file))
    
# clean_file()

def instance_map():
    json_list = os.listdir('../labels/binary_map')
    time00 = time.time()
    length = len(json_list)
    def gkern():
        """\
        creates gaussian kernel with side length l and a sigma of sig
        """
        l=50
        sig=10
        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
        return kernel / np.max(kernel) 
    kernel = gkern()

    def gaussian_kernel(image,p):
        l = max(0,p[1]-25)
        r = min(999,p[1]+25)
        d = max(0,p[0]-25)
        u = min(999,p[0]+25)
        kl = max(0,25 - p[1])
        kr = min(50,999+25 - p[1])
        kd = max(0,25 - p[0])
        ku = min(50,999+25 - p[0])
        image[d:u,l:r] = image[d:u,l:r] + kernel[kd:ku,kl:kr]
        return image

    for j_idx, json_file in enumerate(json_list):
        # json_file = '020197_22.png'
        image = np.array(Image.open('../labels/binary_map/{}'.format(json_file)))[:,:,0]
        gt_labels = measure.label(image / 255,background=0)
        Image.fromarray((gt_labels).astype(np.uint8)).convert('RGB').save('../labels/instance_map/{}'.format(json_file))

        with open('../labels/dense_seq/{}'.format(json_file[:-3]+'json'),'r') as jf:
            data = json.load(jf)

        image = np.zeros((1000,1000))
        for d in data:
            init = d['init_vertex']
            end = d['end_vertex']
            image = gaussian_kernel(image,init)
            image = gaussian_kernel(image,end)
        Image.fromarray((image/np.max(image)*255).astype(np.uint8)).convert('RGB').save('../labels/endpoint_map/{}'.format(json_file))
        print(j_idx,len(json_list))
        # break

# instance_map()

def clean_test():
    with open('./data_split.json','r') as jf:
        data_list = json.load(jf)
    
    
    data_dir = '../labels/annotation_seq'
    print(len(os.listdir(data_dir)))
    print(len(data_list['train']))
    print(len(data_list['valid']))
    print(len(data_list['test']))
    test = [x for x in data_list['test'] if x+'.json' in os.listdir(data_dir)]
    # with open('../data_split.json','w') as jf:
    #     json.dump({'train':data_list['train'],'valid':data_list['valid'],'test':test},jf)

# clean_test()

def orien_debug():
    json_list = os.listdir('../labels/orientation_map')
    length = len(json_list)
    for j_idx, json_file in enumerate(json_list):
        image = np.array(Image.open('../labels/orientation_map/{}'.format(json_file)))
        if np.max(image)>63:
            print(json_file)
            print(np.max(image))
            error_map = (image>63)
            image = image*(1-error_map) + 63*error_map 
            Image.fromarray(image.astype(np.uint8)).convert('RGB').save('../labels/orientation_map/{}'.format(json_file))

# orien_debug()

def fix_split():
    with open('./data_split.json','r') as jf:
        data_list = json.load(jf)
    pretrain = data_list['train'][:10000]
    train = data_list['train'][10000:]
    with open('./data_split.json','w') as jf:
        json.dump({'train':train,'valid':data_list['valid'],'test':data_list['test'],'pretrain':pretrain},jf)
# fix_split()

def orien_anno_seq():
    json_list = os.listdir('../labels/annotation_seq')
    length = len(json_list)
    for j_idx, json_file in enumerate(json_list):
        list_json = []
        with open('../labels/annotation_seq/{}'.format(json_file),'r') as jf:
            data_list = json.load(jf)
        for data_instance in data_list:
            list_ins = []
            for v in data_instance['seq']:
                list_ins.append(v[0])
            list_ins.append(v[1])
            list_json.append({'init_vertex':list_ins[0],'end_vertex':list_ins[-1],'seq':list_ins})
        with open('./annotation_seq/{}'.format(json_file),'w') as jf:
            json.dump(list_json,jf)

        print(j_idx,len(json_list))
# orien_anno_seq()



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
    def find_neighbors(ii,points):
        output_v = []
        output_v.append(points[ii])
        output_v.append(points[ii+2])
        return output_v

    json_list = os.listdir('../labels/dense_seq')
    length = len(json_list)
    time00 = time.time()
    for j_idx, json_file in enumerate(json_list):
        # json_file='007182_03.json'
        with open(os.path.join('../labels/dense_seq',json_file),'r') as jf:
            json_data = json.load(jf)
        output_json = []
        img = Image.open('../labels/binary_map/{}'.format(json_file[:-4]+'png'))
        draw = ImageDraw.Draw(img)
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
            
            for point in vertices:
                draw.ellipse((point[1]-2,point[0]-2,point[1]+2,point[0]+2),fill='yellow',outline='yellow')
            for v in graph.sampled_vertices:
                for u in v.sampled_neighbors:
                    draw.line((v.coord[1],v.coord[0],u.coord[1],u.coord[0]),fill='yellow')
        img.save(os.path.join('./',json_file[:-4]+'png'))
        with open(os.path.join('../labels/sampled_seq',json_file[:-4]+'pickle'),'wb') as jf:
            pickle.dump(output_json,jf)
        time_used = time.time() - time00
        time_remained = time_used / (j_idx+1) * (length-j_idx)
        print('Remained time: ',time_remained/3600,'h || ',j_idx,'/',length)
# generate_graph()
