import time
import numpy as np
import torch
import math
import os
import random
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss, L1Loss, BCEWithLogitsLoss
from scipy.spatial import cKDTree
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage import measure

from utils.dataset import DatasetiCurb,DatasetDagger
from models.models_encoder import FPN
from models.models_decoder import DecoderCoord, DecoderStop

class FrozenClass():
        __isfrozen = False
        def __setattr__(self, key, value):
            if self.__isfrozen and not hasattr(self, key):
                raise TypeError( "%r is a frozen class" % self )
            object.__setattr__(self, key, value)

        def _freeze(self):
            self.__isfrozen = True

class Environment(FrozenClass):
    def __init__(self,args):
        self.args = args
        self.crop_size = 63
        self.agent = Agent(self)
        self.network = Network(self)
        # recordings
        self.training_image_number = self.args.epochs * self.network.train_len()
        self.graph_record = torch.zeros(1,1,1000,1000).to(args.device)
        self.time_start = time.time()
        # ===================parameters===================
        self.training_step = 0
        self.epoch_counter = 0
        self.DAgger_buffer_size = 2048
        self.DAgger_buffer = []
        self.DAgger_buffer_index = 0
        self.init_point_set = []
        self.setup_seed(20)

        self._freeze()

    def setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        
    def init_image(self,valid=False):
        self.graph_record = torch.zeros(1,1,1000,1000).to(self.args.device)
    
    def update_DAgger_buffer(self,data):
        if len(self.DAgger_buffer) < self.DAgger_buffer_size:
            self.DAgger_buffer.append(data)
            self.DAgger_buffer_index += 1
        else:
            if self.DAgger_buffer_index >= self.DAgger_buffer_size:
                self.DAgger_buffer_index = 0
            self.DAgger_buffer[self.DAgger_buffer_index] = data
            self.DAgger_buffer_index += 1
            

    def update_graph(self,start_vertex,end_vertex,graph):
        start_vertex = np.array([int(start_vertex[0]),int(start_vertex[1])])
        end_vertex = np.array([int(end_vertex[0]),int(end_vertex[1])])
        instance_vertices = []
        p = start_vertex
        d = end_vertex - start_vertex
        N = np.max(np.abs(d))
        graph[:,:,start_vertex[0],start_vertex[1]] = 1
        graph[:,:,end_vertex[0],end_vertex[1]] = 1
        if N:
            s = d / (N)
            for i in range(0,N):
                p = p + s
                graph[:,:,int(round(p[0])),int(round(p[1]))] = 1
    
    def expert_restricted_exploration(self,pre_coord,cropped_feature_tensor,orientation_map,correction=False):
        crop_size = self.crop_size
        # coord convert
        pre_coord = self.agent.train2world(pre_coord.cpu().detach().numpy())
        # next vertex for updating
        v_next = pre_coord
        # initialization
        self.agent.taken_stop_action = 0
        self.agent.gt_stop_action = 0
        
        # generate the expert demonstration for coord prediction
        if len(self.agent.candidate_label_points):
            l,d,r,u,_,_ = self.agent.crop_info
            # load data
            candidate_label_points = self.agent.candidate_label_points.copy()
            candidate_label_points_index = candidate_label_points[:,2]
            candidate_label_points = candidate_label_points[:,:2]
            # filter points
            candidate_label_points_index_1 = candidate_label_points_index[:-1]
            candidate_label_points_index_2 = candidate_label_points_index[1:]
            # if only one candidate instances in the cropped region, all elements in delta should be 1
            delta = candidate_label_points_index_2 - candidate_label_points_index_1
            delta = (delta==1)
            filtered_index = [x for x in range(len(delta)) if delta[x]!=1]
            # if delta is not 1, it means a instance is cut into multiple pieces by the crop action
            # only save the piece that the current vertex can reach
            if len(filtered_index):
                candidate_label_points = candidate_label_points[:filtered_index[0]+1]
            # generate label
            tree = cKDTree(candidate_label_points)
            _, iis = tree.query([self.agent.v_now],k=[1])
            iis = iis[0][0]
            v_center = candidate_label_points[iis]
            candidate_label_points = candidate_label_points[iis:]
            # 
            cropped_ahead_points = (candidate_label_points[:,0]>=max(d,v_center[0]-15)) * (candidate_label_points[:,0]<min(u,v_center[0]+15)) * \
                            (candidate_label_points[:,1] >=max(l,v_center[1]-15)) * (candidate_label_points[:,1] <min(r,v_center[1]+15))
            points_index = np.where(cropped_ahead_points==1)
            candidate_label_points = candidate_label_points[points_index]
            if not len(candidate_label_points):
                self.agent.gt_stop_action = 1
                gt_coord = None
                if (np.linalg.norm(np.array(self.agent.v_now) - np.array(self.agent.end_vertex))<10):
                    self.agent.taken_stop_action = 1
                    gt_coord = self.agent.end_vertex.copy()
                    self.agent.candidate_label_points = [gt_coord]
            else:
                orientation = [orientation_map[int(x[0]),int(x[1])] for x in candidate_label_points]
                ori_now = orientation_map[int(v_center[0]),int(v_center[1])]
                if ori_now > 1:
                    ori_now_left = ori_now - 1
                else:
                    ori_now_left = 64
                if ori_now != 64:
                    ori_now_right = ori_now + 1
                else:
                    ori_now_right = 1
                # locate corner pixels (pixels whose orientation is +/- 5 degree)
                gt_candiate = np.where((orientation!=ori_now)*(orientation!=ori_now_left)*(orientation!=ori_now_right))[0]
                if len(gt_candiate):
                    gt_coord = candidate_label_points[min(len(candidate_label_points)-1,min(gt_candiate)+5)]
                else:
                    gt_coord = candidate_label_points[len(candidate_label_points)-1]
                
                dd = np.linalg.norm(gt_coord - pre_coord)
                gt_index = next((i for i, val in enumerate(self.agent.instance_vertices) if np.all(val==gt_coord)), -1)
                if (self.agent.ii <= self.agent.pre_ii):
                    self.agent.gt_stop_action = 1
                self.agent.pre_ii = self.agent.ii
                self.agent.ii = gt_index  

                if dd > 15:
                    v_next = gt_coord.copy()
        else:
            self.agent.gt_stop_action = 1
            # whether reach the end vertex
            gt_coord = None
            if (np.linalg.norm(np.array(self.agent.v_now) - np.array(self.agent.end_vertex))<10):
                self.agent.taken_stop_action = 1
                gt_coord = self.agent.end_vertex.copy()
                self.agent.candidate_label_points = [gt_coord]
        if gt_coord is not None:
            gt_coord[0] -= self.agent.crop_info[4]
            gt_coord[1] -= self.agent.crop_info[5]
            gt_coord = [x/(self.crop_size//2) for x in gt_coord]
        else:
            gt_coord = [-3,-3]
        self.update_graph(self.agent.v_now,v_next,self.graph_record)
        # save data
        v_now_save = [x/1000 for x in self.agent.v_now]
        v_previous_save = [x/1000 for x in self.agent.v_previous]
        stored_data =  {
            'cropped_feature_tensor':cropped_feature_tensor,
            'gt_coord':gt_coord,
            'gt_stop_action':self.agent.gt_stop_action,
            'v_now':v_now_save,
            'v_previous':v_previous_save}

        self.update_DAgger_buffer(stored_data)
        
        # update
        self.agent.v_previous = self.agent.v_now
        self.agent.v_now = v_next
        


    def expert_free_exploration(self,pre_coord,cropped_feature_tensor,orientation_map,correction=False):
        crop_size = self.crop_size
        # coord convert
        pre_coord = self.agent.train2world(pre_coord.cpu().detach().numpy())
        # next vertex for updating
        v_next = pre_coord
        # initialization
        self.agent.taken_stop_action = 0
        self.agent.gt_stop_action = 0
        
        # generate the expert demonstration for coord prediction
        if len(self.agent.candidate_label_points):
            l,d,r,u,_,_ = self.agent.crop_info
            # load data
            candidate_label_points = self.agent.candidate_label_points.copy()
            candidate_label_points_index = candidate_label_points[:,2]
            candidate_label_points = candidate_label_points[:,:2]
            # filter points
            candidate_label_points_index_1 = candidate_label_points_index[:-1]
            candidate_label_points_index_2 = candidate_label_points_index[1:]
            # if only one candidate instances in the cropped region, all elements in delta should be 1
            delta = candidate_label_points_index_2 - candidate_label_points_index_1
            delta = (delta==1)
            filtered_index = [x for x in range(len(delta)) if delta[x]!=1]
            # if delta is not 1, it means a instance is cut into multiple pieces by the crop action
            # only save the piece that the current vertex can reach
            if len(filtered_index):
                candidate_label_points = candidate_label_points[:filtered_index[0]+1]
            # generate label
            tree = cKDTree(candidate_label_points)
            _, iis = tree.query([self.agent.v_now],k=[1])
            iis = iis[0][0]
            v_center = candidate_label_points[iis]
            candidate_label_points = candidate_label_points[iis:]
            # 
            cropped_ahead_points = (candidate_label_points[:,0]>=max(d,v_center[0]-15)) * (candidate_label_points[:,0]<min(u,v_center[0]+15)) * \
                            (candidate_label_points[:,1] >=max(l,v_center[1]-15)) * (candidate_label_points[:,1] <min(r,v_center[1]+15))
            points_index = np.where(cropped_ahead_points==1)
            candidate_label_points = candidate_label_points[points_index]
            if not len(candidate_label_points):
                self.agent.gt_stop_action = 1
                gt_coord = None
                if (np.linalg.norm(np.array(self.agent.v_now) - np.array(self.agent.end_vertex))<10):
                    self.agent.taken_stop_action = 1
                    gt_coord = self.agent.end_vertex.copy()
                    self.agent.candidate_label_points = [gt_coord]
            else:
                orientation = [orientation_map[int(x[0]),int(x[1])] for x in candidate_label_points]
                ori_now = orientation_map[int(v_center[0]),int(v_center[1])]
                if ori_now > 1:
                    ori_now_left = ori_now - 1
                else:
                    ori_now_left = 64
                if ori_now != 64:
                    ori_now_right = ori_now + 1
                else:
                    ori_now_right = 1
                # locate corner pixels (pixels whose orientation is +/- 5 degree)
                gt_candiate = np.where((orientation!=ori_now)*(orientation!=ori_now_left)*(orientation!=ori_now_right))[0]
                if len(gt_candiate):
                    gt_coord = candidate_label_points[min(len(candidate_label_points)-1,min(gt_candiate)+5)]
                else:
                    gt_coord = candidate_label_points[len(candidate_label_points)-1]
                
                # dd = np.linalg.norm(gt_coord - pre_coord)
                gt_index = next((i for i, val in enumerate(self.agent.instance_vertices) if np.all(val==gt_coord)), -1)
                if (self.agent.ii <= self.agent.pre_ii):
                    self.agent.gt_stop_action = 1
                self.agent.pre_ii = self.agent.ii
                self.agent.ii = gt_index  

                if correction and self.epoch_counter==1:
                    beta = 0.5**(self.training_step/1000)
                    v_next = (np.array(v_next) * (1 - beta) + np.array(gt_coord) * beta)
                    v_next = [int(v_next[0]),int(v_next[1])]
        else:
            self.agent.gt_stop_action = 1
            # whether reach the end vertex
            gt_coord = None
            if (np.linalg.norm(np.array(self.agent.v_now) - np.array(self.agent.end_vertex))<10):
                self.agent.taken_stop_action = 1
                gt_coord = self.agent.end_vertex.copy()
                self.agent.candidate_label_points = [gt_coord]
        if gt_coord is not None:
            gt_coord[0] -= self.agent.crop_info[4]
            gt_coord[1] -= self.agent.crop_info[5]
            gt_coord = [x/(self.crop_size//2) for x in gt_coord]
        else:
            gt_coord = [-3,-3]
        self.update_graph(self.agent.v_now,v_next,self.graph_record)
        # save data
        v_now_save = [x/1000 for x in self.agent.v_now]
        v_previous_save = [x/1000 for x in self.agent.v_previous]
        stored_data =  {
            'cropped_feature_tensor':cropped_feature_tensor,
            'gt_coord':gt_coord,
            'gt_stop_action':self.agent.gt_stop_action,
            'v_now':v_now_save,
            'v_previous':v_previous_save}

        self.update_DAgger_buffer(stored_data)
        
        # update
        self.agent.v_previous = self.agent.v_now
        self.agent.v_now = v_next

    def remove_duplicate_init_points(self,v):
        for u in self.init_point_set:
            dis = np.linalg.norm(np.array(u)-np.array(v))
            if dis < 10:
                self.init_point_set.remove(u)

class Agent(FrozenClass):
    def __init__(self,env):
        self.env = env
        self.args = env.args
        # state
        self.v_now = [0,0]
        self.v_previous = [0,0]
        self.taken_stop_action = 0
        self.gt_stop_action = 0
        self.agent_step_counter = 0
        self.local_loop_repeat_counter = 0
        self.empty_crop_repeat_counter = 0
        #
        self.instance_vertices = np.array([])
        self.candidate_label_points = np.array([])
        self.tree = None
        self.crop_info = []
        self.init_vertex = [0,0]
        self.end_vertex = [0,0]
        #
        self.ii = 0
        self.pre_ii = -1
        self._freeze()

    def init_agent(self,init_vertex):
        self.taken_stop_action = 0
        self.gt_stop_action = 0
        self.agent_step_counter = 0
        self.v_now = init_vertex
        self.v_previous = init_vertex
        self.ii = 0
        self.pre_ii = -1
        self.local_loop_repeat_counter = 0
        self.empty_crop_repeat_counter = 0
        
    def train2world(self,coord_in,crop_info=None):
        if crop_info is None:
            crop_info = self.crop_info    
        crop_size = self.env.crop_size
        pre_coord = [int(x*(crop_size//2)) for x in coord_in]
        pre_coord[0] += crop_info[4] 
        pre_coord[1] += crop_info[5]
        pre_coord = [max(min(pre_coord[0],crop_info[3]-1),crop_info[1]),max(min(pre_coord[1],crop_info[2]-1),crop_info[0])]
        return pre_coord

    def crop_attention_region(self,fpn_feature_tensor,val_flag=False):
        r'''
            Crop the current attension region centering at v_now.
        '''
        crop_size = self.env.crop_size
        # find left, right, up and down positions
        l = self.v_now[1]-crop_size//2
        r = self.v_now[1]+crop_size//2+1
        d = self.v_now[0]-crop_size//2
        u = self.v_now[0]+crop_size//2+1
        crop_l, crop_r, crop_d, crop_u = 0, self.env.crop_size, 0, self.env.crop_size
        if l<0:
            crop_l = -l
        if d<0:
            crop_d = -d
        if r>1000:
            crop_r = crop_r-r+1000
        if u>1000:
            crop_u = crop_u-u+1000
        crop_l,crop_r,crop_u,crop_d = int(crop_l),int(crop_r),int(crop_u),int(crop_d)
        l,r,u,d = max(0,min(1000,int(l))),max(0,min(1000,int(r))),max(0,min(1000,int(u))),max(0,min(1000,int(d)))
        self.crop_info = [l,d,r,u,self.v_now[0],self.v_now[1]]
        # cropped feature tensor for iCurb
        cropped_feature_tensor = torch.zeros(1,8,self.env.crop_size,self.env.crop_size)
        cropped_graph = torch.zeros(1,1,self.env.crop_size,self.env.crop_size)
        cropped_feature_tensor[:,:,crop_d:crop_u,crop_l:crop_r] = fpn_feature_tensor[:,:,d:u,l:r]
        cropped_graph[:,:,crop_d:crop_u,crop_l:crop_r] = self.env.graph_record[:,:,d:u,l:r]
        cropped_feature_tensor = torch.cat([cropped_feature_tensor,cropped_graph],dim=1).detach()
        # update the gt pixels within the cropped region
        if not val_flag:
            ahead_points = self.instance_vertices[self.ii:]
            ahead_points = np.array([[x[0],x[1],i] for i,x in enumerate(ahead_points)])
            cropped_ahead_points = (ahead_points[:,0]>=d) * (ahead_points[:,0]<u) * (ahead_points[:,1] >=l) * (ahead_points[:,1] <r)
            points_index = np.where(cropped_ahead_points==1)
            cropped_ahead_points = ahead_points[points_index]
            self.candidate_label_points = cropped_ahead_points#[x for x in cropped_ahead_points if (((x[0] - self.v_now[0])**2 + (x[1]- self.v_now[1])**2)**0.5>15)]
            # if len(self.candidate_label_points):
            #     self.tree = cKDTree(self.candidate_label_points)
            # else:W
            #     self.tree = None
        return cropped_feature_tensor.to(self.args.device)
    
    

class Network(FrozenClass):
    def __init__(self,env):
        self.env = env
        self.args = env.args
        # initialization
        self.encoder = FPN()
        self.decoder_coord = DecoderCoord(visual_size=pow(math.ceil(self.env.crop_size/8),2)*32+4)
        self.decoder_stop = DecoderStop(visual_size=pow(math.ceil(self.env.crop_size/8),2)*32+4)
        self.encoder.to(device=self.args.device)
        self.decoder_coord.to(device=self.args.device)
        self.decoder_stop.to(device=self.args.device)
        # tensorboard
        if not self.args.test:
            self.writer = SummaryWriter('./records/tensorboard')
        # ====================optimizer=======================
        self.optimizer_enc = optim.Adam(list(self.encoder.parameters()), lr=self.args.lr_rate, weight_decay=self.args.weight_decay)
        self.optimizer_coord_dec = optim.Adam(list(self.decoder_coord.parameters()), lr=self.args.lr_rate, weight_decay=self.args.weight_decay)
        self.optimizer_flag_dec = optim.Adam(list(self.decoder_stop.parameters()), lr=self.args.lr_rate, weight_decay=self.args.weight_decay)
        # =====================init losses=======================
        criterion_l1 = L1Loss(reduction='mean')
        criterion_bce = BCEWithLogitsLoss()
        criterion_ce = CrossEntropyLoss()
        self.criterions = {"ce":criterion_ce,'l1':criterion_l1,"bce": criterion_bce}
        # =====================Load data========================
        dataset_train = DatasetiCurb(self.args,mode="train")
        dataset_valid = DatasetiCurb(self.args,mode="valid")
        self.dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True,collate_fn=self.iCurb_collate)
        self.dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False,collate_fn=self.iCurb_collate)
        print("Dataset splits -> Train: {} | Valid: {}\n".format(len(dataset_train), len(dataset_valid)))
        self.loss = 0
        self.best_f1 = 0
        #
        self.load_checkpoints()
        self._freeze()

    def load_checkpoints(self):
        self.encoder.load_state_dict(torch.load('./checkpoints/seg_pretrain.pth',map_location='cpu'))
        if self.args.test:
            self.decoder_coord.load_state_dict(torch.load("./checkpoints/decoder_nodis_coord_best.pth",map_location='cpu'))
            self.decoder_stop.load_state_dict(torch.load("./checkpoints/decoder_nodis_flag_best.pth",map_location='cpu'))
            print('=============')
            print('Successfully loading iCurb checkpoints!')
        
        # self.decoder_seg.load_state_dict(torch.load('./dataset/pretrain_mask_decoder_19.pth',map_location='cpu'))
        print('=============')
        print('Pretrained FPN encoder checkpoint loaded!')
    
    def train_mode(self):
        self.decoder_coord.train()
        self.decoder_stop.train()
    
    def val_mode(self):
        self.decoder_coord.eval()
        self.decoder_stop.eval()
    
    def train_len(self):
        return len(self.dataloader_train)

    def val_len(self):
        return len(self.dataloader_valid)

    def bp(self):
        self.optimizer_coord_dec.zero_grad()
        self.optimizer_flag_dec.zero_grad()
        self.loss.backward()
        self.optimizer_flag_dec.step()
        self.optimizer_coord_dec.step()
        self.loss = 0

    def save_checkpoints(self,i):
        print('Saving checkpoints {}.....'.format(i))
        torch.save(self.decoder_coord.state_dict(), "./checkpoints/decoder_nodis_coord_best.pth")
        torch.save(self.decoder_stop.state_dict(), "./checkpoints/decoder_nodis_flag_best.pth")


    def DAgger_collate(self,batch):
        # variables as tensor
        cat_tiff = torch.stack([x[0] for x in batch])
        v_now = torch.stack([x[1] for x in batch])
        v_previous = torch.stack([x[2] for x in batch])
        gt_coord = torch.stack([x[3] for x in batch])
        gt_stop_action = torch.stack([x[-1] for x in batch]).reshape(-1)
        
        return cat_tiff, v_now, v_previous, gt_coord, gt_stop_action

    def iCurb_collate(self,batch):
        # variables as numpy
        seq = np.array([x[0] for x in batch])
        mask = np.array([x[3] for x in batch])
        orientation_map = np.array([x[4] for x in batch])
        # variables as list
        seq_lens = [x[1] for x in batch]
        image_name = [x[5] for x in batch]
        init_points = [x[6] for x in batch]
        end_points = [x[7] for x in batch]
        # variables as tensor
        tiff = torch.stack([x[2] for x in batch])
        return seq, seq_lens, tiff, mask, orientation_map, image_name, init_points, end_points

