import time
import numpy as np
import torch
import os
import cv2
import random
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss, L1Loss, BCEWithLogitsLoss
from scipy.spatial import cKDTree
import torch.nn.functional as F
from PIL import Image, ImageDraw
from skimage import measure

from utils.dataset import DatasetDAGMapper
from models.DAGMapperNet import *


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
        self.crop_size = 128
        self.ROI_width = 30
        self.ROI_length = 50
        self.agent = Agent(self)
        self.network = Network(self)
        # recordings
        self.training_image_number = self.args.epochs * self.network.train_len()
        self.graph_record = torch.zeros(1,1,1000,1000).to(args.device)
        self.time_start = time.time()
        # ===================parameters===================
        self.training_step = 0
        self.epoch_counter = 0
        self.buffer = {'pre_direction':[],
                        'gt_direction':[],
                        'pre_coord':[],
                        'gt_coord':[],
                        'pre_state':[],
                        'gt_state':[]}
        self.buffer_size = args.batch_size
        self.gaussian_kernel = self.gkern()
        
        yy, xx = np.meshgrid(range(self.ROI_length),range(self.ROI_width))
        ROI_grid = np.array((yy.ravel(), xx.ravel()))
        self.ROI_grid_homo = np.concatenate((ROI_grid,np.ones((1,ROI_grid.shape[1]))),axis=0)
        self.ROI_grid = ROI_grid.astype(int)
        self.ROI_trans_matrix = None

        self.setup_seed(20)

        self._freeze()

    def setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def gkern(self):
        """\
        creates gaussian kernel with side length l and a sigma of sig
        """
        l=self.crop_size
        sig=5
        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
        return kernel / np.max(kernel) 

    def init_image(self,valid=False):
        self.graph_record = torch.zeros(1,1,1000,1000).to(self.args.device)

    
    def update_graph(self,start_vertex,end_vertex,graph,set_value=1):
        start_vertex = np.array([int(start_vertex[0]),int(start_vertex[1])])
        end_vertex = np.array([int(end_vertex[0]),int(end_vertex[1])])
        instance_vertices = []
        p = start_vertex
        d = end_vertex - start_vertex
        N = np.max(np.abs(d))
        graph[:,:,start_vertex[0],start_vertex[1]] = set_value
        graph[:,:,end_vertex[0],end_vertex[1]] = set_value
        if N:
            s = d / (N)
            for i in range(0,N):
                p = p + s
                graph[:,:,int(p[0]),int(p[1])] = set_value
        return graph
    
    def expert_labels(self,pre_coord,teacher_forcing=False):
        # coord convert
        pre_coord = self.agent.train2world(pre_coord)
        # next vertex for updating
        v_next = pre_coord
        
        if v_next is None:
            self.agent.taken_stop_action = 1
            self.agent.gt_stop_action = 1
            gt_coord_map = None
        else:
            # initialization
            self.agent.taken_stop_action = 0
            self.agent.gt_stop_action = 0
            # generate the expert demonstration for coord prediction
            if self.agent.tree:
                dd, ii = self.agent.tree.query([pre_coord],k=[1])
                dd = dd[0]
                ii = ii[0]
                gt_coord = self.agent.candidate_label_points[int(ii)].copy()
                gt_index = next((i for i, val in enumerate(self.agent.instance_vertices) if np.all(val==gt_coord)), -1)
                # update history points (past points)
                self.agent.ii = gt_index
                # teacher forcing in the first epoch
                dd = np.linalg.norm(gt_coord-v_next)
                if dd > 15 and self.epoch_counter==1:
                    v_next = gt_coord
                gt_coord_map = self.get_gt_map(gt_coord,v_now=self.agent.v_now)
                
            else:
                self.agent.gt_stop_action = 1
                # whether reach the end vertex
                if (np.linalg.norm(np.array(self.agent.v_now) - np.array(self.agent.end_vertex)) <20):
                    self.agent.taken_stop_action = 1
                    gt_coord_map = self.get_gt_map(v_next,v_now=self.agent.v_now)
                else:
                    gt_coord_map = None
            # update
            self.update_graph(self.agent.v_now,v_next,self.graph_record)
            self.agent.v_previous = self.agent.v_now
            self.agent.v_now = v_next
        return gt_coord_map
    
    def get_gt_map(self,gt_coord,v_now=None):
        _,_,_,_,vr,vc = self.agent.crop_info
        crop_l,crop_r,crop_u,crop_d = self.agent.window_info
        gt_coord = [int(gt_coord[0]-vr+(self.crop_size/2-1)),int(gt_coord[1]-vc+(self.crop_size/2-1))]
        # create gaussian map
        mask_gt_coord_map = np.zeros((self.crop_size,self.crop_size))
        mask_gt_coord_map[crop_d:crop_u,crop_l:crop_r] = 1
        gt_coord_map = self.gaussian_kernel.copy()
        num_rows, num_cols = gt_coord_map.shape[:2]
        translation_matrix = np.float32([[1,0,gt_coord[1]-(self.crop_size/2-1)], [0,1,gt_coord[0]-(self.crop_size/2-1)] ])
        map_translation = cv2.warpAffine(gt_coord_map, translation_matrix, (num_cols,num_rows))
        gt_coord_map = map_translation * mask_gt_coord_map
        # Image.fromarray((gt_coord_map*255).astype(np.uint8)).convert('RGB').save('./test0.png')
        return gt_coord_map

        

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
        self.ROI_candidate_label_points = np.array([])
        self.tree = None
        self.crop_info = []
        self.window_info = []
        self.init_vertex = [0,0]
        self.end_vertex = [0,0]
        self.copy_points = None
        self.ahead_points = None
        #
        self.ii = 0
        self.pre_ii = 0
        #
        self.RNN_DH_length = 3
        self.RNN_DH = [torch.zeros(1,10,self.env.crop_size,self.env.crop_size).to(self.args.device) for x in range(self.RNN_DH_length)]
        self.RNN_ROI_length = 3
        self.RNN_ROI = [torch.zeros(1,10,self.env.ROI_length,self.env.ROI_width).to(self.args.device) for x in range(self.RNN_ROI_length)]
        self._freeze()

    def init_agent(self,init_vertex):
        self.taken_stop_action = 0
        self.gt_stop_action = 0
        self.agent_step_counter = 0
        self.v_now = init_vertex
        self.v_previous = init_vertex
        self.ii = 0
        self.pre_ii = 0
        self.local_loop_repeat_counter = 0
        self.empty_crop_repeat_counter = 0
        self.env.buffer = {'pre_direction':[],
                        'gt_direction':[],
                        'pre_coord':[],
                        'gt_coord':[],
                        'pre_state':[],
                        'gt_state':[]}
        self.RNN_DH = [torch.zeros(1,10,self.env.crop_size,self.env.crop_size).to(self.args.device) for x in range(self.RNN_DH_length)]
        self.RNN_ROI = [torch.zeros(1,10,self.env.ROI_length,self.env.ROI_width).to(self.args.device) for x in range(self.RNN_ROI_length)]

    def train2world(self,coord_map):
        l,d,r,u,vr,vc = self.crop_info
        coord_map = torch.sigmoid(coord_map)
        coord_map = coord_map.cpu().detach().numpy()[0,0]
        if np.max(coord_map):
            coord_map = coord_map / max(0.1,np.max(coord_map))
            # Image.fromarray(coord_map*255).convert('RGB').save('./test2.png')
            coord_map = coord_map > 0.5
            # Image.fromarray((coord_map*255).astype(np.uint8)).convert('RGB').save('./test.png')
            labels = measure.label(coord_map, connectivity=2)
            props = measure.regionprops(labels)
            max_area = 8
            v_next = None
            for region in props:
                if region.area > max_area:
                    max_area = region.area
                    v_next = region.centroid
            if v_next:
                v_next = [int(v_next[0]+vr-(self.env.crop_size/2-1)),int(v_next[1]+vc-(self.env.crop_size/2-1))]
                v_next = [max(d,min(u-1,v_next[0])),max(l,min(r-1,v_next[1]))]
                return v_next
            else:
                return None
        else:
            return None
            
    def crop_attention_region(self,fpn_feature_tensor,val_flag=False):
        r'''
            Crop the current attension region centering at v_now.
        '''
        crop_size = self.env.crop_size
        # find left, right, up and down positions
        l = self.v_now[1]-crop_size//2+1
        r = self.v_now[1]+crop_size//2+1
        d = self.v_now[0]-crop_size//2+1
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
        self.window_info = [crop_l,crop_r,crop_u,crop_d]
        # cropped feature tensor for iCurb
        cropped_feature_tensor = torch.zeros(1,9,self.env.crop_size,self.env.crop_size)
        cropped_graph = torch.zeros(1,1,self.env.crop_size,self.env.crop_size)
        cropped_feature_tensor[:,:,crop_d:crop_u,crop_l:crop_r] = fpn_feature_tensor[:,:,d:u,l:r]
        cropped_graph[:,:,crop_d:crop_u,crop_l:crop_r] = self.env.graph_record[:,:,d:u,l:r]
        cropped_feature_tensor = torch.cat([cropped_feature_tensor,cropped_graph],dim=1).detach()
        if not val_flag:
            ahead_points = self.instance_vertices[self.ii:]
            cropped_ahead_points = (ahead_points[:,0]>=d) * (ahead_points[:,0]<u) * (ahead_points[:,1] >=l) * (ahead_points[:,1] <r)
            points_index = np.where(cropped_ahead_points==1)
            cropped_ahead_points = ahead_points[points_index]
            self.candidate_label_points = [x for x in cropped_ahead_points if (((x[0] - self.v_now[0])**2 + (x[1]- self.v_now[1])**2)**0.5>25)]
            if len(self.candidate_label_points):
                self.tree = cKDTree(self.candidate_label_points)
            else:
                self.tree = None
            
        return cropped_feature_tensor.to(self.args.device)

class Network(FrozenClass):
    def __init__(self,env):
        self.env = env
        self.args = env.args
        # initialization
        self.DAGMapperEncoder = DAGMapperEncoder()
        # state head
        self.DAGMapperSH = DAGMapperSH(self.env)
        # direciton head
        self.DAGMapperDH = DAGMapperDH(self.env)
        # position head
        self.DAGMapperPH = DAGMapperPH(self.env)
        # distance head
        self.DAGMapperDTH = DAGMapperDTH()
        self.DAGMapperEncoder.to(device=self.args.device)
        self.DAGMapperSH.to(device=self.args.device)
        self.DAGMapperDH.to(device=self.args.device)
        self.DAGMapperDTH.to(device=self.args.device)
        self.DAGMapperPH.to(device=self.args.device)
        # tensorboard
        if not self.args.test:
            self.writer = SummaryWriter('./records/tensorboard')
        # ====================optimizer=======================
        self.optimizer_dme = optim.Adam(list(self.DAGMapperEncoder.parameters()), lr=self.args.lr_rate, weight_decay=self.args.weight_decay)
        self.optimizer_dmdh = optim.Adam(list(self.DAGMapperDH.parameters()), lr=self.args.lr_rate, weight_decay=self.args.weight_decay)
        self.optimizer_dmdth = optim.Adam(list(self.DAGMapperDTH.parameters()), lr=self.args.lr_rate, weight_decay=self.args.weight_decay)
        self.optimizer_dmph = optim.Adam(list(self.DAGMapperPH.parameters()), lr=self.args.lr_rate, weight_decay=self.args.weight_decay)
        self.optimizer_dmsh = optim.Adam(list(self.DAGMapperSH.parameters()),lr=self.args.lr_rate,weight_decay=self.args.weight_decay)
        # =====================init losses=======================
        criterion_mse = MSELoss(reduction='mean')
        criterion_bce = BCEWithLogitsLoss()
        criterion_ce = CrossEntropyLoss()
        self.criterions = {"ce":criterion_ce,'mse':criterion_mse,"bce": criterion_bce,'cos':nn.CosineSimilarity()}
        # =====================Load data========================
        self.dataset_train = DatasetDAGMapper(self.args,mode='train')
        dataset_valid = DatasetDAGMapper(self.args,mode="valid")
        self.dataloader_train = DataLoader(self.dataset_train, batch_size=1, shuffle=False,collate_fn=self.DAGMapper_collate)
        self.dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False,collate_fn=self.DAGMapper_collate)
        print("Dataset modes -> Train: {} | Valid: {}\n".format(len(self.dataset_train), len(dataset_valid)))
        #================recorded list for backpropagation==============
        self.best_f1 = 0
        self.load_checkpoints()
        self._freeze()

    def load_checkpoints(self):
        if not self.args.test and not self.args.pretrain:
            self.DAGMapperEncoder.load_state_dict(torch.load('./checkpoints/DagMapper_encoder_pretrain.pth',map_location='cpu'))
            self.DAGMapperDTH.load_state_dict(torch.load('./checkpoints/DagMapper_DTH_pretrain.pth',map_location='cpu'))
            print('---------------- Segmentation pretrain loaded! ----------------')
        if self.args.test:
            self.DAGMapperEncoder.load_state_dict(torch.load('./checkpoints/DagMapper_encoder_best.pth',map_location='cpu'))
            self.DAGMapperDTH.load_state_dict(torch.load('./checkpoints/DagMapper_DTH_best.pth',map_location='cpu'))
            self.DAGMapperDH.load_state_dict(torch.load('./checkpoints/DagMapper_DH_best.pth',map_location='cpu'))
            self.DAGMapperSH.load_state_dict(torch.load('./checkpoints/DagMapper_SH_best.pth',map_location='cpu'))
            self.DAGMapperPH.load_state_dict(torch.load('./checkpoints/DagMapper_PH_best.pth',map_location='cpu'))
            print('----------------------Best checkpoint loaded---------------------')

    def train_mode(self):
        self.DAGMapperEncoder.train()
        self.DAGMapperDTH.train()
        self.DAGMapperPH.train()
        self.DAGMapperSH.train()
        self.DAGMapperDH.train()
    
    def val_mode(self):
        self.DAGMapperEncoder.eval()
        self.DAGMapperDTH.eval()
        self.DAGMapperPH.eval()
        self.DAGMapperSH.eval()
        self.DAGMapperDH.eval()
    
    def train_len(self):
        return len(self.dataloader_train)

    def val_len(self):
        return len(self.dataloader_valid)

    def bp(self,loss,seg=False):
        if not seg:
            self.optimizer_dmdh.zero_grad()
            self.optimizer_dmsh.zero_grad()
            self.optimizer_dmph.zero_grad()
            loss.backward()
            self.optimizer_dmdh.step()
            self.optimizer_dmsh.step()
            self.optimizer_dmph.step()
        else:
            self.optimizer_dme.zero_grad()
            self.optimizer_dmdth.zero_grad()
            loss.backward()
            self.optimizer_dme.step()
            self.optimizer_dmdth.step()

    def save_checkpoints(self,i):
        print('Saving checkpoints {}.....'.format(i))
        if self.env.args.pretrain:
            torch.save(self.DAGMapperEncoder.state_dict(), "./checkpoints/DagMapper_encoder_pretrain.pth")
            torch.save(self.DAGMapperDTH.state_dict(), "./checkpoints/DagMapper_DTH_pretrain.pth")
        else:
            torch.save(self.DAGMapperEncoder.state_dict(), "./checkpoints/DagMapper_encoder_best.pth")
            torch.save(self.DAGMapperDTH.state_dict(), "./checkpoints/DagMapper_DTH_best.pth")
            torch.save(self.DAGMapperPH.state_dict(), "./checkpoints/DagMapper_PH_best.pth")
            torch.save(self.DAGMapperSH.state_dict(), "./checkpoints/DagMapper_SH_best.pth")
            torch.save(self.DAGMapperDH.state_dict(), "./checkpoints/DagMapper_DH_best.pth")
            print('Evaluation......')

    def DAGMapper_collate(self,batch):
        # variables as numpy
        seq = np.array([x[0] for x in batch])
        mask = np.array([x[3] for x in batch])
        # variables as list
        seq_lens = [x[1] for x in batch]
        image_name = [x[4] for x in batch]
        init_points = [x[5] for x in batch]
        end_points = [x[6] for x in batch]
        # variables as tensor
        tiff = torch.stack([x[2] for x in batch])
        return seq, seq_lens, tiff, mask, image_name, init_points, end_points

    def buffer_collate(self,batch):
        # variables as numpy
        v_next = np.array([x[3] for x in batch])
        # variables as tensor
        pre_direction = torch.stack([x[0] for x in batch]).squeeze(1)
        gt_direction = torch.FloatTensor([x[1] for x in batch])
        pre_coord = torch.stack([x[2] for x in batch]).squeeze(1)
        pre_state = torch.stack([x[4] for x in batch]).squeeze(1)
        gt_state = torch.FloatTensor([x[5] for x in batch])
        return pre_direction, gt_direction, pre_coord, v_next, pre_state, gt_state