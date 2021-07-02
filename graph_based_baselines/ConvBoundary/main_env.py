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

from utils.dataset import DatasetConvBoundary
from models.FPN import *
from models.loss import *


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
        self.ROI_width = 32
        self.ROI_length = 64
        self.agent = Agent(self)
        self.network = Network(self)
        # recordings
        self.training_image_number = self.args.epochs * self.network.train_len()
        self.graph_record = torch.zeros(1,1,1000,1000).to(args.device)
        self.time_start = time.time()
        # ===================parameters===================
        self.training_step = 0
        self.epoch_counter = 0
        self.buffer = {
                        'pre_coord':[],
                        'gt_coord':[],
                        }
        self.gaussian_kernel = self.gkern()
        self.buffer_size = args.batch_size
        
        yy, xx = np.meshgrid(range(self.ROI_length),range(self.ROI_width))
        ROI_grid = np.array((yy.ravel(), xx.ravel()))
        self.ROI_grid_homo = np.concatenate((ROI_grid,np.ones((1,ROI_grid.shape[1]))),axis=0)
        self.ROI_grid = ROI_grid.astype(int)
        self.ROI_trans_matrix = None
        self.clockwise_turn = np.array([[0,1],
                                        [-1,0]])
        self.counter_clockwise_turn = np.array([[0,-1],
                                        [1,0]])

        self.setup_seed(20)

        self._freeze()

    def setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def gkern(self):
        """
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

    def update_graph(self,start_vertex,end_vertex,graph,train_mode=True):
        start_vertex = np.array([int(start_vertex[0]),int(start_vertex[1])])
        end_vertex = np.array([int(end_vertex[0]),int(end_vertex[1])])
        all_points = []
        p = start_vertex
        d = end_vertex - start_vertex
        N = np.max(np.abs(d))
        graph[:,:,start_vertex[0],start_vertex[1]] = 1
        graph[:,:,end_vertex[0],end_vertex[1]] = 1
        if N:
            s = d / (N)
            for i in range(0,N):
                p = p + s
                graph[:,:,int(p[0]),int(p[1])] = 1
    
    def expert_exploration(self,pre_coord,teacher_forcing=False):
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
        # gt_coord_map = torch.FloatTe

        

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
        self.direction = [0,0]
        self.copy_points = None
        self.ahead_points = None
        #
        self.ii = 0
        self.pre_ii = 0
        #
        self.RNN_DH_length = 5
        self.RNN_DH = [torch.zeros(1,10,self.env.crop_size,self.env.crop_size).to(self.args.device) for x in range(self.RNN_DH_length)]
        self.RNN_ROI_length = 5
        self.RNN_ROI = [torch.zeros(1,10,self.env.ROI_length,self.env.ROI_width).to(self.args.device) for x in range(self.RNN_ROI_length)]
        self._freeze()

    def init_agent(self,init_vertex):
        self.taken_stop_action = 0
        self.gt_stop_action = 0
        self.agent_step_counter = 0
        self.v_now = init_vertex
        self.direction = [0,0]
        # vector pointing into the image
        self.v_previous = [0,0]
        if init_vertex[0] == 0:
            self.v_previous[0] = init_vertex[0] - 1
        elif init_vertex[0] == 999:
            self.v_previous[0] = init_vertex[0] + 1
        if init_vertex[1] == 0:
            self.v_previous[1] = init_vertex[1] - 1
        elif init_vertex[1] == 999:
            self.v_previous[1] = init_vertex[1] + 1

        self.ii = 0
        self.pre_ii = 0
        self.local_loop_repeat_counter = 0
        self.empty_crop_repeat_counter = 0
        self.env.buffer = {
                        'pre_coord':[],
                        'gt_coord':[],
                        }
        self.RNN_DH = [torch.zeros(1,10,self.env.crop_size,self.env.crop_size).to(self.args.device) for x in range(self.RNN_DH_length)]
        self.RNN_ROI = [torch.zeros(1,10,self.env.ROI_length,self.env.ROI_width).to(self.args.device) for x in range(self.RNN_ROI_length)]

    def train2world(self,coord_map):
        l,d,r,u,vr,vc = self.crop_info
        coord_map = torch.sigmoid(coord_map)
        coord_map = coord_map.cpu().detach().numpy()[0,0]
        if np.max(coord_map):
            coord_map = coord_map / max(0.5,np.max(coord_map))
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
        cropped_feature_tensor = torch.zeros(1,3,self.env.crop_size,self.env.crop_size)
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
        self.ConvBoundarySeg = FPNSeg()
        self.ConvBoundaryAgent = FPNAgent(self.args.device)
        self.ConvBoundarySeg.to(device=self.args.device)
        self.ConvBoundaryAgent.to(device=self.args.device)
        # tensorboard
        if not self.args.test:
            self.writer = SummaryWriter('./records/tensorboard')
        # ====================optimizer=======================
        self.optimizer_seg = optim.Adam(list(self.ConvBoundarySeg.parameters()), lr=self.args.lr_rate, weight_decay=self.args.weight_decay)
        self.optimizer_agent = optim.Adam(list(self.ConvBoundaryAgent.parameters()), lr=self.args.lr_rate, weight_decay=self.args.weight_decay)
        # =====================init losses=======================
        criterion_l1 = L1Loss(reduction='mean')
        criterion_bce = BCEWithLogitsLoss()
        criterion_ce = CrossEntropyLoss()
        self.criterions = {"ce":criterion_ce,'l1':criterion_l1,"bce": criterion_bce,'cos':cos_loss()}
        # =====================Load data========================
        self.dataset_train = DatasetConvBoundary(self.args,mode='train')
        dataset_valid = DatasetConvBoundary(self.args,mode="valid")
        self.dataloader_train = DataLoader(self.dataset_train, batch_size=1, shuffle=True,collate_fn=self.ConvBoundary_collate)
        self.dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False,collate_fn=self.ConvBoundary_collate)
        print("Dataset modes -> Train: {} | Valid: {}\n".format(len(self.dataset_train), len(dataset_valid)))
        #================recorded list for backpropagation==============
        self.best_f1 = 0
        self.load_checkpoints()
        self._freeze()

    def load_checkpoints(self):
        self.ConvBoundarySeg.load_state_dict(torch.load('./checkpoints/convBoundary_seg_pretrain.pth',map_location='cpu'))
        if self.args.test:
            self.ConvBoundarySeg.load_state_dict(torch.load('./checkpoints/ConvBoundary_seg_best.pth',map_location='cpu'))
            self.ConvBoundaryAgent.load_state_dict(torch.load('./checkpoints/ConvBoundary_agent_best.pth',map_location='cpu'))
            print('----------------------Best checkpoint loaded---------------------')

    def train_mode(self):
        self.ConvBoundarySeg.train()
        self.ConvBoundaryAgent.train()
    
    def val_mode(self):
        self.ConvBoundarySeg.eval()
        self.ConvBoundaryAgent.eval()
    
    def train_len(self):
        return len(self.dataloader_train)

    def val_len(self):
        return len(self.dataloader_valid)

    def bp(self,loss,seg=False):
        if not seg:
            self.optimizer_agent.zero_grad()
            loss.backward()
            self.optimizer_agent.step()
        else:
            self.optimizer_seg.zero_grad()
            loss.backward()
            self.optimizer_seg.step()

    def save_checkpoints(self,i):
        print('Saving checkpoints {}.....'.format(i))
        torch.save(self.ConvBoundarySeg.state_dict(), "./checkpoints/ConvBoundary_seg_best.pth")
        torch.save(self.ConvBoundaryAgent.state_dict(), "./checkpoints/ConvBoundary_agent_best.pth")
        print('Evaluation......')

    def ConvBoundary_collate(self,batch):
        # variables as numpy
        seq = np.array([x[0] for x in batch])
        mask = np.array([x[3] for x in batch])
        # variables as list
        seq_lens = [x[1] for x in batch]
        image_name = [x[6] for x in batch]
        init_points = [x[7] for x in batch]
        end_points = [x[8] for x in batch]
        # variables as tensor
        tiff = torch.stack([x[2] for x in batch])
        inv = torch.FloatTensor([x[4] for x in batch])
        direction = torch.FloatTensor([x[5] for x in batch])
        return seq, seq_lens, tiff, mask, inv, direction, image_name, init_points, end_points

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