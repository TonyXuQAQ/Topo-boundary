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

from utils.dataset import DatasetVecRoad
from models.vecRoadNet import VecRoadNet


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
        self.agent = Agent(self)
        self.network = Network(self)
        # recordings
        self.training_image_number = self.args.epochs * self.network.train_len()
        self.graph_record = torch.zeros(1,1,1000,1000).to(args.device)
        self.time_start = time.time()
        # ===================parameters===================
        self.training_step = 0
        self.epoch_counter = 0
        self.buffer = {'pre_coord':[],'pre_seg':[],'gt_coord':[],'gt_seg':[]}
        self.buffer_size = self.args.batch_size
        self.gaussian_kernel = self.gkern()
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
    
    def expert_exploration(self,pre_coord,cropped_feature_tensor,teacher_forcing=False):
        # coord convert
        pre_coord = self.agent.train2world(pre_coord)
        # next vertex for updating
        v_next = pre_coord
        if v_next is None:
            if not teacher_forcing:
                self.agent.taken_stop_action = 1
                self.agent.gt_stop_action = 1
                gt_coord_map = None
            else:
                if len(self.agent.candidate_label_points):
                    gt_coord = self.agent.candidate_label_points[0].copy()
                    gt_index = next((i for i, val in enumerate(self.agent.instance_vertices) if np.all(val==gt_coord)), -1)
                    self.agent.ii = gt_index
                else:
                    gt_coord = self.agent.end_vertex
                v_next = gt_coord
                gt_coord_map = self.get_gt_map(gt_coord,v_now=self.agent.v_now)
                self.update_graph(self.agent.v_now,v_next,self.graph_record)
                # update
                self.agent.v_previous = self.agent.v_now
                self.agent.v_now = v_next
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
                if teacher_forcing:
                    v_next = gt_coord
                else:
                    dd = np.linalg.norm(gt_coord-v_next)
                    if dd > 60 and self.epoch_counter==1:
                        v_next = gt_coord
                gt_coord_map = self.get_gt_map(gt_coord,v_now=self.agent.v_now)
            else:
                # self.agent.taken_stop_action = 1
                self.agent.gt_stop_action = 1
                # whether reach the end vertex
                if (np.linalg.norm(np.array(self.agent.v_now) - np.array(self.agent.end_vertex)) <20):
                    if teacher_forcing:
                        v_next = self.agent.end_vertex
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
        Image.fromarray((gt_coord_map*255).astype(np.uint8)).convert('RGB').save('./test0.png')
        gt_coord_map = torch.FloatTensor(gt_coord_map).unsqueeze(0).unsqueeze(0)
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
        # self.env.buffer = {'pre_coord':[],'pre_seg':[],'gt_coord':[],'gt_seg':[]}

    def train2world(self,coord_map):
        l,d,r,u,vr,vc = self.crop_info
        coord_map = torch.sigmoid(coord_map)
        coord_map = coord_map.cpu().detach().numpy()[0,0]
        if np.max(coord_map):
            # print(np.max(coord_map))
            coord_map = coord_map / max(0.1,np.max(coord_map))
            Image.fromarray(coord_map*255).convert('RGB').save('./test2.png')
            coord_map = coord_map > 0.2
            Image.fromarray((coord_map*255).astype(np.uint8)).convert('RGB').save('./test.png')
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

    def crop_attention_region(self,fpn_feature_tensor,mask=None,val_flag=False):
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
        cropped_feature_tensor = torch.zeros(1,4,self.env.crop_size,self.env.crop_size)
        cropped_graph = torch.zeros(1,1,self.env.crop_size,self.env.crop_size)
        cropped_feature_tensor[:,:,crop_d:crop_u,crop_l:crop_r] = fpn_feature_tensor[:,:,d:u,l:r]
        cropped_graph[:,:,crop_d:crop_u,crop_l:crop_r] = self.env.graph_record[:,:,d:u,l:r]
        cropped_feature_tensor = torch.cat([cropped_feature_tensor,cropped_graph],dim=1).detach()
        if not val_flag:
            ahead_points = self.instance_vertices[self.ii:]
            cropped_ahead_points = (ahead_points[:,0]>=d) * (ahead_points[:,0]<u) * (ahead_points[:,1] >=l) * (ahead_points[:,1] <r)
            points_index = np.where(cropped_ahead_points==1)
            cropped_ahead_points = ahead_points[points_index]
            self.copy_points = cropped_ahead_points.copy()
            self.ahead_points = ahead_points.copy()
            self.candidate_label_points = [x for x in cropped_ahead_points if (((x[0] - self.v_now[0])**2 + (x[1]- self.v_now[1])**2)**0.5>25)]
            cropped_mask = np.zeros((self.env.crop_size,self.env.crop_size))
            cropped_mask[crop_d:crop_u,crop_l:crop_r] = mask[d:u,l:r]
            if len(self.candidate_label_points):
                self.tree = cKDTree(self.candidate_label_points)
                Image.fromarray((cropped_mask*255).astype(np.uint8)).convert('RGB').save('./test1.png')
            else:
                self.tree = None
            
            return cropped_feature_tensor.to(self.args.device), torch.FloatTensor(cropped_mask).to(self.args.device).unsqueeze(0).unsqueeze(0)
        else:
            return cropped_feature_tensor.to(self.args.device)

class Network(FrozenClass):
    def __init__(self,env):
        self.env = env
        self.args = env.args
        # initialization
        self.vecRoadNet = VecRoadNet()
        self.vecRoadNet.to(self.args.device)
        # tensorboard
        if not self.args.test:
            self.writer = SummaryWriter('./records/tensorboard')
        # ====================optimizer=======================
        self.optimizer_rt = optim.Adam(list(self.vecRoadNet.parameters()), lr=self.args.lr_rate, weight_decay=self.args.weight_decay)
        # =====================init losses=======================
        criterion_l1 = L1Loss(reduction='mean')
        criterion_bce = BCEWithLogitsLoss()
        criterion_ce = CrossEntropyLoss()
        self.criterions = {"ce":criterion_ce,'l1':criterion_l1,"bce": criterion_bce}
        # =====================Load data========================
        self.dataset_train = DatasetVecRoad(self.args,mode='train')
        dataset_valid = DatasetVecRoad(self.args,mode="valid")
        self.dataloader_train = DataLoader(self.dataset_train, batch_size=1, shuffle=True,collate_fn=self.vecRoad_collate)
        self.dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False,collate_fn=self.vecRoad_collate)
        print("Dataset modes -> Train: {} | Valid: {}\n".format(len(self.dataset_train), len(dataset_valid)))
        #================recorded list for backpropagation==============
        self.best_f1 = 0
        self.load_checkpoints()
        self._freeze()

    def load_checkpoints(self):
        if self.args.teacher_forcing_number < 0:
            self.vecRoadNet.load_state_dict(torch.load('./checkpoints/vecRoad_pretrain.pth',map_location='cpu'))
            print('----------------------Pretrain checkpoint loaded---------------------')
        if self.args.test:
            self.vecRoadNet.load_state_dict(torch.load('./checkpoints/vecRoad_best.pth',map_location='cpu'))
            print('----------------------Best checkpoint loaded---------------------')

    def train_mode(self):
        self.vecRoadNet.train()
    
    def val_mode(self):
        self.vecRoadNet.eval()
    
    def train_len(self):
        return len(self.dataloader_train)

    def val_len(self):
        return len(self.dataloader_valid)

    def bp(self,loss):
        self.optimizer_rt.zero_grad()
        loss.backward()
        self.optimizer_rt.step()

    def save_checkpoints(self,i):
        print('Saving checkpoints {}.....'.format(i))
        torch.save(self.vecRoadNet.state_dict(), "./checkpoints/vecRoad_best.pth")
        if self.env.epoch_counter == 1 and self.args.teacher_forcing_number > 0:
            torch.save(self.vecRoadNet.state_dict(), "./checkpoints/vecRoad_pretrain.pth")
        print('Evaluation......')

    def vecRoad_collate(self,batch):
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
