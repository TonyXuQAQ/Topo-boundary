import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tvf
from PIL import Image

import numpy as np
import json
import os 
import random

class DatasetDAGMapper(Dataset):    
    r'''
    DataLoader for sampling. Iterate the aerial images dataset
    '''
    def __init__(self,args, mode):
        #
        assert mode in {"train", "valid", "test"}
        #
        if args.test:
            mode = 'test'
        seq_path = args.seq_dir
        mask_path = args.mask_dir
        image_path = args.image_dir
        seq_list, mask_list, image_list = load_datadir(seq_path,mask_path,image_path,mode,args)
        self.args = args
        self.seq_len = len(seq_list)
        self.image_list = image_list
        self.seq_list = seq_list
        self.mask_list = mask_list
    
    def __len__(self):
        r"""
        :return: data length
        """
        return self.seq_len
    
    def __getitem__(self, idx):
        seq, seq_lens, init_points, end_points = load_seq(self.seq_list[idx])
        tiff, mask = load_image(self.image_list[idx],self.mask_list[idx])
        image_name = self.seq_list[idx]
        return seq, seq_lens, tiff, mask, image_name, init_points, end_points

class DatasetBuffer(Dataset):
    r'''
    DataLoader for training. Iterate the buffer.
    '''
    def __init__(self,data):
        self.data = data
        self.seq_len = len(data)

    def __len__(self):
        r"""
        :return: data length
        """
        return self.seq_len

    def __getitem__(self, idx):
        pre_direction = self.data[idx]['cropped_feature_tensor']
        gt_direction = self.data[idx]['v_now']
        pre_coord = self.data[idx]['v_previous']
        pre_coord_np = self.data[idx]['gt_stop_action']
        pre_state = self.data[idx]['crop_info']
        gt_state = self.data[idx]['ahead_vertices']
        return pre_direction, gt_direction, pre_coord, pre_coord_np, pre_state, gt_state

def load_datadir(seq_path,mask_path,image_path,mode,args):
    with open('./dataset/data_split.json','r') as jf:
        json_list = json.load(jf)
    train_list = json_list['train']
    test_list = json_list['test']
    val_list = json_list['valid']
    pretrain_list = json_list['pretrain']

    if mode == 'valid':
        json_list = [x+'.json' for x in val_list][:150]
    elif mode == 'test':
        json_list = [x+'.json' for x in test_list]
        random.shuffle(json_list)
    elif mode == 'train':
        if not args.pretrain:
            json_list = [x+'.json' for x in train_list]
        else:
            json_list = [x+'.json' for x in pretrain_list]

    seq_list = []
    image_list = []
    mask_list = []
    for jsonf in json_list:
        seq_list.append(os.path.join(seq_path,jsonf))
        mask_list.append(os.path.join(mask_path,jsonf[:-4] + 'png'))
        image_list.append(os.path.join(image_path,jsonf[:-4]+'tiff'))
    return seq_list, mask_list, image_list
    
def load_seq(seq_path):
    r''' 
    Load the dense sequence of the current image. It may contains the vertices of multiple boundary instances.
    '''
    with open(seq_path) as json_file:
        load_json = json.load(json_file)
        data_json = load_json
    seq_lens = []
    end_points = []
    init_points = []
    for area in data_json:
        seq_lens.append(len(area['seq']))
        end_points.append(area['end_vertex'])
        init_points.append(area['init_vertex'])
    seq = np.zeros((len(seq_lens),max(seq_lens),2))
    for idx,area in enumerate(data_json):
        seq[idx,:seq_lens[idx]] = [x[0:2] for x in area['seq']]
    # seq = torch.FloatTensor(seq)

    return seq, seq_lens, init_points, end_points

def load_image(image_path,mask_path):
    img = Image.open(image_path)
    img = tvf.to_tensor(img)
    assert img.shape[1] == img.shape[2]
    mask = np.array(Image.open(mask_path))[:,:,0]
    mask = mask / 255
    return img, mask


