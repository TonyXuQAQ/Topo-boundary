from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import json
import os 

class BasicDataset(Dataset):
    def __init__(self,args,valid=False):
        
        self.valid = valid
        self.imgs_dir = args.image_dir
        self.masks_dir = args.mask_dir
        with open('./dataset/data_split.json','r') as jf:
            data_load = json.load(jf)
        if args.test:
            self.ids = data_load['test']
            print('Testing mode. Data length {}.'.format(len(self.ids)))
        else:
            if valid:
                self.ids = data_load['valid']
                print('Validation length {}.'.format(len(self.ids)))
                print('=================')
            else:
                self.ids = data_load['train'] + data_load['pretrain']
                print('=================')
                print('Training mode: Training length {}.'.format(len(self.ids)))
        
        

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, whether_mask):
        w, h = pil_img.size
        newW, newH = int(w), int(h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
        img_nd = np.array(pil_img)
        if whether_mask:
            img_nd = img_nd[:,:,0]
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        #
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = os.path.join(self.imgs_dir, idx + '.tiff')
        mask_file = os.path.join(self.masks_dir, idx + '.png')
        img = Image.open(img_file)
        mask = Image.open(mask_file)
        img = self.preprocess(img,False)
        mask = self.preprocess(mask,True)
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'name':idx
        }