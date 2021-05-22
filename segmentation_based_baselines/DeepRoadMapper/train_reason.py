from __future__ import print_function

import argparse
import json
import os
from datetime import datetime
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import scipy
from skimage import measure
import numpy as np
import argparse
import json
import os
import json
import shutil
from tqdm import tqdm
#
from scipy.spatial import cKDTree
import scipy
from skimage import measure
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tvf
from tensorboardX import SummaryWriter
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize
#
from models.reason_net import ReasonNet
from arguments import *

class dataset(Dataset):
    def __init__(self,valid=False):
        # get lists of good and bad candidates
        good_list = os.listdir('./records/candidate_train/good')
        good_list = [x[:-4] for x in good_list if x[:3]!='rgb']
        bad_list = os.listdir('./records/candidate_train/bad')
        bad_list = [x[:-4] for x in bad_list if x[:3]!='rgb']
        # load cropped rgb images
        self.good_rgb_list = ['./records/candidate_train/good/rgb_{}.png'.format(x) for x in good_list]
        self.bad_rgb_list = ['./records/candidate_train/bad/rgb_{}.png'.format(x) for x in bad_list]
        # load cropped images of cancidate connections
        self.good_list = ['./records/candidate_train/good/{}.png'.format(x) for x in good_list]
        self.bad_list = ['./records/candidate_train/bad/{}.png'.format(x) for x in bad_list]
        #
        self.rgb_list = self.good_rgb_list + self.bad_rgb_list
        self.list = self.good_list + self.bad_list
        self.label = [1 for x in range(len(self.good_list))] + [0 for x in range(len(self.bad_list))]
        # shuffle
        zip_list = list(zip(self.rgb_list,self.list,self.label))
        random.shuffle(zip_list)
        self.rgb_list, self.list, self.label = zip(*zip_list)
        print('Finish loading the training data set lists!{}'.format(len(self.list)))

    def __len__(self):
        return len(self.list)

    def __getitem__(self,idx):
            rgb = tvf.to_tensor(Image.open(self.rgb_list[idx]))
            image = tvf.to_tensor(Image.open(self.list[idx]))
            label = torch.FloatTensor([self.label[idx]])
            return torch.cat([rgb,image],dim=0),label


class valid_dataset(Dataset):
    def __init__(self):
        good_list = os.listdir('./records/candidate_valid/good')
        good_list = [x[:-4] for x in good_list if x[:3]!='rgb'][:500]
        bad_list = os.listdir('./records/candidate_valid/bad')
        bad_list = [x[:-4] for x in bad_list if x[:3]!='rgb'][:500]

        # 
        self.good_rgb_list = ['./records/candidate_valid/good/rgb_{}.png'.format(x) for x in good_list]
        self.bad_rgb_list = ['./records/candidate_valid/bad/rgb_{}.png'.format(x) for x in bad_list]
        self.good_list = ['./records/candidate_valid/good/{}.png'.format(x) for x in good_list]
        self.bad_list = ['./records/candidate_valid/bad/{}.png'.format(x) for x in bad_list]
        #
        self.rgb_list = (self.good_rgb_list + self.bad_rgb_list)
        self.list = (self.good_list + self.bad_list)
        self.label = [1 for x in range(len(self.good_list))] + [0 for x in range(len(self.bad_list))]

        print('Finish loading the valid data set lists!{}'.format(len(self.list)))

    def __len__(self):
        return len(self.list)

    def __getitem__(self,idx):
            rgb = tvf.to_tensor(Image.open(self.rgb_list[idx]))
            image = tvf.to_tensor(Image.open(self.list[idx]))
            label = torch.FloatTensor([self.label[idx]])
            return torch.cat([rgb,image],dim=0),label

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def train(args,epoch,net,dataloader,train_len,optimizor,criterion,writer,valid_dataloader,valid_len):
    net.train()
    counter = 0
    with tqdm(total=train_len, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
        for idx,data in enumerate(dataloader):
            im, label = data
            im, label = im.to(args.device), label.type(torch.LongTensor).to(args.device).reshape(label.shape[0])
            pre_reason = net(im)
            loss = criterion['ce'](pre_reason,label)
            optimizor.zero_grad()
            loss.backward()
            optimizor.step()
            # print('Epoch: {}/{} || batch: {}/{} || Loss seg: {}'.format(epoch,args.epochs,idx,train_len,round(loss.item(),3)))
            writer.add_scalar('train/reason_loss',loss.item(),counter + train_len*epoch)
            counter += 1
            pbar.set_postfix(**{'loss':round(loss.item(),3)})
            pbar.update()
            if idx % (train_len-1) == 0 and idx:
                val(args,epoch,net,valid_dataloader,counter + train_len*epoch,valid_len,writer)
                torch.save(net.state_dict(), "./checkpoints/reason_{}.pth".format(epoch))

def val(args,epoch,net,dataloader,ii,val_len,writer,all_image=0):
    net.eval()
    num_correct = 0
    with tqdm(total=val_len, desc='Validation' , unit='img') as pbar:
        for idx,data in enumerate(dataloader):
            im, label = data
            im, label = im.to(args.device), label.to(args.device)
            with torch.no_grad():
                pre_reason = net(im)
                output = (pre_reason[0][1]>pre_reason[0][0])
                if output == label:
                    num_correct += 1
            pbar.update()
    # the accuracy of determining whether a candidate connection should be accepted (good or bad)
    acc = num_correct / val_len
    print('Validation: {}/{} || Acc: {}'.format(epoch,args.epochs,round(acc,3)))
    writer.add_scalar('val_accuracy',acc,ii)

def test(device,net):
    net.eval()
    with open('./dataset/data_split.json','r') as jf:
        json_list = json.load(jf)['test']
    json_list = [x+'.png' for x in json_list]
    skeleton_dir = './records/skeleton/test'
    json_dir = './records/candidate_test/json'
    candidate_dir = './records/candidate_test/reason'
    candidate_rgb_dir = './records/candidate_test/rgb'
    candidate_list = os.listdir(candidate_dir)
    rgb_list = os.listdir(candidate_rgb_dir)

    print('Start testing .....')
    with tqdm(total=len(json_list), desc='Testing' , unit='img') as pbar:
        for idx,name in enumerate(json_list):
            image_list = [x for x in candidate_list if x[:9]==name[:-4]]
            # read in the raw predicted skeleton
            binary_image = Image.open(os.path.join(skeleton_dir,name))
            draw = ImageDraw.Draw(binary_image)
            # check samples (candidate connections)
            for candidate in image_list:
                im_candidate = tvf.to_tensor(Image.open(os.path.join(candidate_dir,candidate)))
                im_rgb = tvf.to_tensor(Image.open(os.path.join(candidate_rgb_dir,candidate)))
                im = torch.cat([im_rgb,im_candidate],dim=0).unsqueeze(0).to(device)
                pre_reason = net(im)
                if pre_reason[0][1] > pre_reason[0][0]:
                    with open(os.path.join(json_dir,candidate[:-3]+'json'),'r') as jf:
                        json_data = json.load(jf)
                    src = json_data['src']
                    dst = json_data['dst']
                    draw.line((src[1],src[0], dst[1],dst[0]), fill='red')
            # save the image after post-processing
            binary_image.save(os.path.join('./records/reason/test/vis', name))
            # print('Testing {}/{}'.format(idx,len(json_list)))
            pbar.update()

if __name__ == '__main__':
    parser = get_parser('reason')
    args = parser.parse_args()
    if args.mode != 'train':
        update_dir_reason_test(args)
    else:
        update_dir_reason_train(args)
    parser = argparse.ArgumentParser()
    device = args.device
    # load data
    train_dataset = dataset()
    valid_dataset = valid_dataset()
    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset,batch_size=1,shuffle=False)
    train_len = len(train_dataloader)
    valid_len = len(valid_dataloader)
    # network
    net = ReasonNet()
    if args.load_checkpoint is not None and args.mode=='test':
        net.load_state_dict(torch.load(args.load_checkpoint, map_location='cpu'))
    net.to(args.device)
    optimizor = torch.optim.Adam(list(net.parameters()),lr=1e-4,weight_decay=1e-5)
    criterion = {'ce':nn.CrossEntropyLoss(),'l1':nn.L1Loss()}
    writer = SummaryWriter('./records/tensorboard/reason')
    
    for i in range(args.epochs):
        if args.mode == 'test':
            test(args.device,net)
            break
        else:
            train(args,i,net,train_dataloader,train_len,optimizor,criterion,writer,valid_dataloader,valid_len)