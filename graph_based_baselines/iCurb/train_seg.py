import argparse
import json
import os
import json
import shutil
import pickle
#
from scipy.spatial import cKDTree
import scipy
from skimage import measure
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tvf
from tensorboardX import SummaryWriter
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize
#
from models.models_encoder import *
from arguments import *

class dataset(Dataset):
    def __init__(self,args,valid=False):
        # train the network with pretrain patches
        with open('./dataset/data_split.json','r') as jf:
            json_list = json.load(jf)['pretrain']
        
        self.file_list = json_list
        self.tiff_list = [os.path.join(args.image_dir,'{}.tiff'.format(x)) for x in self.file_list]
        self.mask_list = [os.path.join(args.mask_dir,'{}.png'.format(x)) for x in self.file_list]
        print('Finish loading the training data set lists {}!'.format(len(self.file_list)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,idx):
            tiff = tvf.to_tensor(Image.open(self.tiff_list[idx]))
            mask = tvf.to_tensor(Image.open(self.mask_list[idx]))
            return tiff,mask

class valid_dataset(Dataset):
    def __init__(self,args):
        with open('./dataset/data_split.json','r') as jf:
            json_list = json.load(jf) 
        self.file_list = json_list['valid'][:500]
        self.tiff_list = [os.path.join(args.image_dir,'{}.tiff'.format(x)) for x in self.file_list]
        self.mask_list = [os.path.join(args.mask_dir,'{}.png'.format(x)) for x in self.file_list]
        print('Finish loading the valid data set lists {}!'.format(len(self.file_list)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,idx):
        tiff = tvf.to_tensor(Image.open(self.tiff_list[idx]))
        mask = tvf.to_tensor(Image.open(self.mask_list[idx]))
        name = self.file_list[idx]
        return tiff,mask,name 

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def train(args,epoch,net,dataloader,train_len,optimizor,criterion,writer,valid_dataloader,valid_len):
    net.train()
    counter = 0
    best_f1 = 0
    for idx,data in enumerate(dataloader):
        img, mask= data
        img, mask= img.to(args.device), mask[:,0:1,:,:].type(torch.FloatTensor).to(args.device)
        predictions,_ = net(img)
        loss = criterion['bce'](predictions,mask)
        optimizor.zero_grad()
        loss.backward()
        optimizor.step()
        print('Epoch: {}/{} || batch: {}/{} || loss:{}'.format(epoch,args.epochs,idx,train_len,round(loss.item(),3)))
        writer.add_scalar('train/loss',loss.item(),counter + train_len*epoch)
        counter += 1
        if idx % (train_len-1) == 0 and idx:
            f1 = val(args,epoch,net,valid_dataloader,counter + train_len*epoch,valid_len,writer)
            if f1 > best_f1:
                torch.save(net.state_dict(), "./checkpoints/seg_pretrain.pth")
                best_f1 = f1

def val(args,epoch,net,dataloader,ii,val_len,writer,mode=0):
    def eval_metric(seg_result,mask):
        '''
        Evaluate the predicted image by F1 score during evaluation
        '''
        def tuple2list(t):
            return [[t[0][x],t[1][x]] for x in range(len(t[0]))]

        skel = skeletonize(seg_result, method='lee')
        gt_points = tuple2list(np.where(mask!=0))
        graph_points = tuple2list(np.where(skel!=0))

        graph_acc = 0
        graph_recall = 0
        gt_tree = scipy.spatial.cKDTree(gt_points)
        for c_i,thre in enumerate([5]):
            if len(graph_points):
                graph_tree = scipy.spatial.cKDTree(graph_points)
                graph_dds,_ = graph_tree.query(gt_points, k=1)
                gt_acc_dds,gt_acc_iis = gt_tree.query(graph_points, k=1)
                graph_recall = len([x for x in graph_dds if x<thre])/len(graph_dds)
                graph_acc = len([x for x in gt_acc_dds if x<thre])/len(gt_acc_dds)
        r_f = 0
        if graph_acc*graph_recall:
            r_f = 2*graph_recall * graph_acc / (graph_acc+graph_recall)
        return graph_acc, graph_recall,r_f

    net.eval()
    f1_ave = 0
    for idx,data in enumerate(dataloader):
        img, mask, name = data
        img, mask = img.to(args.device), mask[0,0,:,:].cpu().detach().numpy()
        with torch.no_grad():
            pre_segs,_ = net(img)
            pre_segs = torch.sigmoid(pre_segs[0,0,:,:]).cpu().detach().numpy()
            Image.fromarray(pre_segs/np.max(pre_segs)*255).convert('RGB').save('./records/seg/valid/{}.png'.format(name[0]))
            pre_segs = (pre_segs>0.2)
            prec, recall, f1 = eval_metric(pre_segs,mask)
            f1_ave = (f1_ave * idx + f1) / (idx+1)
            print('Validation:{}/{} || Image:{}/{} || Precision/Recall/f1:{}/{}/{}'.format(epoch,args.epochs,idx,val_len,round(prec,3),round(recall,3),round(f1,3)))

    print('Validation Summary:{}/{} || Average loss:{}'.format(epoch,args.epochs,round(f1_ave,3)))
    writer.add_scalar('val_loss',f1_ave,ii)
    return f1_ave

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    update_dir_seg(args)
    parser = argparse.ArgumentParser()
    device = args.device
    # load data
    train_dataset = dataset(args)
    valid_dataset = valid_dataset(args)
    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset,batch_size=1,shuffle=False)
    train_len = len(train_dataloader)
    valid_len = len(valid_dataloader)
    # network
    net = FPN()
    net.to(device=device)
    optimizor = torch.optim.Adam(list(net.parameters()),lr=1e-4)
    criterion = {'bce':nn.BCEWithLogitsLoss()}
    writer = SummaryWriter('./records/seg/tensorboard')
    
    for i in range(args.epochs):
        train(args,i,net,train_dataloader,train_len,optimizor,criterion,writer,valid_dataloader,valid_len)