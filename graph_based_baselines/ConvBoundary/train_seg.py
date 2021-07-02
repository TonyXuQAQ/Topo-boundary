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
from models.FPN import *
from models.loss import cos_loss
from arguments import *

class dataset(Dataset):
    def __init__(self,args,valid=False):
        # train the network with pretrain patches
        with open('./dataset/data_split.json','r') as jf:
            json_list = json.load(jf)['pretrain']
        
        self.file_list = json_list
        self.tiff_list = [os.path.join(args.image_dir,'{}.tiff'.format(x)) for x in self.file_list]
        self.mask_list = [os.path.join(args.mask_dir,'{}.png'.format(x)) for x in self.file_list]
        self.inverse_list = [os.path.join(args.inverse_dir,'{}.png'.format(x)) for x in self.file_list]
        self.direction_list = [os.path.join(args.direction_dir,'{}.pickle'.format(x)) for x in self.file_list]
        print('Finish loading the training data set lists {}!'.format(len(self.file_list)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,idx):
            tiff = tvf.to_tensor(Image.open(self.tiff_list[idx]))
            mask = tvf.to_tensor(Image.open(self.mask_list[idx]))
            inverse = tvf.to_tensor(Image.open(self.inverse_list[idx]))
            with open(self.direction_list[idx],'rb') as pk:
                direction = pickle.load(pk)
            direction = torch.FloatTensor(direction)
            return tiff,mask, inverse, direction

class valid_dataset(Dataset):
    def __init__(self,args):
        with open('./dataset/data_split.json','r') as jf:
            json_list = json.load(jf) 
        if args.mode=='train':
            self.file_list = json_list['valid']
        elif args.mode == 'infer_train':
            self.file_list = json_list['train']
        elif args.mode == 'infer_valid':
            self.file_list = json_list['valid']
        else:
            self.file_list = json_list['test']
            
        self.tiff_list = [os.path.join(args.image_dir,'{}.tiff'.format(x)) for x in self.file_list]
        self.mask_list = [os.path.join(args.mask_dir,'{}.png'.format(x)) for x in self.file_list]
        self.inverse_list = [os.path.join(args.inverse_dir,'{}.png'.format(x)) for x in self.file_list]
        self.direction_list = [os.path.join(args.direction_dir,'{}.pickle'.format(x)) for x in self.file_list]
        if args.mode == 'train':
            print('Finish loading the valid data set lists {}!'.format(len(self.file_list)))
        else:
            print('Finish loading the testing data set lists {}!'.format(len(self.file_list)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,idx):
        tiff = tvf.to_tensor(Image.open(self.tiff_list[idx]))
        mask = tvf.to_tensor(Image.open(self.mask_list[idx]))
        # inverse = tvf.to_tensor(Image.open(self.inverse_list[idx]))
        # with open(self.direction_list[idx],'rb') as pk:
        #     direction = pickle.load(pk)
        # direction = torch.FloatTensor(direction)
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
        img, mask, inverse, direction = data
        img, mask, inverse, direction = img.to(args.device), mask[:,0:1,:,:].type(torch.FloatTensor).to(args.device), inverse[:,0,:,:].to(args.device), direction.to(args.device)
        pre_inverse, pre_direction = net(img)
        pre_inverse = pre_inverse.squeeze(1)
        pre_direction = pre_direction.squeeze(1)
        loss_inverse = criterion['ce'](pre_inverse,inverse)
        loss_direction = criterion['cos'](pre_direction,direction)
        loss = loss_inverse+loss_direction
        optimizor.zero_grad()
        loss.backward()
        optimizor.step()
        print('Epoch: {}/{} || batch: {}/{} || loss_inv:{}/loss_cos:{}'.format(epoch,args.epochs,idx,train_len,round(loss_inverse.item(),3),round(loss_direction.item(),3)))
        writer.add_scalar('train/inverse_loss',loss_inverse.item(),counter + train_len*epoch)
        writer.add_scalar('train/direction_loss',loss_direction.item(),counter + train_len*epoch)
        counter += 1
        if idx % (train_len-1) == 0 and idx:
            f1 = val(args,epoch,net,valid_dataloader,counter + train_len*epoch,valid_len,writer)
            if f1 > best_f1:
                f1 = best_f1
                torch.save(net.state_dict(), "./checkpoints/convBoundary_seg_pretrain.pth")

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
            binary_seg, pre_direction_map = net(img)
            if args.mode == 'train':
                binary_seg = torch.sigmoid(binary_seg[0,0,:,:]).cpu().detach().numpy()
                Image.fromarray(binary_seg/np.max(binary_seg)*255).convert('RGB').save('./records/seg/valid/segmentation/{}.png'.format(name[0]))
                binary_seg = (binary_seg>0.2)
                prec, recall, f1 = eval_metric(binary_seg,mask)
                f1_ave = (f1_ave * idx + f1) / (idx+1)
                print('Validation:{}/{} || Image:{}/{} || Precision/Recall/f1:{}/{}/{}'.format(epoch,args.epochs,idx,val_len,round(prec,3),round(recall,3),round(f1,3)))

                pre_direction_map = torch.sigmoid(pre_direction_map[0,:,:,:]).cpu().detach().numpy().transpose(1,2,0)
                vis = np.zeros((1000,1000,3))
                vis[:,:,:2] = pre_direction_map
                vis[:,:,2] = (pre_direction_map[:,:,0] + pre_direction_map[:,:,1])/2
                Image.fromarray((vis*255).astype(np.uint8)).convert('RGB').save('./records/seg/valid/direction/{}.png'.format(name[0]))

    print('Validation Summary:{}/{} || Average loss:{}'.format(epoch,args.epochs,round(f1_ave,3)))
    writer.add_scalar('val_loss',f1_ave,ii)
    return f1_ave

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.mode != 'train':
        update_dir_seg_test(args)
    else:
        update_dir_seg_train(args)
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
    net = FPNSeg()
    net.to(device=device)
    if args.mode!='train':
        net.load_state_dict(torch.load('./checkpoints/convBoundary_seg_pretrain.pth', map_location='cpu'))
    net.to(args.device)
    optimizor = torch.optim.Adam(list(net.parameters()),lr=1e-4)
    criterion = {'ce':nn.BCEWithLogitsLoss(),'cos':cos_loss()}
    writer = SummaryWriter('./records/tensorboard/seg')
    
    for i in range(args.epochs):
        if args.mode != 'train':
            val(args,0,net,valid_dataloader,0,valid_len,writer,mode=args.mode)
            break
        train(args,i,net,train_dataloader,train_len,optimizor,criterion,writer,valid_dataloader,valid_len)