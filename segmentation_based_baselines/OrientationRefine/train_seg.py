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
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tvf
from tensorboardX import SummaryWriter
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize
import torch.nn.functional as F
#
from utils.loss import CrossEntropyLoss2d, mIoULoss
from model.models import MODELS
from arguments import *

class dataset(Dataset):
    def __init__(self,valid=False):
        with open('./dataset/data_split.json','r') as jf:
            json_list = json.load(jf)['pretrain']
        self.file_list = json_list
        self.tiff_list = [os.path.join(args.image_dir,'{}.tiff'.format(x)) for x in self.file_list]
        self.mask_list = [os.path.join(args.mask_dir,'{}.png'.format(x)) for x in self.file_list]
        self.ori_list = [os.path.join(args.ori_dir,'{}.png'.format(x)) for x in self.file_list]
        print('Finish loading the training data set lists {}!'.format(len(self.file_list)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,idx):
            tiff = tvf.to_tensor(Image.open(self.tiff_list[idx]))
            mask = tvf.to_tensor(Image.open(self.mask_list[idx]))
            ori = torch.LongTensor(np.array(Image.open(self.ori_list[idx]))[:,:,0])
            return tiff,mask,ori

class valid_dataset(Dataset):
    def __init__(self,mode=0):
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
        if args.mode == 'train':
            print('Finish loading the valid data set lists {}!'.format(len(self.file_list)))
        else:
            print('Finish loading the test data set lists {}!'.format(len(self.file_list)))

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
    with tqdm(total=train_len, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
        for idx,data in enumerate(dataloader):
            img, mask, ori_mask = data
            img, mask, ori_mask = img.to(args.device), mask[:,0:1,:,:].type(torch.FloatTensor).to(args.device), ori_mask.to(args.device)
            #
            pre_segs, pre_oris = net(img)
            
            loss_seg = 0
            loss_seg += criterion['ce'](F.interpolate(pre_segs[0], scale_factor=(4,4), mode='bilinear', align_corners=True),mask)
            loss_seg += criterion['ce'](F.interpolate(pre_segs[1], scale_factor=(4,4), mode='bilinear', align_corners=True),mask)
            loss_seg += criterion['ce'](F.interpolate(pre_segs[2], scale_factor=(2,2), mode='bilinear', align_corners=True),mask)
            loss_seg += criterion['ce'](pre_segs[3],mask)
            loss_ori = 0
            loss_ori += criterion['orien_loss'](F.interpolate(pre_oris[0], scale_factor=(4,4), mode='bilinear', align_corners=True),ori_mask)
            loss_ori += criterion['orien_loss'](F.interpolate(pre_oris[1], scale_factor=(4,4), mode='bilinear', align_corners=True),ori_mask)
            loss_ori += criterion['orien_loss'](F.interpolate(pre_oris[2], scale_factor=(2,2), mode='bilinear', align_corners=True),ori_mask)
            loss_ori += criterion['orien_loss'](pre_oris[3],ori_mask)

            loss = loss_seg + loss_ori
            optimizor.zero_grad()
            loss.backward()
            optimizor.step()
            pbar.set_postfix(**{'loss seg/ori': '{}/{}'.format(round(loss_seg.item(),3),round(loss_ori.item(),3))})
            # print('Epoch: {}/{} || batch: {}/{} || Loss seg: {}/ Loss ori: {}'.format(epoch,args.epochs,idx,train_len,round(loss_seg.item(),3),round(loss_ori.item(),3)))
            writer.add_scalar('train/seg_loss',loss_seg.item(),counter + train_len*epoch)
            writer.add_scalar('train/ori_loss',loss_ori.item(),counter + train_len*epoch)
            counter += 1
            pbar.update()
            if idx % (train_len-1) == 0 and idx:
                val(args,epoch,net,valid_dataloader,counter + train_len*epoch,valid_len,writer)
                torch.save(net.state_dict(), "./checkpoints/seg_{}.pth".format(epoch))
                print('Checkpoint {} saved!'.format(epoch))

def val(args,epoch,net,dataloader,ii,val_len,writer):
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
    with tqdm(total=val_len, desc='Validation' if args.mode=='train' else args.mode, unit='img') as pbar:
        for idx,data in enumerate(dataloader):
            img, mask,name = data
            img, mask = img.to(args.device), mask[0,0,:,:].cpu().detach().numpy()
            with torch.no_grad():
                pre_segs, _ = net(img)
                pre_segs = torch.sigmoid(pre_segs[3]).cpu().detach().numpy()[0,0]
                if args.mode == 'train':
                    Image.fromarray(pre_segs/np.max(pre_segs)*255).convert('RGB').save('./records/seg/valid/segmentation/{}.png'.format(name[0]))
                    pre_segs = (pre_segs>0.2)
                    prec, recall, f1 = eval_metric(pre_segs,mask)
                    f1_ave = (f1_ave * idx + f1) / (idx+1)
                    # print('Validation: {}/{} || Image: {}/{} || Precision/Recall/f1: {}/{}/{}'.format(epoch,args.epochs,idx,val_len,round(prec,3),round(recall,3),round(f1,3)))
                elif args.mode == 'infer_train':
                    Image.fromarray(pre_segs/np.max(pre_segs)*255).convert('RGB').save('./records/segmentation/train/{}.png'.format(name[0]))
                    # print('Generating samples for segmentation train: Image: {}/{} '.format(idx,val_len))
                elif args.mode == 'infer_valid':
                    Image.fromarray(pre_segs/np.max(pre_segs)*255).convert('RGB').save('./records/segmentation/valid/{}.png'.format(name[0]))
                    # print('Generating samples for segmentation valid: Image: {}/{} '.format(idx,val_len))
                else:
                    Image.fromarray(pre_segs/np.max(pre_segs)*255).convert('RGB').save('./records/segmentation/test/{}.png'.format(name[0]))
                    # print('Generating samples for segmentation test: Image: {}/{} '.format(idx,val_len))
            pbar.set_postfix(**{'F1-score': round(f1_ave,3)})
            pbar.update()
    print('Validation Summary: {}/{} || Average loss: {}'.format(epoch,args.epochs,round(f1_ave,3)))
    writer.add_scalar('val_f1_score',f1_ave,ii)

if __name__ == '__main__':
    parser = get_parser('seg')
    args = parser.parse_args()
    print(args.mode)
    if args.mode != 'train':
        update_dir_seg_test(args)
    else:
        update_dir_seg_train(args)
    print('=======================')
    print('Start segmentation of OrientationRefine...')
    print('Device: ',args.device)
    print('Batch size: ',args.batch_size)
    print('Mode: ',args.mode)
    print('=======================')
    parser = argparse.ArgumentParser()
    device = args.device
    # load data
    train_dataset = dataset()
    valid_dataset = valid_dataset(mode=args.mode)
    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset,batch_size=1,shuffle=False)
    train_len = len(train_dataloader)
    valid_len = len(valid_dataloader)
    # network
    net = MODELS['StackHourglassNetMTL'](
        in_channels=4,task1_classes=1
    )
    if args.load_checkpoint is not None and args.mode !='train':
        net.load_state_dict(torch.load(args.load_checkpoint, map_location='cpu'))
    net.to(device)
    optimizor = torch.optim.Adam(list(net.parameters()),lr=1e-4,weight_decay=1e-5)
    orien_loss = CrossEntropyLoss2d(
     size_average=True, ignore_index=255, reduce=True
    )
    road_loss = mIoULoss(n_classes=1,
        device=device
    )
    criterion = {'orien_loss':orien_loss,'road_loss':road_loss,'ce':nn.BCEWithLogitsLoss()}
    writer = SummaryWriter('./records/tensorboard/seg')
    
    
    for i in range(args.epochs):
        if args.mode != 'train':
            val(args,0,net,valid_dataloader,0,valid_len,writer)
            break
        train(args,i,net,train_dataloader,train_len,optimizor,criterion,writer,valid_dataloader,valid_len)