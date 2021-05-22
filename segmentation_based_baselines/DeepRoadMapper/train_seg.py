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
#
from unet import UNet
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
            print('Finish loading the testing data set lists {}!'.format(len(self.file_list)))

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
            img, mask = data
            img, mask = img.to(args.device), mask[:,0:1,:,:].type(torch.FloatTensor).to(args.device)
            pre_segs = net(img)
            loss_seg = criterion['ce'](pre_segs,mask)
            optimizor.zero_grad()
            loss_seg.backward()
            optimizor.step()
            # print('Epoch: {}/{} || batch: {}/{} || Loss seg: {}'.format(epoch,args.epochs,idx,train_len,round(loss_seg.item(),3)))
            writer.add_scalar('train/seg_loss',loss_seg.item(),counter + train_len*epoch)
            counter += 1
            pbar.set_postfix(**{'loss':round(loss_seg.item(),3)})
            pbar.update()
            if idx % (train_len-1) == 0 and idx:
                val(args,epoch,net,valid_dataloader,counter + train_len*epoch,valid_len,writer)
                torch.save(net.state_dict(), "./checkpoints/seg_{}.pth".format(epoch))

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
    with tqdm(total=val_len, desc='Validation' if args.mode=='train' else args.mode, unit='img') as pbar:
        for idx,data in enumerate(dataloader):
            img, mask, name = data
            img, mask = img.to(args.device), mask[0,0,:,:].cpu().detach().numpy()
            with torch.no_grad():
                pre_segs = net(img)
                pre_segs = torch.sigmoid(pre_segs).cpu().detach().numpy()[0,0]
                if args.mode == 'train':
                    Image.fromarray(pre_segs/np.max(pre_segs)*255).convert('RGB').save('./records/seg/valid/segmentation/{}.png'.format(name[0]))
                    pre_segs = (pre_segs>0.2)
                    prec, recall, f1 = eval_metric(pre_segs,mask)
                    f1_ave = (f1_ave * idx + f1) / (idx+1)
                    # print('Validation:{}/{} || Image:{}/{} || Precision/Recall/f1:{}/{}/{}'.format(epoch,args.epochs,idx,val_len,round(prec,3),round(recall,3),round(f1,3)))
                else:
                    Image.fromarray(pre_segs/np.max(pre_segs)*255).convert('RGB').save('./records/segmentation/{}/{}.png'.format(args.mode[6:],name[0]))
                    # print('Sementation testing: Image: {}/{} '.format(idx,val_len))
            pbar.set_postfix(**{'F1-score': round(f1_ave,3)})
            pbar.update()
    print('Validation Summary:{}/{} || Average loss:{}'.format(epoch,args.epochs,round(f1_ave,3)))
    writer.add_scalar('val_loss',f1_ave,ii)

def skeleton(mode):
    print('Start skeletonization...')
    segmentation_dir = './records/segmentation/{}'.format(mode[6:])
    seg_dir = os.listdir(segmentation_dir)
    for i,seg in enumerate(seg_dir):
        seg_name = os.path.join(segmentation_dir,seg)
        img = np.array(Image.open(seg_name))
        img = img[:,:,0] 
        img = img / (np.max(img))
        img = (img > 0.2)
        seg_skeleton = skeletonize(img, method='lee')
        all_labels = measure.label(seg_skeleton / 255,background=0)
        indexs = np.unique(all_labels)[1:]
        for index in indexs:
            index_map = (all_labels == index)
            index_points = np.where(index_map==1)
            if len(index_points[0]) < 30:
                seg_skeleton[index_points] = 0
        if mode == 'infer_train':
            Image.fromarray(seg_skeleton).convert('RGB').save(os.path.join('./records/skeleton/train/{}'.format(seg)))
        elif mode == 'infer_valid':
            Image.fromarray(seg_skeleton).convert('RGB').save(os.path.join('./records/skeleton/valid/{}'.format(seg)))
        else:
            Image.fromarray(seg_skeleton).convert('RGB').save(os.path.join('./records/skeleton/test/{}'.format(seg)))
        # print(str(i),'/',str(len(seg_dir)),'/',seg)
    print('Finish skeletonization...')

if __name__ == '__main__':
    parser = get_parser('seg')
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
    net = UNet(n_channels=4,n_classes=1,bilinear=True)
    net.to(device=device)
    if args.load_checkpoint is not None and args.mode!='train':
        net.load_state_dict(torch.load(args.load_checkpoint, map_location='cpu'))
    net.to(args.device)
    optimizor = torch.optim.Adam(list(net.parameters()),lr=1e-4,weight_decay=1e-5)
    criterion = {'ce':nn.BCEWithLogitsLoss()}
    writer = SummaryWriter('./records/tensorboard/seg')
    
    for i in range(args.epochs):
        if args.mode != 'train':
            val(args,0,net,valid_dataloader,0,valid_len,writer,mode=args.mode)
            skeleton(args.mode)
            break
        train(args,i,net,train_dataloader,train_len,optimizor,criterion,writer,valid_dataloader,valid_len)