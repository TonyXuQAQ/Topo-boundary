import argparse
import json
import os
import json
import shutil
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
from arguments import *
from model.models import MODELS_REFINE

from tqdm import tqdm
class dataset(Dataset):
    def __init__(self,args,valid=False):
        with open('./dataset/data_split.json','r') as jf:
            json_list = json.load(jf)['train']
        self.file_list = json_list
        self.tiff_list = [os.path.join(args.image_dir,'{}.tiff'.format(x)) for x in self.file_list]
        self.mask_list = [os.path.join(args.mask_dir,'{}.png'.format(x)) for x in self.file_list]
        if args.pretrain:
            self.segmentaition_list = ['./records/corrupted_mask/{}.png'.format(x) for x in self.file_list]
        else:
            self.segmentaition_list = ['./records/segmentation/train/{}.png'.format(x) for x in self.file_list]
        print('Finish loading the training data set lists! {}'.format(len(self.file_list)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,idx):
            tiff = tvf.to_tensor(Image.open(self.tiff_list[idx]))
            mask = tvf.to_tensor(Image.open(self.mask_list[idx]))
            seg = Image.open(self.segmentaition_list[idx])
            seg = tvf.to_tensor(seg)
            return tiff,mask,seg

class valid_dataset(Dataset):
    def __init__(self,args):
        with open('./dataset/data_split.json','r') as jf:
            json_list = json.load(jf) 
        if args.mode == 'test':
            self.file_list = json_list['test']
            if args.pretrain:
                self.processed_mask_list = ['./records/corrupted_mask/{}.png'.format(x) for x in self.file_list]
            else:
                self.processed_mask_list = ['./records/segmentation/test/{}.png'.format(x) for x in self.file_list]
        else:
            self.file_list = json_list['valid']
            if args.pretrain:
                self.processed_mask_list = ['./records/corrupted_mask/{}.png'.format(x) for x in self.file_list]
            else:
                self.processed_mask_list = ['./records/segmentation/valid/{}.png'.format(x) for x in self.file_list]
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
        seg = tvf.to_tensor(Image.open(self.processed_mask_list[idx]))
        mask = tvf.to_tensor(Image.open(self.mask_list[idx]))
        name = self.file_list[idx]
        return tiff,mask,seg,name 


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def train(args,epoch,net,dataloader,train_len,optimizor,criterion,writer,valid_dataloader,valid_len):
    net.train()
    counter = 0
    with tqdm(total=train_len, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
        for idx,data in enumerate(dataloader):
            img, mask, seg = data
            img, mask, seg = img.to(args.device), mask[:,0:1,:,:].type(torch.FloatTensor).to(args.device), seg[:,0:1,:,:].type(torch.FloatTensor).to(args.device)
            temp = seg
            loss_seg = 0
            for refine_idx in range(3):
                cat_feature = torch.cat([img,seg,temp],dim=1)
                pre_segs = net(cat_feature)
                # 
                loss_seg += criterion['ce'](pre_segs,mask)
                temp = pre_segs
            optimizor.zero_grad()
            loss_seg.backward()
            optimizor.step()
            pbar.set_postfix(**{'loss': round(loss_seg.item(),3)})
            pbar.update()
            # print('Epoch: {}/{} || batch: {}/{} || Loss seg: {}'.format(epoch,args.epochs,idx,train_len,round(loss_seg.item(),3)))
            writer.add_scalar('train/seg_loss',loss_seg.item(),counter + train_len*epoch)
            counter += 1
            if idx % (train_len-1) == 0 and idx:
                val(args,epoch,net,valid_dataloader,counter + train_len*epoch,valid_len,writer)
                if args.pretrain:
                    torch.save(net.state_dict(), "./checkpoints/refine_pretrain_{}.pth".format(epoch))
                else:
                    torch.save(net.state_dict(), "./checkpoints/refine_{}.pth".format(epoch))

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
        if not len(gt_points) or not len(graph_points):
            return 0,0,0

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
    # net.eval()
    f1_ave = 0
    with tqdm(total=val_len, desc='Validation' if args.mode=='train' else args.mode, unit='img') as pbar:
        for idx,data in enumerate(dataloader):
            img, mask, seg,name = data
            mask = mask[0,0].cpu().detach().numpy()
            img, seg = img.to(args.device), seg[:,0:1,:,:].type(torch.FloatTensor).to(args.device)
            temp = seg
            for refine_idx in range(3):
                cat_feature = torch.cat([img,seg,temp],dim=1)
                pre_segs = net(cat_feature)
                temp = pre_segs

            pre_segs = torch.sigmoid(pre_segs).cpu().detach().numpy()[0,0]
            if args.mode == 'train':
                Image.fromarray(pre_segs/np.max(pre_segs)*255).convert('RGB').save('./records/refine/valid/vis/{}.png'.format(name[0]))
                pre_segs = (pre_segs>0.2)
                prec, recall, f1 = eval_metric(pre_segs,mask)
                f1_ave = (f1_ave * idx + f1) / (idx+1)
                # print('Validation: {}/{} || Image: {}/{} || Precision/Recall/f1: {}/{}/{}'.format(epoch,args.epochs,idx,val_len,round(prec,3),round(recall,3),round(f1,3)))
            else:
                Image.fromarray(pre_segs/np.max(pre_segs)*255).convert('RGB').save('./records/refine/test/refined_seg/{}.png'.format(name[0]))
                # print('Refine test: Image: {}/{} '.format(idx,val_len))
            pbar.set_postfix(**{'F1-score': round(f1_ave,3)})
            pbar.update()
    print('Validation Summary: {}/{} || Average loss: {}'.format(epoch,args.epochs,round(f1_ave,3)))
    writer.add_scalar('val_f1_score',f1_ave,ii)

def skeleton():
    print('Start skeletonization...')
    with open('./dataset/data_split.json','r') as jf:
        json_data = json.load(jf)['test']
    skel_list = [x+'.png' for x in json_data]
    with tqdm(total=len(skel_list), unit='img') as pbar:
        thr = 0.2
        for i,seg in enumerate(skel_list):
            seg_name = os.path.join('./records/refine/test/refined_seg',seg)
            img = np.array(Image.open(seg_name))[:,:,0] / 255
            img = img / (np.max(img))
            # binarization
            img = (img > thr)
            # skeletonization
            seg_skeleton = skeletonize(img, method='lee')
            instances = measure.label(seg_skeleton / 255,background=0)
            indexs = np.unique(instances)[1:]
            # remove too short segments as outliers
            for index in indexs:
                instance_map = (instances == index)
                instance_points = np.where(instance_map==1)
                if len(instance_points[0]) < 30:
                    seg_skeleton[instance_points] = 0
            Image.fromarray(seg_skeleton).convert('RGB').save(os.path.join('./records/refine/test/skeleton',seg))
            pbar.update()
    print('Finish skeletonization...')

if __name__ == '__main__':
    parser = get_parser('refine')
    args = parser.parse_args()
    if args.mode != 'train':
        update_dir_refine_test(args)
    else:
        update_dir_refine_train(args)
    print('=======================')
    if args.pretrain:
        print('Start pretraining refinement of OrientationRefine...')
    else:
        print('Start refinement of OrientationRefine...')
    print('Device: ',args.device)
    print('Batch size: ',args.batch_size)
    print('Mode: ',args.mode)
    print('=======================')
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
    net = MODELS_REFINE['LinkNet34'](
        in_channels=6, num_classes=1
    )
    if not args.pretrain:
        net.load_state_dict(torch.load('./checkpoints/refine_pretrain_9.pth', map_location='cpu'))
        if args.load_checkpoint is not None and args.mode=='test':
            net.load_state_dict(torch.load(args.load_checkpoint, map_location='cpu'))
    net.to(device)
    optimizor = torch.optim.Adam(list(net.parameters()),lr=1e-4,weight_decay=1e-5)
    criterion = {'ce':nn.BCEWithLogitsLoss()}
    writer = SummaryWriter('./records/tensorboard/refine')
    
    epochs = args.epochs if not args.pretrain else max(1,args.epochs//2)
    for i in range(args.epochs):
        if args.mode == 'test':
            val(args,0,net,valid_dataloader,0,valid_len,writer)
            skeleton()
            break
        train(args,i,net,train_dataloader,train_len,optimizor,criterion,writer,valid_dataloader,valid_len)