import argparse
import logging
import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from models import UNet
from PIL import Image
from skimage.morphology import skeletonize
from skimage import measure

from tensorboardX import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from arguments import get_parser, update_dir_train, update_dir_test

def train_net(net,
              args,
              epochs,
              batch_size,
              lr,
              device):

    train = BasicDataset(args,False)
    val = BasicDataset(args,True)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=False,  pin_memory=True, drop_last=True)
    n_val = len(val)
    n_train = len(train) 
    # 
    writer = SummaryWriter('./records/tensorboard')
    global_step = 0
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # 
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                masks = batch['mask']
                name = batch['name'][0]
                imgs = imgs.to(device=device, dtype=torch.float32)
                masks = masks.to(device=device, dtype=torch.float32)
                #
                masks_pred = net(imgs)
                loss = criterion(masks_pred,masks)
                pbar.set_postfix(**{'loss': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(imgs.shape[0])
                #
                global_step += 1
                writer.add_scalar('train', loss.item(), global_step)
                if (global_step % (n_train // (batch_size)) == 0):
                    valid_score = eval_net(args,net,val_loader,device)
                    # scheduler.step(val_score)
                    writer.add_scalar('valid', valid_score, global_step)
                    torch.save(net.state_dict(),
                        args.checkpoints_dir + 'naive_baseline_{}.pth'.format(epoch))

def skeleton():
    print('Start skeletonization...')
    with open('./dataset/data_split.json','r') as jf:
        json_data = json.load(jf)['test']
    skel_list = [x+'.png' for x in json_data]
    with tqdm(total=len(skel_list), unit='img') as pbar:
        thr = 0.2
        for i,seg in enumerate(skel_list):
            seg_name = os.path.join('./records/test/segmentation',seg)
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
            Image.fromarray(seg_skeleton).convert('RGB').save(os.path.join('./records/test/skeleton',seg))
            pbar.update()
    print('Finish skeletonization...')

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.test:
        update_dir_test(args)
    else:
        update_dir_train(args)
    device = torch.device(args.device)
    net = UNet(n_channels=4,n_classes=1,bilinear=True)
    if args.test:
        net.load_state_dict(
            torch.load(args.load_checkpoint, map_location='cpu')
        )
    net.to(device=device)
    try:
        if not args.test:
            train_net(net=net,
                    args=args,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=device)
        else:
            test = BasicDataset(args)
            test_loader = DataLoader(test, batch_size=1, shuffle=False,  pin_memory=True, drop_last=True)
            eval_net(args,net, test_loader, device)
            skeleton()

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
