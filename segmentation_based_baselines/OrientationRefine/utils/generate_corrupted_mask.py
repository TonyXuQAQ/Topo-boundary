import numpy as np
from PIL import Image
import sys
import os 
import shutil
import json
import pickle
import cv2
from tqdm import tqdm
sys.path.append(os.path.abspath('.'))
from arguments import *

parser = get_parser()
args = parser.parse_args()
update_dir_candidate(args)

def corrupted_mask(set_name):
    print('Start generating corrupted masks of {}'.format(set_name))
    with open('./dataset/data_split.json','r') as jf:
        json_list = json.load(jf)[set_name]
    json_list = [x + '.png' for x in json_list]
    gt_dir = args.mask_dir
    with tqdm(total=len(json_list),  unit='img') as pbar:
        for idx, image_name in enumerate(json_list):
            gt_mask = np.array(Image.open(os.path.join(gt_dir,image_name)))
            points = np.where(gt_mask[:,:,0]!=0)
            points = [[points[0][x],[points[1][x]]] for x in range(len(points[0]))]
            if len(points) > 200:
                chosen_ids = np.random.choice(range(len(points)), min(20,len(points)//20), replace=False)
                random_angles = np.random.uniform(0,np.pi,20)
                random_widths = np.random.uniform(10,50,20)
                random_length = np.random.uniform(50,150,20)
                for i in range(len(chosen_ids)):
                    central_point = points[chosen_ids[i]]
                    src = [central_point[1] - random_length[i] / 2 * np.sin(random_angles[i]),central_point[0] - random_length[i] / 2 * np.cos(random_angles[i])]
                    dst = [central_point[1] + random_length[i] / 2 * np.sin(random_angles[i]),central_point[0] + random_length[i] / 2 * np.cos(random_angles[i])]
                    src = [max(0,min(999,int(x))) for x in src]
                    dst = [max(0,min(999,int(x))) for x in dst]
                    cv2.line(gt_mask,tuple(src),tuple(dst),(0,0,0),int(random_widths[i]))
            pbar.update()
            Image.fromarray(gt_mask).convert('RGB').save('./records/corrupted_mask/{}.png'.format(image_name[:-4]))
    print('Finish generating corrupted masks of set {}'.format(set_name))  

corrupted_mask('train')
corrupted_mask('valid')
corrupted_mask('test')