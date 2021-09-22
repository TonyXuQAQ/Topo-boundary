import numpy as np
from PIL import Image
import os
from skimage.morphology import skeletonize
from skimage import measure
import json
from tqdm import tqdm
from scipy.spatial import cKDTree

def trans_v(v):
    if v[0] < 10:
        v[0] = 0
    elif v[0] >= 990:
        v[0] = 999

    if v[1] < 10:
        v[1] = 0
    elif v[1] >= 990:
        v[1] = 999

def process(pre_keypoint_map,name):
    
    prob_labels = measure.label(pre_keypoint_map>10, connectivity=2)
    # print(name)
    # Image.fromarray(((pre_keypoint_map>15)*255).astype(np.uint8)).convert('RGB').save('./test.png')                                                                                                                                    
    props = measure.regionprops(prob_labels)
    max_area = 5
    init_vertices = []
    for index, region in enumerate(props):
        v = region.centroid
        # trans_v(v)
        init_vertices.append([int(v[0]),int(v[1])])
        
    with open(f'./records/endpoint/vertices/{name}.json','w') as jf:
        json.dump(init_vertices,jf)

image_dir = './records/endpoint/test'
image_list = os.listdir(image_dir)
with tqdm(total=len(image_list), unit='img') as pbar:
    for ii, image_name in enumerate(image_list):
        image = np.array(Image.open(os.path.join(image_dir,image_name)))[:,:,0]
        process(image,image_name[:-4])
        pbar.update()
        # break
