import os
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

with open('./data_split.json','r') as jf:
    json_data = json.load(jf)
tiff_list = json_data['train'] + json_data['valid'] + json_data['test'] + json_data['pretrain']

jp2_list = os.listdir('./temp_raw_tiff')
jp2_list = [x for x in jp2_list if x[-3:]=='jp2']
with tqdm(total=len(jp2_list), unit='img') as pbar:
    for jp2_name in jp2_list:
        raw_tiff = np.array(Image.open(f'./temp_raw_tiff/{jp2_name}'))
        for ii in range(5):
            for jj in range(5):
                cropped_tiff_name = f'{jp2_name[:-4]}_{ii}{jj}'
                if cropped_tiff_name in tiff_list:
                    Image.fromarray(raw_tiff[1000*ii:1000*(ii+1),1000*jj:1000*(jj+1)]).save(f'./cropped_tiff/{cropped_tiff_name}.tiff')
        pbar.update()