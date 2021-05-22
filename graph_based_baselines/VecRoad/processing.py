# import os 
# import numpy as np
# import json

# seq_path = "./dataset/road_boundary/seq"

# for kk, json_file in enumerate(os.listdir(seq_path)):
#     def neighbor(v1,v2):
#         if (abs(v1[0]-v2[0])<=1 and abs(v1[1]-v2[1])<=1):
#             return True
#         else:
#             return False

#     # json_file='055192_12.json'
#     with open(os.path.join(seq_path,json_file),'r') as jf:
#         json_data = json.load(jf)
    
#     json_new = []
#     for ii, instance in enumerate(json_data):
#         seq = instance['seq']
#         init_node = None
#         end_node = instance['end_node']
#         if len(seq) <= 5:
#             continue
#         # print(init_node)
#         if 1:#init_node[0]!=0 and init_node[0]!=999 and init_node[1]!=0 and init_node[1]!=999:
#             new_seq = [seq[0]]
#             pre_point = seq[0]
#             next_point = seq[2]
#             if not neighbor(seq[1],seq[2]):
#                 init_node = seq[1]
#             else:
#                 new_seq.append(seq[1])
#             for jj, point in enumerate(seq[2:-1]):
#                 next_point = seq[jj+3]
#                 pre_point = seq[jj+1]
#                 # print(seq)
#                 if (not neighbor(point,next_point)) and (not neighbor(point,pre_point)):
#                     # print('Incorrect sequence',ii,jj,point,next_point)
#                     if init_node is not None:
#                         new_seq.insert(1,init_node)
#                     init_node = point
#                 else:
#                     new_seq.append(point)
#             # whether the init is the last vertex
#             if init_node is not None:
#                 new_seq.insert(0,init_node)
#                 new_seq.append(seq[-1])
#             elif neighbor(seq[-2],seq[-1]):
#                 init_node = new_seq[0]
#                 new_seq.append(seq[-1])
#             else:
#                 init_node = seq[-1]
#                 new_seq.insert(0,init_node)
#                 end_node = new_seq[-1]
#         else:
#             new_seq = seq
#         if new_seq[0]!=init_node or new_seq[-1]!=end_node:
#             print('Error!!!!!! Not equal end points!!!! {}'.format(ii))
#             print(new_seq[0],init_node)
#             print(new_seq[1],end_node)
#             print(kk,json_file)
#         # check
#         for jj, point in enumerate(new_seq[1:]):
#             pre_point = new_seq[jj]
#             if not neighbor(pre_point,point):
#                 print('Error!! Not neighbor!!! {}------{}'.format(pre_point,point))
#                 print(kk,json_file)

#         json_new.append({'seq':new_seq,'init_node':init_node,'end_node':end_node})
#     with open(os.path.join('./dataset/road_boundary/seq_new',json_file),'w') as jf:
#         json.dump(json_new,jf)


import numpy as np
import scipy.stats as st
from PIL import Image
import cv2
def gkern(l=20, sig=2.5):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.max(kernel) 


A = gkern(128, 5) 

vr,vc = 20,10
l = vc-64+1
r = vc+64+1
d = vr-64+1
u = vr+64+1
crop_l, crop_r, crop_d, crop_u = 0, 128, 0, 128
if l<0:
    crop_l = -l
if d<0:
    crop_d = -d
if r>1000:
    crop_r = crop_r-r+1000
if u>1000:
    crop_u = crop_u-u+1000
gt_coord = [1,20]
gt_coord = [int(gt_coord[0]-vr+(128/2-1)),int(gt_coord[1]-vc+(128/2-1))]
# create gaussian map
print(crop_l, crop_r, crop_d, crop_u)
mask_gt_coord_map = np.zeros((128,128))
mask_gt_coord_map[crop_d:crop_u,crop_l:crop_r] = 1
gt_coord_map = A
num_rows, num_cols = gt_coord_map.shape[:2]
translation_matrix = np.float32([[1,0,gt_coord[1]-(128/2-1)], [0,1,gt_coord[0]-(128/2-1)] ])
map_translation = cv2.warpAffine(gt_coord_map, translation_matrix, (num_cols,num_rows))
gt_coord_map = map_translation * mask_gt_coord_map
image = Image.fromarray(gt_coord_map*255).convert('RGB').save('./test.png')