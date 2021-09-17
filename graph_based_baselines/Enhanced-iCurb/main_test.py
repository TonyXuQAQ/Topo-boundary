import numpy as np
import torch
import random
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
import torch.nn.functional as F
import os
import scipy
import json
from tqdm import tqdm

def run_test(env):
    agent = env.agent
    network = env.network
    args = env.args
    crop_size = env.crop_size

    def update_coord(pre_coord):
        pre_coord = pre_coord.cpu().detach().numpy()
        v_next = agent.train2world(pre_coord)
        if (agent.v_now == v_next):
            agent.taken_stop_action = 1
        env.remove_duplicate_init_points(v_next)
        agent.v_now = v_next
        
    def update_graph(start_vertex,end_vertex,graph,set_value=1):
        start_vertex = np.array([int(start_vertex[0]),int(start_vertex[1])])
        end_vertex = np.array([int(end_vertex[0]),int(end_vertex[1])])
        instance_vertices = []
        p = start_vertex
        d = end_vertex - start_vertex
        N = np.max(np.abs(d))
        graph[:,:,start_vertex[0],start_vertex[1]] = set_value
        graph[:,:,end_vertex[0],end_vertex[1]] = set_value
        if N:
            s = d / (N)
            for i in range(0,N):
                p = p + s
                graph[:,:,int(round(p[0])),int(round(p[1]))] = set_value
        return graph

    def visualization_image(graph_record,name,init_points):
        output = graph_record[0,0,:,:].cpu().detach().numpy()
        points = np.where(output!=0)
        graph_map = Image.fromarray(output * 255).convert('RGB')
        # draw = ImageDraw.Draw(graph_map)
        # for p in init_points:
        #     draw.ellipse((p[1]-4,p[0]-4,p[1]+4,p[0]+4),fill='yellow',outline='yellow')
        # if len(points[0]):
        graph_map.save('./records/test/skeleton/{}.png'.format(name))

        draw = ImageDraw.Draw(graph_map)
        for p in init_points:
            draw.ellipse((p[1]-4,p[0]-4,p[1]+4,p[0]+4),fill='yellow',outline='yellow')
        # if len(points[0]):
        graph_map.save('./records/test/graph/{}.png'.format(name))
     
    network.val_mode()
    eval_len = network.val_len()
    graph_acc_ave = 0
    graph_recall_ave=0
    r_f_ave = 0
    # =================working on an image=====================
    with tqdm(total=len(network.dataloader_valid), unit='img') as pbar:
        for i, data in enumerate(network.dataloader_valid):
            if data is None:
                continue
            _, _, tiff, mask, _, name, init_points, _ = data
            name = name[0][-14:-5]
            tiff = tiff.to(args.device)
            mask, init_points = mask[0], init_points[0]
            init_points = [[int(x[0]),int(x[1])] for x in init_points]
            env.init_point_set = init_points.copy()
            # clear history info
            env.init_image(valid=True)
            with torch.no_grad():
                fpn_feature_map = network.encoder(tiff)
                # pre_seg_mask = network.decoder_seg(fpn_feature_map)
                fpn_feature_map = F.interpolate(fpn_feature_map, size=(1100, 1100), mode='bilinear', align_corners=True)
                fpn_feature_tensor = fpn_feature_map#torch.cat([tiff,fpn_feature_map],dim=1)
                
            # record generated vertices
            vertices_output = []
            vertices_image_visualization = []
            if len(env.init_point_set):
                while len(env.init_point_set):
                    init_point = env.init_point_set.pop()
                    # ===============working on a curb instance======================
                    agent.init_agent(init_point)
                    vertices_record_list = []
                    vertices_record_list.append(init_point)
                    while 1:
                        agent.agent_step_counter += 1
                        # network predictions
                        cropped_feature_tensor = agent.crop_attention_region(fpn_feature_tensor,val_flag=True)
                        with torch.no_grad():
                            v_now = [x/1100 for x in agent.v_now]
                            v_now = torch.FloatTensor(v_now).unsqueeze(0).to(args.device)
                            v_previous = [x/1100 for x in agent.v_previous]
                            v_previous = torch.FloatTensor(v_previous).unsqueeze(0).to(args.device)
                            pre_stop_action = network.decoder_stop(cropped_feature_tensor,v_now,v_previous)
                            pre_coord = network.decoder_coord(cropped_feature_tensor,v_now,v_previous)
                        pre_stop_action = pre_stop_action.squeeze(1)
                        pre_coord = pre_coord.squeeze(0).squeeze(0)
                        pre_stop_action = F.softmax(pre_stop_action,dim=1)
                        agent.v_previous = agent.v_now
                        # update vertex coord
                        update_coord(pre_coord)
                        if (pre_stop_action[0][1] > 0.5 and agent.agent_step_counter > 20):
                            break
                        # record
                        env.graph_record = update_graph(agent.v_now,agent.v_previous,env.graph_record)
                        vertices_record_list.append(np.array(agent.v_now).tolist())
                        if (agent.agent_step_counter > args.max_length*1.2)\
                            or (((agent.v_now[0]>=1095) or (agent.v_now[0]<=5)or (agent.v_now[1]>=1095) or (agent.v_now[1]<=5)) and agent.agent_step_counter > 10):
                            agent.taken_stop_action = 1
                            
                        if agent.taken_stop_action:    
                            break
                    vertices_output.append(vertices_record_list)
                    vertices_image_visualization.extend(vertices_record_list)
            # calculate metrics
            # vis and print
            visualization_image(env.graph_record,name,init_points)
            if args.test:
                with open('./records/test/vertices_record/{}.json'.format(name),'w') as jf:
                    json.dump(vertices_output,jf)
            else:
                with open('./records/valid/vertices_record/{}.json'.format(name),'w') as jf:
                    json.dump(vertices_output,jf)
            pbar.update()
    # finish
        # break
    print('----Finishing Validation')
    return r_f_ave

