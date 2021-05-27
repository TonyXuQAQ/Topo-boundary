import numpy as np
import torch
import random
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
import torch.nn.functional as F
import os
import scipy
import json

def run_val(env,val_index):
    agent = env.agent
    network = env.network
    args = env.args
    crop_size = env.crop_size

    def update_coord(pre_coord):
        pre_coord = pre_coord.cpu().detach().numpy()
        v_next = agent.train2world(pre_coord)
        if (agent.v_now == v_next):
            agent.taken_stop_action = 1
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

    def visualization_image(graph_record,name,mask,vertices):
        output = graph_record[0,0,:,:].cpu().detach().numpy()
        points = np.where(output!=0)
        graph_map = Image.fromarray(output * 255).convert('RGB')
        if len(points[0]):
            if args.test:
                graph_map.save('./records/test/skeleton/{}.png'.format(name))
            else:    
                draw = ImageDraw.Draw(graph_map)
                for vertex in vertices:
                    draw.ellipse((vertex[1]-1,vertex[0]-1,vertex[1]+1,vertex[0]+1),fill='yellow',outline='yellow')
                graph_map.save('./records/valid/vis/{}_{}.png'.format(val_index,name))

    def eval_metric(graph,mask,name):
        def tuple2list(t):
            return [[t[0][x],t[1][x]] for x in range(len(t[0]))]

        graph = graph.cpu().detach().numpy()
        gt_image = mask
        gt_points = tuple2list(np.where(gt_image!=0))
        graph_points = tuple2list(np.where(graph!=0))

        graph_acc = 0
        graph_recall = 0
        gt_tree = scipy.spatial.cKDTree(gt_points)
        for c_i,thre in enumerate([5]):
            if len(graph_points):
                graph_tree = scipy.spatial.cKDTree(graph_points)
                dis_gt2graph,_ = graph_tree.query(gt_points, k=1)
                dis_graph2gt,_ = gt_tree.query(graph_points, k=1)
                graph_recall = len([x for x in dis_gt2graph if x<thre])/len(dis_gt2graph)
                graph_acc = len([x for x in dis_graph2gt if x<thre])/len(dis_graph2gt)
        r_f = 0
        if graph_acc*graph_recall:
            r_f = 2*graph_recall * graph_acc / (graph_acc+graph_recall)
        return graph_acc, graph_recall, r_f


    network.val_mode()
    eval_len = network.val_len()
    graph_acc_ave = 0
    graph_recall_ave=0
    r_f_ave = 0
    # =================working on an image=====================
    for i, data in enumerate(network.dataloader_valid):
        _, _, tiff, mask, _, name, init_points, end_points = data
        name = name[0][-14:-5]
        tiff = tiff.to(args.device)
        mask, init_points, end_points = mask[0], init_points[0], end_points[0]
        init_points = [[int(x[0]),int(x[1])] for x in init_points]
        # clear history info
        env.init_image(valid=True)
        with torch.no_grad():
            fpn_feature_map = network.encoder(tiff)
            # pre_seg_mask = network.decoder_seg(fpn_feature_map)
            fpn_feature_map = F.interpolate(fpn_feature_map, size=(1000, 1000), mode='bilinear', align_corners=True)
            fpn_feature_tensor = fpn_feature_map#torch.cat([tiff,fpn_feature_map],dim=1)
            # mask_to_save = torch.sigmoid(pre_seg_mask).squeeze(0).squeeze(0).cpu().detach().numpy()
            # output_img = mask_to_save
            # save_img = Image.fromarray( (output_img/np.max(output_img) )* 255) 
            # save_img.convert('RGB').save('./records/output_seg/{}.png'.format(name))

        # record generated vertices
        vertices_record_list = []
        if len(init_points):
            for v_idx,init_point in enumerate(init_points):
                # ===============working on a curb instance======================
                agent.init_agent(init_point)
                vertices_record_list.append(init_point)
                while 1:
                    agent.agent_step_counter += 1
                    # network predictions
                    cropped_feature_tensor = agent.crop_attention_region(fpn_feature_tensor,val_flag=True)
                    with torch.no_grad():
                        v_now = [x/1000 for x in agent.v_now]
                        v_now = torch.FloatTensor(v_now).unsqueeze(0).to(args.device)
                        v_previous = [x/1000 for x in agent.v_previous]
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
                        or (((agent.v_now[0]>=995) or (agent.v_now[0]<=5)or (agent.v_now[1]>=995) or (agent.v_now[1]<=5)) and agent.agent_step_counter > 10):
                        agent.taken_stop_action = 1
                        
                    if agent.taken_stop_action:    
                        break
                        
        # calculate metrics
        graph_acc, graph_recall, r_f = eval_metric(env.graph_record[0,0],mask,name)
        graph_acc_ave = (graph_acc_ave * i + graph_acc) / (i + 1)
        graph_recall_ave = (graph_recall_ave * i + graph_recall) / (i + 1)
        r_f_ave = (r_f_ave * i + r_f) / (i + 1)
        # vis and print
        visualization_image(env.graph_record,name,mask,vertices_record_list)
        print('Validation {}-{}/{} || graph {}/{}/{}'.format(name,i,eval_len,round(graph_acc,4)
                ,round(graph_recall,4),round(r_f_ave,4)))
        if args.test:
            with open('./records/test/vertices_record/{}.json'.format(name),'w') as jf:
                json.dump(vertices_record_list,jf)
        else:
            with open('./records/valid/vertices_record/{}.json'.format(name),'w') as jf:
                json.dump(vertices_record_list,jf)
    # finish
    print('----Finishing Validation || graph {}/{}'.format(round(graph_acc_ave,4),round(graph_recall_ave,4))) 
    if not args.test:   
        network.writer.add_scalars('Val/Val_Accuracy by image',{'graph_acc':graph_acc_ave},env.training_step)
        network.writer.add_scalars('Val/Val_Recall by image',{'graph_recall':graph_recall_ave},env.training_step)
        network.writer.add_scalars('Val/Val_F1 by image',{'graph_f1':r_f_ave},env.training_step)
    return r_f_ave

