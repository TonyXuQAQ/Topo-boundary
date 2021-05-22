import numpy as np
import torch
import random
from PIL import Image
from scipy.spatial import cKDTree
import torch.nn.functional as F

def run_explore(env,seq,tiff, seq_lens, mask,name, init_points, end_points,iCurb_image_index):
    agent = env.agent
    network = env.network
    args = env.args

    def visualization_image(graph_record,epoch,iCurb_image_index,name,mask):
        graph_record = Image.fromarray(graph_record[0,0,:,:].cpu().detach().numpy() * 255).convert('RGB')
        mask = Image.fromarray(mask * 255).convert('RGB')
        dst = Image.new('RGB', (graph_record.width * 2, graph_record.height ))
        dst.paste(graph_record,(0,0))
        dst.paste(mask,(mask.width,0))
        dst.save('./records/train/vis/{}_{}_{}.png'.format(epoch,iCurb_image_index,name))

    def train_buffer():
        pre_coords = torch.stack(env.buffer['pre_coord']).to(env.args.device)
        gt_coords = torch.stack(env.buffer['gt_coord']).to(env.args.device)
        loss_coord = env.network.criterions['bce'](pre_coords,gt_coords)
        pre_segmentations = torch.stack(env.buffer['pre_seg']).to(env.args.device)
        gt_segmentations = torch.stack(env.buffer['gt_seg']).to(env.args.device)
        loss_seg = env.network.criterions['bce'](pre_segmentations,gt_segmentations)
        env.network.bp(loss_coord + loss_seg * 5)
        return loss_coord.item(), loss_seg.item()

    train_len = network.train_len()
    # load data
    init_points = [[int(x[0]),int(x[1])] for x in init_points]
    end_points = [[int(x[0]),int(x[1])] for x in end_points]
    instance_num = seq.shape[0]
    # init environment
    env.init_image()
    coord_counter = 0
    stop_action_counter = 0
    loss_coord_ave = 0
    loss_seg_ave = 0
    loss_stop_action_ave = 0
    for instance_id in range(instance_num):
        # ========================= working on a curb instance =============================
        instance_vertices = seq[instance_id]
        agent.instance_vertices = instance_vertices[:seq_lens[instance_id]].copy()
        if len(agent.instance_vertices):
            init_vertex = init_points[instance_id]
            # init_vertex =  init_vertex + 0 * np.random.normal(0, 1, 2)
            agent.init_agent(init_vertex)
            agent.end_vertex = end_points[instance_id]
            while 1:
                agent.agent_step_counter += 1
                stop_action_counter += 1
                # crop rectcoord centering v_now
                cropped_feature_tensor, gt_segmentation = agent.crop_attention_region(tiff,mask)
                # 
                pre_segmentation, pre_coord = network.vecRoadNet(cropped_feature_tensor)
                # generate labels
                if env.epoch_counter == 1 and env.training_step <= args.teacher_forcing_number:
                    gt_coord = env.expert_exploration(pre_coord,cropped_feature_tensor=cropped_feature_tensor,teacher_forcing=True)
                else:
                    gt_coord = env.expert_exploration(pre_coord,cropped_feature_tensor=cropped_feature_tensor)
                # save to buffer
                if gt_coord is not None:
                    env.buffer['pre_coord'].append(pre_coord)
                    env.buffer['gt_coord'].append(gt_coord)
                    env.buffer['pre_seg'].append(pre_segmentation)
                    env.buffer['gt_seg'].append(gt_segmentation)
                # stop action
                if (agent.agent_step_counter > args.max_length) \
                    or (((agent.v_now[0]>=999) or (agent.v_now[0]<=0)or (agent.v_now[1]>=999) or (agent.v_now[1]<=0)) and agent.agent_step_counter > 10):
                        agent.taken_stop_action = 1
                # training
                if len(env.buffer['pre_coord']) and (len(env.buffer['pre_coord']) >= env.buffer_size or (instance_id == instance_num-1 and agent.taken_stop_action)):
                    loss_coord, loss_seg = train_buffer()
                    coord_counter += len(env.buffer['pre_coord']) 
                    stop_action_counter += len(env.buffer['pre_seg']) 
                    loss_coord_ave += loss_coord * len(env.buffer['pre_seg']) 
                    loss_seg_ave += loss_seg * len(env.buffer['pre_seg'])
                    env.buffer = {'pre_coord':[],'pre_seg':[],'gt_coord':[],'gt_seg':[]} 
                if agent.taken_stop_action:
                    break
    # visualization
    visualization_image(env.graph_record,env.epoch_counter,iCurb_image_index,name,mask)
    if coord_counter == 0:
        return 0,0, 0
    else:
        return loss_coord_ave / coord_counter, 0, loss_seg_ave / coord_counter
    