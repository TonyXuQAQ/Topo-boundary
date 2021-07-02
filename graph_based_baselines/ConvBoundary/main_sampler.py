import numpy as np
import torch
import time
import random
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
import torch.nn.functional as F
from torchvision.transforms.functional_pil import crop

def run_explore(env,seq,pre_concat_map, seq_lens, mask,name, init_points, end_points,iCurb_image_index):
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
        if len(env.buffer['pre_coord']):
            pre_coords = torch.stack(env.buffer['pre_coord']).to(env.args.device).squeeze(1).squeeze(1)
            gt_coords = torch.FloatTensor(env.buffer['gt_coord']).to(env.args.device)
            loss_coord = env.network.criterions['bce'](pre_coords,gt_coords)
            env.network.bp(loss_coord)
            return loss_coord.item()
        return 0
        

    train_len = network.train_len()
    # load data
    init_points = [[int(x[0]),int(x[1])] for x in init_points]
    end_points = [[int(x[0]),int(x[1])] for x in end_points]
    instance_num = seq.shape[0]
    # init environment
    env.init_image()
    training_counter = 0
    loss_coord_ave = 0
    for instance_id in range(instance_num):
        # ========================= working on a curb instance =============================
        instance_vertices = seq[instance_id]
        agent.instance_vertices = instance_vertices[:seq_lens[instance_id]].copy()
        if len(agent.instance_vertices):
            init_vertex = init_points[instance_id]
            # init_vertex =  init_vertex + 0 * np.random.normal(0, 1, 2)
            agent.init_agent(init_vertex)
            agent.end_vertex = end_points[instance_id]
            # whether use gt_direction to crop ROI
            while 1:
                agent.agent_step_counter += 1
                # crop rectcoord centering v_now
                cropped_feature_tensor = agent.crop_attention_region(pre_concat_map)
                pre_position = network.ConvBoundaryAgent(cropped_feature_tensor)
                # generate labels
                gt_coord = env.expert_exploration(pre_position)
                if gt_coord is not None:
                    env.buffer['pre_coord'].append(pre_position)
                    env.buffer['gt_coord'].append(gt_coord)
                # stop action
                if (agent.agent_step_counter > args.max_length) \
                    or (((agent.v_now[0]>=999) or (agent.v_now[0]<=0)or (agent.v_now[1]>=999) or (agent.v_now[1]<=0)) and agent.agent_step_counter > 10):
                        agent.taken_stop_action = 1
                if len(env.buffer['pre_coord']) and (len(env.buffer['pre_coord']) >= env.buffer_size or (instance_id == instance_num-1 and agent.taken_stop_action)):
                    loss_coord = train_buffer()
                    loss_coord_ave += loss_coord * len(env.buffer['pre_coord'])
                    training_counter += len(env.buffer['pre_coord'])
                    env.buffer = {
                        'pre_coord':[],
                        'gt_coord':[],
                        }
                if agent.taken_stop_action:
                    break
    # visualization
    visualization_image(env.graph_record,env.epoch_counter,iCurb_image_index,name,mask)
    if training_counter:
        return loss_coord_ave / training_counter
    return 0