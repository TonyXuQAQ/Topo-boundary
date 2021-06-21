import numpy as np
import torch
import random
from PIL import Image
from scipy.spatial import cKDTree
import torch.nn.functional as F

def run_explore(env,seq,cat_feature_map, seq_lens, mask,name, init_points, end_points,iCurb_image_index):
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
        loss = 0
        
        loss_state_num = 0
        pre_states = torch.stack(env.buffer['pre_state']).to(env.args.device).squeeze(1)
        gt_states = torch.LongTensor(env.buffer['gt_state']).to(env.args.device).squeeze(1)
        loss_state = env.network.criterions['ce'](pre_states,gt_states)
        loss_state_num = loss_state.item()
        loss += loss_state
        #
        loss_direction_num = 0
        if len(env.buffer['pre_direction']):
            pre_directions = torch.stack(env.buffer['pre_direction']).to(env.args.device).squeeze(1)
            gt_directions = torch.FloatTensor(env.buffer['gt_direction']).to(env.args.device)
            loss_direction = torch.sum(1 - env.network.criterions['cos'](pre_directions,gt_directions))/len(env.buffer['pre_direction'])
            loss_direction_num = loss_direction.item()
            loss += loss_direction * 10

        loss_coord_num = 0
        if len(env.buffer['pre_coord']):    
            pre_coords = torch.stack(env.buffer['pre_coord']).to(env.args.device).squeeze(1)
            gt_coords = torch.FloatTensor(env.buffer['gt_coord']).to(env.args.device).unsqueeze(1)
            loss_coord = env.network.criterions['bce'](pre_coords,gt_coords)
            loss_coord_num = loss_coord.item()
            loss += loss_coord * 10

        env.network.bp(loss)
        return loss_coord_num, loss_direction_num, loss_state_num
        

    train_len = network.train_len()
    # load data
    init_points = [[int(x[0]),int(x[1])] for x in init_points]
    end_points = [[int(x[0]),int(x[1])] for x in end_points]
    instance_num = seq.shape[0]
    # init environment
    env.init_image()
    coord_counter = 0
    stop_action_counter = 0
    direction_counter = 0
    coord_counter = 0
    loss_coord_ave = 0
    loss_direction_ave = 0
    loss_state_ave = 0
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
            next_direction = None
            while 1:
                agent.agent_step_counter += 1
                stop_action_counter += 1
                # crop rectcoord centering v_now
                cropped_feature_tensor = agent.crop_attention_region(cat_feature_map)
                pre_state = network.DAGMapperSH(cropped_feature_tensor)
                pre_state = pre_state.reshape(pre_state.shape[0],-1)
                pre_position = network.DAGMapperPH(cropped_feature_tensor)
                # generate labels
                gt_coord_map = env.expert_labels(pre_position,pre_state)
                env.buffer['pre_state'].append(pre_state)
                env.buffer['gt_state'].append([agent.gt_stop_action])
                if gt_coord_map is not None:
                    env.buffer['pre_coord'].append(pre_position)
                    env.buffer['gt_coord'].append(gt_coord_map)
                    coord_counter += 1
                # stop action
                if (agent.agent_step_counter > args.max_length) \
                    or (((agent.v_now[0]>=999) or (agent.v_now[0]<=0)or (agent.v_now[1]>=999) or (agent.v_now[1]<=0)) and agent.agent_step_counter > 10):
                        agent.taken_stop_action = 1
                if len(env.buffer['pre_state']) and (len(env.buffer['pre_state']) >= env.buffer_size or (instance_id == instance_num-1 and agent.taken_stop_action)):
                    loss_coord, loss_direction, loss_state = train_buffer()
                    if len(env.buffer['pre_state']) >= env.buffer_size:
                        loss_coord_ave += loss_coord * env.buffer_size
                        loss_direction_ave += loss_direction * env.buffer_size
                        loss_state_ave += loss_state * env.buffer_size
                    else:
                        loss_coord_ave += loss_coord * coord_counter
                        loss_direction_ave += loss_direction * direction_counter
                        loss_state_ave += loss_state * stop_action_counter
                    env.buffer = {'pre_direction':[],
                        'gt_direction':[],
                        'pre_coord':[],
                        'gt_coord':[],
                        'pre_state':[],
                        'gt_state':[]}
                if agent.taken_stop_action:
                    break
    # visualization
    if args.visualization:
        visualization_image(env.graph_record,env.epoch_counter,iCurb_image_index,name,mask)
    loss_direction_ave = 0 if direction_counter == 0 else loss_direction_ave / direction_counter
    loss_coord_ave = 0 if coord_counter == 0 else loss_coord_ave / coord_counter
    return loss_coord_ave, loss_direction_ave, loss_state_ave / stop_action_counter