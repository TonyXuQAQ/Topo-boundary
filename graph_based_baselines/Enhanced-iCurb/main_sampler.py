import numpy as np
import torch
import random
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
import torch.nn.functional as F

def run_restricted_explore(env,seq,fpn_feature_tensor, seq_lens, mask, orientation_map, name, init_points, end_points,iCurb_image_index,num):
    agent = env.agent
    network = env.network
    args = env.args
    train_len = network.train_len()
    def visualization_image(graph_record,epoch,iCurb_image_index,name,mask,num,vertices):
        graph_record = Image.fromarray(graph_record[0,0,:,:].cpu().detach().numpy() * 255).convert('RGB')
        draw = ImageDraw.Draw(graph_record)
        for vertex in vertices:
            draw.ellipse((vertex[1]-1,vertex[0]-1,vertex[1]+1,vertex[0]+1),fill='yellow',outline='yellow')
        mask = Image.fromarray(mask * 255).convert('RGB')
        dst = Image.new('RGB', (graph_record.width * 2, graph_record.height ))
        dst.paste(graph_record,(0,0))
        dst.paste(mask,(mask.width,0))
        dst.save('./records/train/vis/restricted_exploration/{}_{}_{}_{}.png'.format(epoch,iCurb_image_index,name,num))

    # load data
    init_points = [[int(x[0]),int(x[1])] for x in init_points]
    end_points = [[int(x[0]),int(x[1])] for x in end_points]
    instance_num = seq.shape[0]
    # init environment
    env.init_image()
    vis_generated_vertices = []
    for instance_id in range(instance_num):
        # ========================= working on a curb instance =============================
        instance_vertices = seq[instance_id]
        agent.instance_vertices = instance_vertices[:seq_lens[instance_id]].copy()
        if len(agent.instance_vertices):
            init_vertex = init_points[instance_id]
            # init_vertex =  init_vertex + 0 * np.random.normal(0, 1, 2)
            agent.init_agent(init_vertex)
            agent.end_vertex = end_points[instance_id]
            vis_generated_vertices.append(init_vertex)
            while 1:
                agent.agent_step_counter += 1
                # crop rectangle centering v_now
                cropped_feature_tensor = agent.crop_attention_region(fpn_feature_tensor)
                with torch.no_grad():
                    v_now = [x/1000 for x in agent.v_now]
                    v_now = torch.FloatTensor(v_now).unsqueeze(0).to(args.device)
                    v_previous = [x/1000 for x in agent.v_previous]
                    v_previous = torch.FloatTensor(v_previous).unsqueeze(0).to(args.device)
                    pre_stop_action = network.decoder_stop(cropped_feature_tensor,v_now,v_previous)
                    pre_coord = network.decoder_coord(cropped_feature_tensor,v_now,v_previous)
                pre_stop_action = pre_stop_action.squeeze(1)
                pre_coord = pre_coord.squeeze(0).squeeze(0)
                
                env.expert_restricted_exploration(pre_coord,cropped_feature_tensor,orientation_map,True)
                vis_generated_vertices.append(agent.v_now)
                if (agent.agent_step_counter > 120) \
                    or (((agent.v_now[0]>=999) or (agent.v_now[0]<=0)or (agent.v_now[1]>=999) or (agent.v_now[1]<=0)) and agent.agent_step_counter > 10):
                        agent.taken_stop_action = 1
                if agent.taken_stop_action:
                    break
    # visualization
    if args.visualization:
        visualization_image(env.graph_record,env.epoch_counter,iCurb_image_index,name,mask,num,vis_generated_vertices)

def run_free_explore(env,seq,fpn_feature_tensor, seq_lens, mask, orientation_map, name, init_points, end_points,iCurb_image_index,num):
    agent = env.agent
    network = env.network
    args = env.args

    def visualization_image(graph_record,epoch,iCurb_image_index,name,mask,num,vertices):
        graph_record = Image.fromarray(graph_record[0,0,:,:].cpu().detach().numpy() * 255).convert('RGB')
        draw = ImageDraw.Draw(graph_record)
        for vertex in vertices:
            draw.ellipse((vertex[1]-1,vertex[0]-1,vertex[1]+1,vertex[0]+1),fill='yellow',outline='yellow')
        mask = Image.fromarray(mask * 255).convert('RGB')
        dst = Image.new('RGB', (graph_record.width * 2, graph_record.height ))
        dst.paste(graph_record,(0,0))
        dst.paste(mask,(mask.width,0))
        dst.save('./records/train/vis/free_exploration/{}_{}_{}_{}.png'.format(epoch,iCurb_image_index,name,num))

    train_len = network.train_len()
    # load data
    init_points = [[int(x[0]),int(x[1])] for x in init_points]
    end_points = [[int(x[0]),int(x[1])] for x in end_points]
    instance_num = seq.shape[0]
    # init environment
    env.init_image()
    vis_generated_vertices = []
    for instance_id in range(instance_num):
        # ========================= working on a curb instance =============================
        instance_vertices = seq[instance_id]
        agent.instance_vertices = instance_vertices[:seq_lens[instance_id]].copy()
        if len(agent.instance_vertices):
            init_vertex = init_points[instance_id]
            # init_vertex =  init_vertex + 0 * np.random.normal(0, 1, 2)
            agent.init_agent(init_vertex)
            agent.end_vertex = end_points[instance_id]
            vis_generated_vertices.append(init_vertex)
            while 1:
                agent.agent_step_counter += 1
                # crop rectangle centering v_now
                cropped_feature_tensor = agent.crop_attention_region(fpn_feature_tensor)
                with torch.no_grad():
                    v_now = [x/1000 for x in agent.v_now]
                    v_now = torch.FloatTensor(v_now).unsqueeze(0).to(args.device)
                    v_previous = [x/1000 for x in agent.v_previous]
                    v_previous = torch.FloatTensor(v_previous).unsqueeze(0).to(args.device)
                    pre_stop_action = network.decoder_stop(cropped_feature_tensor,v_now,v_previous)
                    pre_coord = network.decoder_coord(cropped_feature_tensor,v_now,v_previous)
                pre_stop_action = pre_stop_action.squeeze(1)
                pre_coord = pre_coord.squeeze(0).squeeze(0)
                
                env.expert_free_exploration(pre_coord,cropped_feature_tensor,orientation_map,True)
                vis_generated_vertices.append(agent.v_now)
                if (agent.agent_step_counter > args.max_length) \
                    or (((agent.v_now[0]>=999) or (agent.v_now[0]<=0)or (agent.v_now[1]>=999) or (agent.v_now[1]<=0)) and agent.agent_step_counter > 10):
                        agent.taken_stop_action = 1
                if agent.taken_stop_action:
                    break
    # visualization
    if args.visualization:
        visualization_image(env.graph_record,env.epoch_counter,iCurb_image_index,name,mask,num,vis_generated_vertices)
