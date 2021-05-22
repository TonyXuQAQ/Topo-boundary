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
        #
        pre_actions = torch.stack(env.buffer['pre_action']).to(env.args.device).squeeze(1)
        gt_actions = torch.stack(env.buffer['gt_action']).to(env.args.device).squeeze(1)
        loss_action = env.network.criterions['ce'](pre_actions,gt_actions)
        if len(env.buffer['pre_angle']):
            pre_angles = torch.stack(env.buffer['pre_angle']).to(env.args.device).squeeze(1)
            gt_angles = torch.stack(env.buffer['gt_angle']).to(env.args.device).squeeze(1)
            loss_angle = env.network.criterions['ce'](pre_angles,gt_angles)
            #
            pre_segmentations = torch.stack(env.buffer['pre_seg']).to(env.args.device)
            gt_segmentations = torch.stack(env.buffer['gt_seg']).to(env.args.device)
            loss_seg = env.network.criterions['bce'](pre_segmentations,gt_segmentations)
            env.network.bp(loss_angle * 10 + loss_seg * 50 + loss_action)
            return loss_angle.item(), loss_seg.item(), loss_action.item()
        else:
            env.network.bp(loss_action)
            return 0, 0, loss_action.item()

    train_len = network.train_len()
    # load data
    init_points = [[int(x[0]),int(x[1])] for x in init_points]
    end_points = [[int(x[0]),int(x[1])] for x in end_points]
    instance_num = seq.shape[0]
    # init environment
    env.init_image()
    angle_counter = 0
    stop_action_counter = 0
    loss_angle_ave = 0
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
                # crop rectangle centering v_now
                cropped_feature_tensor, gt_segmentation = agent.crop_attention_region(tiff,mask)
                # 
                v_now = [x/(env.crop_size) for x in agent.v_now]
                v_now = torch.FloatTensor(v_now).unsqueeze(0).to(args.device)
                pre_segmentation, pre_angle, pre_stop_action = network.roadTracerNet(cropped_feature_tensor,v_now)
                pre_stop_action = pre_stop_action.reshape(pre_stop_action.shape[0],-1)
                pre_angle = pre_angle.reshape(pre_angle.shape[0],-1)

                if env.epoch_counter == 1 and env.args.pretrain:
                    gt_angle, gt_stop_action = env.expert_exploration(pre_angle,cropped_feature_tensor=cropped_feature_tensor,teacher_forcing=True)
                else:
                    gt_angle, gt_stop_action = env.expert_exploration(pre_angle,cropped_feature_tensor=cropped_feature_tensor)
                if gt_angle is not None:
                    env.buffer['pre_angle'].append(pre_angle)
                    env.buffer['gt_angle'].append(torch.LongTensor([gt_angle]).to(args.device))
                    env.buffer['pre_seg'].append(pre_segmentation)
                    env.buffer['gt_seg'].append(gt_segmentation)
                env.buffer['pre_action'].append(pre_stop_action)
                env.buffer['gt_action'].append(torch.LongTensor([gt_stop_action]).to(args.device))
                #
                if (agent.agent_step_counter > args.max_length) \
                    or (((agent.v_now[0]>=999) or (agent.v_now[0]<=0)or (agent.v_now[1]>=999) or (agent.v_now[1]<=0)) and agent.agent_step_counter > 10):
                        agent.taken_stop_action = 1
                if len(env.buffer['pre_action']) and (len(env.buffer['pre_action']) >= env.buffer_size or (agent.taken_stop_action)):
                    loss_angle, loss_seg, loss_action = train_buffer()
                    angle_counter += len(env.buffer['pre_angle']) 
                    stop_action_counter += len(env.buffer['pre_seg']) 
                    loss_angle_ave += loss_angle * len(env.buffer['pre_angle']) 
                    loss_seg_ave += loss_seg * len(env.buffer['pre_seg']) 
                    loss_stop_action_ave += loss_action * len(env.buffer['pre_action'])
                    env.buffer = {'pre_angle':[],'pre_seg':[],'gt_angle':[],'gt_seg':[],'pre_action':[],'gt_action':[]}
                if agent.taken_stop_action:
                    break
    # visualization
    if args.visualization:
        visualization_image(env.graph_record,env.epoch_counter,iCurb_image_index,name,mask)
    if angle_counter == 0:
        return 0,loss_stop_action_ave / stop_action_counter, loss_seg_ave / angle_counter
    else:
        return loss_angle_ave / angle_counter, loss_stop_action_ave / stop_action_counter, loss_seg_ave / angle_counter
    