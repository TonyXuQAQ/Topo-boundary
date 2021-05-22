import argparse
import shutil
import os
import yaml

def get_parser(net_choice=''):
    def process(**params):    # pass in variable numbers of args
        for key, value in params.items():
            if key==net_choice:
                for key2, value2 in value.items():
                    parser.add_argument('--'+key2, default=value2)
            else:    
                parser.add_argument('--'+key, default=value)

    parser = argparse.ArgumentParser()
    
    with open('./config.yml', 'r') as f:
        conf = yaml.safe_load(f.read())    # load the config file
    process(**conf) 

    with open('./dataset/config_dir.yml', 'r') as f:
        conf = yaml.safe_load(f.read())    # load the config file
    process(**conf)
    return parser



def check_and_add_dir(dir_path,clear=True):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            if clear:
                shutil.rmtree(dir_path)
                os.makedirs(dir_path)

def update_dir_seg_train(args):
    check_and_add_dir('./records/tensorboard/seg')
    check_and_add_dir('./records/seg/valid/segmentation')

def update_dir_seg_test(args):
    check_and_add_dir('./records/segmentation/{}'.format(args.mode[6:]))

def update_dir_candidate(args):
    check_and_add_dir('./records/corrupted_mask',clear=False)

def update_dir_refine_train(args):
    check_and_add_dir('./records/tensorboard/refine')
    check_and_add_dir('./records/refine/valid/vis')

def update_dir_refine_pretrain(args):
    check_and_add_dir('./records/tensorboard/refine_pretrain')
    check_and_add_dir('./records/refine_pretrain/valid/vis')

def update_dir_refine_test(args):
    check_and_add_dir('./records/refine/test/refined_seg')
    check_and_add_dir('./records/refine/test/final_vis')
    check_and_add_dir('./records/refine/test/skeleton')
    check_and_add_dir('./records/refine/test/graph')

