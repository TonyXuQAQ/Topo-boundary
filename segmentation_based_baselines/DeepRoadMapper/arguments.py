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
    check_and_add_dir('./records/segmentation/{}'.format(args.mode[6:]),clear=False)
    check_and_add_dir('./records/skeleton/{}'.format(args.mode[6:]),clear=False)

def update_dir_candidate_train(args):
    check_and_add_dir('./records/candidate_train/good')
    check_and_add_dir('./records/candidate_train/bad')

def update_dir_candidate_valid(args):
    check_and_add_dir('./records/candidate_valid/good')
    check_and_add_dir('./records/candidate_valid/bad')

def update_dir_candidate_test(args):
    check_and_add_dir('./records/candidate_test/reason')
    check_and_add_dir('./records/candidate_test/rgb')
    check_and_add_dir('./records/candidate_test/json')

def update_dir_reason_train(args):
    check_and_add_dir('./records/tensorboard/reason')
    check_and_add_dir('./records/reason/valid/vis')

def update_dir_reason_test(args):
    check_and_add_dir('./records/reason/test/vis')
    check_and_add_dir('./records/reason/test/graph')
    check_and_add_dir('./records/reason/test/final_vis')

