import argparse
import shutil
import os

import argparse
import shutil
import os
import yaml

def get_parser():
    def process(**params):    # pass in variable numbers of args
        for key, value in params.items():
            parser.add_argument('--'+key, default=value)

    parser = argparse.ArgumentParser()
    
    with open('./dataset/config_dir.yml', 'r') as f:
        conf = yaml.safe_load(f.read())    # load the config file
    process(**conf)
    
    with open('./config.yml', 'r') as f:
        conf = yaml.safe_load(f.read())    # load the config file
    process(**conf) 

    return parser

def update_dir(args):
    def check_and_add_dir(dir_path,clear=True):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            if clear:
                shutil.rmtree(dir_path)
                os.makedirs(dir_path)
                
    check_and_add_dir('./checkpoints',clear=False)
    check_and_add_dir('./records/tensorboard')
    check_and_add_dir('./records/train/vis')
    check_and_add_dir('./records/valid/seg')
    check_and_add_dir('./records/valid/vis')
    check_and_add_dir('./records/valid/vertices_record')
    check_and_add_dir('./records/test/skeleton',clear=False)
    check_and_add_dir('./records/test/graph',clear=False)
    check_and_add_dir('./records/test/final_vis',clear=False)
    check_and_add_dir('./records/test/vertices_record',clear=False)
