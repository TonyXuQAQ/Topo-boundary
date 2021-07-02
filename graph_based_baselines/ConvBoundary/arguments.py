import argparse
import shutil
import os
import yaml

def get_parser():
    def process(**params):    # pass in variable numbers of args
        for key, value in params.items():
            parser.add_argument('--'+key, default=value, type=type(value))

    parser = argparse.ArgumentParser()
    
    with open('./dataset/config_dir.yml', 'r') as f:
        conf = yaml.safe_load(f.read())    # load the config file
    process(**conf)
    
    with open('./config.yml', 'r') as f:
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

def update_dir(args):
    check_and_add_dir('./records/tensorboard')
    check_and_add_dir('./records/train/vis')
    check_and_add_dir('./records/valid/seg')
    check_and_add_dir('./records/valid/vis')
    check_and_add_dir('./records/valid/vertices_record')
    
def update_dir_test(args):
    check_and_add_dir('./records/test/skeleton')
    check_and_add_dir('./records/test/graph')
    check_and_add_dir('./records/test/final_vis')
    check_and_add_dir('./records/test/vertices_record')

def update_dir_seg_train(args):
    check_and_add_dir('./records/tensorboard/seg')
    check_and_add_dir('./records/seg/valid/segmentation')
    check_and_add_dir('./records/seg/valid/direction')

def update_dir_seg_test(args):
    check_and_add_dir('./records/label/inv/{}'.format(args.mode[6:]))
    check_and_add_dir('./records/label/direction/{}'.format(args.mode[6:]))
