import json
import random


def split_dataset():
    with open('./data_split.json','r') as jf:
        data = json.load(jf)
    train_len = len(data['train'])
    valid_len = len(data['valid'])
    test_len = len(data['test'])
    # train list
    all_images = data['train'] + data['valid'] + data['test'] + data['pretrain']
    random.shuffle(all_images)
    with open('./data_split_previous.json','w') as jf:
        json.dump(data,jf)
    with open('./scripts/data_split.json','w') as jf:
        json.dump({'train':all_images[:train_len],
                    'valid':all_images[train_len:train_len+valid_len],
                    'test':all_images[train_len+valid_len:train_len+valid_len+test_len],
                    'pretrain':all_images[train_len+valid_len+test_len:]
                    },jf)

split_dataset()