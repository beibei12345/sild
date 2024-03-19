import os
import numpy as np
import torch

import os
import matplotlib.pyplot as plt
dir, fname = os.path.abspath(__file__).split(os.sep)[-2:]
DATA_dir = os.path.abspath(os.path.join(dir,"../../../data/"))

def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def prepare_dir(output_folder):
    mkdirs(output_folder)
    log_folder = mkdirs(output_folder)
    return log_folder

def load_synthetic_data(args):
    dataset = args.dataset
    print(f'Loading dataset {dataset}')
    datafile = os.path.join(DATA_dir,'synthetic_SBM',args.dataset,"data")
    data = torch.load(datafile)
    for i, mode in enumerate("train val test".split()):
        data[f'{mode}_mask'] = (data['env']==i).squeeze()
    args.nfeat=data['x'].shape[1]
    args.num_nodes = data['x'].shape[0]
    args.length = len(data['edge_index'])
    args.num_classes = max(data['y']).item() + 1
    return args,data

    