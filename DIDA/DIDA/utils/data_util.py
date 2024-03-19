import os
from symbol import shift_expr
import numpy as np
import torch
from torch_geometric.utils import train_test_split_edges, add_self_loops
from torch_geometric.data import Data
import pickle
from .mutils import seed_everything
def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def prepare_dir(output_folder):
    mkdirs(output_folder)
    log_folder = mkdirs(output_folder)
    return log_folder

def select_by_field(edges,fields=[0,1]):
    # field [0,1,2,3,4]
    res=[]
    for f in fields:
        e=edges[edges[:,4]==f]
        res.append(e)
    edges=torch.concat(res,dim=0)
    res=[]
    for i in range(16):
        e=edges[edges[:,2]==i]
        e=e[:,:2]
        res.append(e)
    edges=res
    return edges

def select_by_venue(edges,venues=[0,1]):
    # venue [0-21]
    res=[]
    for f in venues:
        e=edges[edges[:,3]==f]
        res.append(e)
    edges=torch.concat(res,dim=0)
    res=[]
    for i in range(16):
        e=edges[edges[:,2]==i]
        e=e[:,:2]
        res.append(e)
    edges=res
    return edges

def turn_undirected(edges): # original is undirected
    # edges = torch.cat([edges, edges[[1, 0],:]], dim=1)
    edges = add_self_loops(edges)[0]
    return edges

def load_data(args):
    seed_everything(0)
    dataset = args.dataset
    print(f'Loading dataset {dataset}')
    if dataset == 'collab':
        from ..data_configs.collab import (testlength,vallength,length,split,processed_datafile)
        args.dataset = dataset
        args.testlength = testlength
        args.vallength = vallength
        args.length = length
        args.split = split
        data = torch.load(f'{processed_datafile}-{split}')
        args.nfeat=data['x'].shape[1]
        args.num_nodes = len(data['x'])

    elif dataset == 'yelp':
        from ..data_configs.yelp import (testlength,vallength,length,split,processed_datafile,shift,num_nodes)
        args.dataset = dataset
        args.testlength = testlength
        args.vallength = vallength
        args.length = length
        args.split = split
        args.shift = shift
        args.num_nodes = num_nodes
        data = torch.load(f'{processed_datafile}-{split}')
        args.nfeat=data['x'].shape[1]
        args.num_nodes = len(data['x'])
        # turn undirected and add self-loop
        # if args.undirected:
        #     data['train']["edge_index_list"] = [ turn_undirected(x) for x in data['train']["edge_index_list"]]
        #     data['test']["edge_index_list"] = [ turn_undirected(x) for x in data['test']["edge_index_list"]]
        
    elif 'synthetic' in dataset:
        from ..data_configs.synthetic import (testlength,vallength,synthetic_file,P,SIGMA,TEST_P,TEST_SIGMA)
        args.testlength=testlength
        args.vallength=vallength
        P=dataset.split('-')
        P=float(P[-1]) if len(P)>1 else 0.6
        args.dataset=f'synthetic-{P}' 
        args.P=P
        args.SIGMA=SIGMA
        args.TEST_P=TEST_P
        args.TEST_SIGMA=TEST_SIGMA
        datafile=f'{synthetic_file}-{P,SIGMA,TEST_P,TEST_SIGMA}'
        data = torch.load(datafile)
        args.nfeat=data['x'][0].shape[1]
        args.num_nodes = len(data['x'][0])
        args.length=len(data['x'])
    else:
        raise NotImplementedError(f'Unknown dataset {dataset}')
    for mode in 'train test'.split():
        data[mode]['edge_index_list'] = [x.long() for x in data[mode]['edge_index_list']]
        data[mode]['pedges'] = [x.long() for x in data[mode]['pedges']]
        data[mode]['nedges'] = [x.long() for x in data[mode]['nedges']]        
    return args,data

    