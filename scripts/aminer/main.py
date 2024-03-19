from config import get_args
import warnings
from sild.utils import COLLECTOR
from sild.utils import setup_seed, get_arg_dict, move_to
from sild.utils import load_synthetic_data
import os
import json
from os import path as osp
import torch
warnings.simplefilter("ignore")

# args/seed/data
args = get_args()
setup_seed(args.seed)
args.dataset = 'aminer'
args.testlength = 3
args.vallength = 3
args.length = 17
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.abspath(f'{CUR_DIR}/../../data/Aminer/processed_data_128.pt')
data = torch.load(data_file)
args.nfeat = data['x'].shape[1]
args.num_nodes = len(data['x'])
args.num_classes = max(data['y']).item() + 1
print(f'Loading dataset {args.dataset}')
data = move_to(data, args.device)

# pre-logs
log_dir = args.log_dir
info_dict = get_arg_dict(args)
json.dump(info_dict,open(os.path.join(log_dir,'args.json'),'w'),indent=2)
print(args)

# Runner
from model import SGCNNet, Trainer
model = SGCNNet(args, data).to(args.device)
trainer = Trainer(model, data, args)
results = trainer.train_till_end(data)

# post-logs
measure_dict=results
info_dict.update(measure_dict)
json.dump(info_dict, open(osp.join(log_dir, 'info.json'), 'w'))
COLLECTOR.add_GPU_MEM(args.device, id=False)
COLLECTOR.save_all_time()
COLLECTOR.save(os.path.join(log_dir,'collector.json'))

