from config import get_args
import warnings
from sild.utils import *
import os
import json
from os import path as osp
warnings.simplefilter("ignore")

# args/seed/data
args = get_args()
setup_seed(args.seed)
args, data = load_synthetic_data(args)
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
COLLECTOR.add_GPU_MEM(args.device,id=False)
COLLECTOR.save_all_time()
COLLECTOR.save(os.path.join(log_dir,'collector.json'))

