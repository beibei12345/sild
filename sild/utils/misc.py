import random
import numpy as np
import torch
import os 
def setup_seed(seed: int):
    r"""Sets the seed for generating random numbers in PyTorch, numpy and
    Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import math
def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

def convert_time(size_bytes):
    if size_bytes == 0:
        return "0S"
    size_name = ("S", "M", "H")
    base = 60 
    if size_bytes< base:
        i = 0
    else:
        i = int(math.floor(math.log(size_bytes, base)))
    i = min(i,2)
    p = math.pow(60, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'Timing : Func {f.__name__} took: {convert_time(te-ts)}')
        return result
    return wrap

def move_to(obj, device):
    if isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    elif isinstance(obj,tuple):
        return tuple(move_to(v, device) for v in obj)
    if hasattr(obj, 'to'):
        return obj.to(device)
    return obj

class EarlyStopping:
    """ EarlyStopping class to keep NN from overfitting. copied from nni
    if mode=='min' : lower the better
    function step support metric_dict and metric_name is the main metric
    property best_metrics is the best metric_dict
    """
    def __init__(self, metric_name, mode="max", min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)
        self.metric_name = metric_name
        self.best_metrics = {}
        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step_one(self, metrics):
        """ EarlyStopping step on each epoch
        @params metrics: metric value
        @return : True if stop
        """

        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False
    
    def step(self, **kwargs):
        main_metric = kwargs[self.metric_name]
        # record all
        if not self.best or self.is_better(main_metric, self.best):
            self.best_metrics = kwargs
        # only main metric judge
        to_stop = self.step_one(main_metric)
        return to_stop
    
    def better(self, **kwargs):
        main_metric = kwargs[self.metric_name]
        return not self.best or self.is_better(main_metric, self.best)
    
    def reset(self):
        self.best = None
        self.best_metrics = {}

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)

import os.path as osp

def get_arg_dict(args):
    info_dict=args.__dict__
    ks=list(info_dict.keys())
    arg_dict={}
    for k in ks:
        v=info_dict[k]
        for t in [int,float,str,bool,torch.Tensor]:
            if isinstance(v,t):
                arg_dict[k]=v
                break
    return arg_dict

class DummyArgs:
    def __init__(self,**kwargs) -> None:
        for k,v in kwargs.items():
            setattr(self,k,v)

from sklearn.metrics import f1_score,accuracy_score
def cal_metric(preds, target, args):
    if args.metric == 'acc':
        auc = (preds.cpu() == target.cpu()).sum().item() / target.shape[0]
        auc = float(auc)
    elif args.metric == 'auc':
        auc = f1_score(target.cpu().numpy(),preds.cpu().numpy(),average='macro')
    else:
        raise NotImplementedError(f'metric {args.metric} not implemented')
    return auc