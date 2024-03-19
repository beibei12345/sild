import argparse
import torch
import os

def get_args(args=None):
    parser = argparse.ArgumentParser()
    # 1.dataset
    parser.add_argument('--dataset', type=str, default='collab', help='datasets')
    parser.add_argument('--num_nodes', type=int, default=-1, help='num of nodes')
    parser.add_argument('--nfeat', type=int, default=128, help='dim of input feature')
    parser.add_argument('--num_classes', type=int, default=-1, help='')
    parser.add_argument('--length', type=int, default=-1, help='')
    parser.add_argument('--testlength', type=int, default=3, help='length for test')
    parser.add_argument('--P',type=float,default=0.5)
    parser.add_argument('--SIGMA',type=float,default=0.3)
    parser.add_argument('--TEST_P',type=float,default=-0.8)
    parser.add_argument('--TEST_SIGMA',type=float,default=0.1)
    parser.add_argument('--use_cfg',type=int,default=1)
    
    # 1.5 exp misc 
    parser.add_argument('--output_folder', type=str, default='', help='need to be modified')
    parser.add_argument('--log_dir',type=str,default="EXP")
    parser.add_argument('--log_interval', type=int, default=20, help='')
    
    # 2.experiments
    parser.add_argument('--max_epoch', type=int, default=1000, help='number of epochs to train.')
    parser.add_argument('--min_epoch', type=int, default=0, help='min epoch')
    parser.add_argument('--device', type=str, default='cpu', help='training device')
    parser.add_argument('--device_id', type=str, default='0', help='device id for gpu')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--patience', type=int, default=50, help='patience for early stop')
    parser.add_argument('--sampling_times', type=float, default=1, help='negative sampling times')

    # 3. params
    
    parser.add_argument('--nhid', type=int, default=8, help='dim of hidden embedding') # 8
    parser.add_argument('--n_layers',type=int,default=2)

    parser.add_argument('--undirected',type=int,default=0)
    
    parser.add_argument('--attn_drop',type=float,default=0)
    parser.add_argument('--temp_attn_res',type=int,default=1)
    

    # 4. special
    parser.add_argument('--window_size', type=int, default=5, help='')
    parser.add_argument('--spec_len',type=int,default=-1)
    parser.add_argument('--mtype', type=int, default=0, help='')
    
    # type
    parser.add_argument('--ctype',type=int,default=0)
    parser.add_argument('--ltype', type=int, default=0, help='')
    parser.add_argument('--use_filt', type=int, default=1, help='')
    
    
    # searchable args
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-7, help='weight for L2 loss on basic models.')
    parser.add_argument('--heads', type=int, default=4, help='attention heads.') # 4
    parser.add_argument('--norm',type=int,default=1)
    parser.add_argument('--use_RTE', type=int, default=1, help='')
    parser.add_argument('--spec_RTE', type=int, default=1, help='')
    parser.add_argument('--spec_res', type=int, default=1, help='')

    parser.add_argument('--backbone',type=str,default="GAT")
    parser.add_argument('--static_conv',type=str,default="GCN")
    parser.add_argument('--spec_filt', type=str, default='mask')
    parser.add_argument('--post_temporal', type=int, default=0)
    
    parser.add_argument('--temporature', type=float, default=1, help='')
    parser.add_argument('--intervene_lambda', type=float, default=0, help='')
    parser.add_argument('--intervene_times', type=int, default=100, help='')
    parser.add_argument('--learns', type=int, default=0)
    
    # 5. nodeclf
    parser.add_argument('--clf_layers', type=int, default=2, help='')
    parser.add_argument('--model', type=str, help='')
    parser.add_argument('--metric', type=str, default='acc')
    parser.add_argument('--main_metric', type=str, default='val_acc')
    
    # 6. dida
    parser.add_argument('--only_causal', type=int, default=0)
    parser.add_argument('--fmask', type=int, default=1)
    parser.add_argument('--lin_bias', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--skip', type=int, default=0)
    
    
    
    
    
    
    
    
    args = parser.parse_args()
    
    # set the running device
    if int(args.device_id) >= 0 and torch.cuda.is_available():
        args.device = torch.device("cuda:{}".format(args.device_id))
        print('using gpu:{} to train the model'.format(args.device_id))
    else:
        args.device = torch.device("cpu")
        print('using cpu to train the model')

    def setargs(args,hp):
        for k,v in hp.items():
            setattr(args,k,v)
            

    # if args.use_cfg:
    #     if args.dataset == 'collab':
    #         hp={"n_layers": 2, "heads": 4, "norm": 1, "skip": 0, "dropout": 0.0, "use_RTE": 1, "fmask": 1, "lin_bias": 0}
    #         setargs(args,hp)
    #     elif args.dataset == 'yelp':
    #         hp={"n_layers": 2, "heads": 4, "norm": 1, "skip": 0, "dropout": 0.0, "use_RTE": 1,  "fmask": 1, "lin_bias": 0}
    #         setargs(args,hp)
    #     elif 'synthetic' in args.dataset:
    #         hp={"n_layers": 2, "heads": 2, "norm": 1, "skip": 1, "dropout": 0.0, "use_RTE": 1,  "fmask": 1, "lin_bias": 0}
    #         setargs(args,hp)
    #     else:
    #         raise NotImplementedError(f"dataset {args.dataset} not implemented")
        
    if args.log_dir == 'EXP':
        CURDIR = os.path.dirname(__file__)
        args.log_dir = os.path.abspath(os.path.join(CURDIR,f'../../logs/tmp/'))
    
    os.makedirs(args.log_dir,exist_ok=True)
    return args
