import os
from sild.utils import *
import pandas as pd
from libwon import ParallelerGrid

grid_list=[
    {
        'dataset':['collab'],         
        'backbone':['GAT'],
        'spec_filt':['mask'],
        'post_temporal':[0],
        'intervene_lambda':[1e-4],
        'intervene_times':[1000],
        'min_epoch':[0],
        'heads':[2],
        'norm':[2],
        'temporature':[0.5],
        'dropout':[0.4],
        'seed': range(3)
    },
    {
        'dataset':['yelp'],         
        'backbone':['dida'],
        'spec_filt':['mask'],
        'post_temporal':[0],
        'intervene_lambda':[1e-2],
        'intervene_times':[1000],
        'min_epoch':[200],
        'heads':[4],
        'norm':[1],
        'temporature':[1],
        'dropout':[0],
        'seed': range(3)
    },
    {
        'dataset':['synthetic-0.4'],         
        'backbone':['GAT'],
        'spec_filt':['mask'],
        'post_temporal':[0],
        'intervene_lambda':[1e-2],
        'intervene_times':[100],
        'min_epoch':[0],
        'heads':[1],
        'norm':[1],
        'temporature':[0.5],
        'dropout':[0.4],
        'seed': range(3)
    },
    {
        'dataset':['synthetic-0.6'],         
        'backbone':['GAT'],
        'spec_filt':['mask'],
        'post_temporal':[0],
        'intervene_lambda':[1e-2],
        'intervene_times':[100],
        'min_epoch':[0],
        'heads':[1],
        'norm':[1],
        'temporature':[0.5],
        'dropout':[0.4],
        'seed': range(3)
    },
    {
        'dataset':['synthetic-0.8'],         
        'backbone':['GAT'],
        'spec_filt':['mask'],
        'post_temporal':[0],
        'intervene_lambda':[1e-2],
        'intervene_times':[1000],
        'min_epoch':[0],
        'heads':[4],
        'norm':[1],
        'temporature':[0.5],
        'dropout':[0.4],
        'seed': range(3)
    },
]

# gpus = [0,3,5]
gpus = [0,1,2,3,4,5]

cmd = 'python main.py'
readme = ''
phone_notice = ""
base_dir = "../../sild/" 

class Parallel(ParallelerGrid):
    def show(self,c):
        exp_dir=self.exp_dir
        ana_dir=self.ana_dir
        cols = ParallelerGrid.collect_keys(grid_list)
        print(cols)
        import json
        import datetime
        from libwon.utils.collector import COLLECTOR
        files=os.listdir(exp_dir)
        res={}
        for fname in files:
            try:
                folder=os.path.join(exp_dir,fname)
                info=json.load(open(os.path.join(folder,'args.json')))
                hp=tuple([info[k] for k in cols])
            
                COLLECTOR.load(os.path.join(exp_dir,fname,'collector.json'))
                cache=COLLECTOR.cache
                res[hp]=cache   
            except Exception as e:
                print(e)
        
        if c==0:
            table=[]
            metrics = 'best_test_test_auc'.split()
            for k,v in res.items():
                table.append([*k]+[v[x][0] for x in metrics])
            df = pd.DataFrame(table,columns=cols+metrics)
            df = df.sort_values(by="best_test_test_auc".split())
            g = cols.copy()
            del g[g.index('seed')]
            agg_dict = {f"mean_{x}":(x,"mean") for x in metrics}
            agg_dict.update({f"std_{x}":(x,"std") for x in metrics})
            
            df = df.drop(columns='seed'.split()).groupby(g).agg(**agg_dict).reset_index()
            def format(x):
                x = x * 100 if x<=1 else x
                return f"{x:.2f}"
            for m in metrics:
                df[f"{m}"] = df[f"mean_{m}"].apply(format) +"+-" + df[f"std_{m}"].apply(format)
            df = df.drop(columns = [f"mean_{m}" for m in metrics] + [f"std_{m}" for m in metrics])
            # df = df.sort_values(by = "dataset best_test_test_auc".split())
            df = df['dataset best_test_test_auc'.split()]
            print(df) 

            df.to_csv(os.path.join(self.ana_dir,f"{os.path.basename(__file__)}-{c}.csv"))
            
            

gpu_arg, log_arg = "device_id", "log_dir"
dir,fname=os.path.abspath(__file__).split(os.sep)[-2:]
log_dir=os.path.join('../../logs/',dir,os.path.splitext(fname)[0])     
parallel=Parallel(
    gpus=gpus,
    log_dir=log_dir,
    grid_list=grid_list,
    cmd=cmd,
    gpu_arg=gpu_arg,
    log_arg=log_arg,
    epoch_arg="max_epoch",
    exp_script_dir=os.path.dirname(os.path.abspath(__file__)),
    base_script_dir=base_dir,
    readme=readme,
    phone_notice=phone_notice
).execute()