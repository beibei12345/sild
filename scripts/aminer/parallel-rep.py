import os
from sild.utils import *
import pandas as pd
from libwon import ParallelerGrid

grid_list=[
    {
        'dataset':['Aminer'], 
        
        # backbone params
        'backbone':['GAT'],
        'spec_filt':['mask'],
        
        # ood
        'intervene_lambda':[1e-3],
        'intervene_times':[100],
        'learns':[1],
        
        # simple params
        'heads':[2],
        'norm':[1],
        'use_RTE':[1],
        'spec_RTE':[1],
        'spec_res':[0],
        'temporature':[0.5],
        'dropout':[0],
        
        # train params
        'lr':[1e-2],
        'weight_decay':[5e-7],
        'nhid':[16],
        'min_epoch':[0],
        'patience':[50],
        
        'seed': range(3)
    },
]

gpus = [0,1,2,4,5]
# gpus = [4]

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
            for k,v in res.items():
                aucs = np.array(v['auc_list'])
                
                train = np.mean(aucs[:,:11],axis=-1)
                val = np.mean(aucs[:,11:14],axis=-1)
                test1 = np.mean(aucs[:,[14]],axis=-1)
                test2 = np.mean(aucs[:,[15]],axis=-1)
                test3 = np.mean(aucs[:,[16]],axis=-1)
                
                best_epoch = np.argmax(val)
                table.append([*k,train[best_epoch],val[best_epoch],test1[best_epoch],test2[best_epoch],test3[best_epoch]])
                
            df=pd.DataFrame(table,columns=cols+'train val test1 test2 test3'.split())
            def format(x):
                return f'{100*x:.2f}'
            df = df.drop(columns='seed')
            scols = list(set(cols) - set(['seed']))
            df = df.groupby(scols).agg(
                mean_train=('train','mean'),
                std_train=('train','std'),
                mean_val=('val','mean'),
                std_val=('val','std'),
                mean_test1=('test1','mean'),
                std_test1=('test1','std'),
                mean_test2=('test2','mean'),
                std_test2=('test2','std'),
                mean_test3=('test3','mean'),
                std_test3=('test3','std'),
                ).reset_index()
            
            for split in 'train val test1 test2 test3'.split():
                df[f"{split}"] = df[f"mean_{split}"].apply(format) +"+-" + df[f"std_{split}"].apply(format)
            df = df.drop(columns = [ f"mean_{split}" for split in 'train val test1 test2 test3'.split()]
                         +  [ f"std_{split}" for split in 'train val test1 test2 test3'.split()])
            df = df.sort_values(by='test1',ascending=True)
            df = df['test1 test2 test3'.split()]
            print(df.to_string())
            
            df = df.to_csv(os.path.join(self.ana_dir,f"{os.path.basename(__file__)}-{c}.csv"))
            

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