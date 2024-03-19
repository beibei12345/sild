import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,SAGEConv,GINConv,GATConv
import numpy as np
from sild.utils import DummyArgs
from torch.fft import rfft,irfft,fft,ifft
from DIDA.utils.loss import EnvLoss
from utils import *
from sild.utils import COLLECTOR
from dida import DGNN

class SpecMask(nn.Module):
    def __init__(self, hid_dim, temporature, K_len) -> None:
        super().__init__()
        self.K_len = K_len
        self.node_spec_map = nn.Sequential(nn.Linear(self.K_len * hid_dim * 2, hid_dim *2), nn.ReLU(), nn.Linear(hid_dim*2, self.K_len)) # N, K x d x 2 - > N x f 
        self.temporature = temporature
        self.K_len = K_len 
        self.hid_dim = hid_dim
        
    def forward(self, specs):
        # specs [N, T, d]
        # learn causal and spurious masks
        spec_real, spec_imag = specs.real, specs.imag # [N, K, d]
        spec_real_imag = torch.stack([spec_real,spec_imag],dim=-1) # [N, K, d, 2]
        node_choice = self.node_spec_map(spec_real_imag.view(-1, self.K_len * self.hid_dim *2)) # [N, K * d * 2] -> [N, K]
        
        cmask_ = torch.sigmoid( node_choice / self.temporature)
        smask_ = torch.sigmoid( - node_choice / self.temporature) # [N, K]
        if len(COLLECTOR.cache.get('cmask0',[])) == len(COLLECTOR.cache.get('loss',[])):
            COLLECTOR.add('cmask0', seq2str(cmask_[0].detach().cpu().numpy()))
            COLLECTOR.add('smask0', seq2str(smask_[0].detach().cpu().numpy()))
            
        cmask = cmask_.unsqueeze(-1).expand_as(spec_imag)
        smask = smask_.unsqueeze(-1).expand_as(spec_imag)
        
        # filter in the spectral domain 
        c_spec_real = spec_real * cmask # [N, K, d] * [N, K, d]
        c_spec_imag = spec_imag * cmask # [N, K, d] * [N, K, d]
        c_spec = torch.complex(c_spec_real, c_spec_imag) # [N, K, d]
        
        s_spec_real = spec_real * smask # [N, K, d] * [N, K, d]
        s_spec_imag = spec_imag * smask # [N, K, d] * [N, K, d]
        s_spec = torch.complex(s_spec_real, s_spec_imag) # [N, K, d]

        return c_spec, s_spec
    
class SpecAttn(nn.Module):
    def __init__(self, hid_dim, args) -> None:
        super().__init__()
        self.spec_attn = SpecAttentionLayer(hid_dim, 1, args.attn_drop, args.temp_attn_res, use_RTE = args.spec_RTE, temporature = args.temporature, norm = True)
        self.hid_dim = hid_dim
        
    def forward(self, specs):
        # specs [N, T, d]
        c_spec, s_spec = self.spec_attn(specs) # [N, T, d]
        return c_spec, s_spec

class SGCNNetIter(torch.nn.Module):
    def __init__(self, args, data):
        super(SGCNNetIter, self).__init__()
        self.args = args
        
        # hyperparams
        in_dim = args.nfeat
        hid_dim = 2 * args.nhid
        out_dim = 2 * args.nhid
        num_layers = args.n_layers
        time_length = args.length
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.window_size = args.window_size 
        self.temporature = args.temporature
        self.norm = args.norm
        
        if args.spec_len == -1:
            self.spec_len = args.length
        elif args.spec_len == -2:
            self.spec_len = args.window_size
        else:
            self.spec_len = args.spec_len
        self.K_len = 1 + self.spec_len//2

        # dataset
        self.time_length = time_length 
        self.len = args.length
        self.len_train = self.len - args.testlength - args.vallength
        self.len_val = args.vallength
        self.len_test = args.testlength
        x = data['x']
        self.x = [x for _ in range(self.len)] if len(x.shape) <= 2 else x
        print('total length: {}, test length: {}, window_size : {}'.format(self.len, args.testlength, self.window_size))
        
        # spatio model
        if args.backbone == 'dida':
            self.backbone = DGNN(args)
        elif args.backbone in 'GCN GIN GAT'.split():
            args.static_conv = args.backbone
            self.backbone = SpaSeqNet(args)
        else:
            raise NotImplementedError()
        
        # spectral model
        if args.spec_filt == "mask":
            self.spec_filt = SpecMask(hid_dim, args.temporature, self.K_len)
        elif args.spec_filt == 'attn':
            self.spec_filt = SpecAttn(hid_dim, args)
        else:
            raise NotImplementedError()
        
        # post temporal
        if args.post_temporal:
            self.temporal_attn = TemporalAttentionLayer(hid_dim, args.heads, args.attn_drop, args.temp_attn_res, use_RTE = args.use_RTE, only_last = True)
        
        # clf
        self.ctype = args.ctype
        self.ltype = args.ltype
        
        # measure and link prediction 
        self.cs_decoder = self.backbone.cs_decoder
        self.ss_decoder = self.backbone.ss_decoder
        self.measure = EnvLoss(args)
         
        # optim
        self.optim = torch.optim.Adam(self.parameters(), lr = args.lr, weight_decay = args.weight_decay)
        self.criterion = torch.nn.BCELoss()
    
    def cal_loss(self, zs, edge_list, epoch):
        cs, ss = zs # [T, N, F], [T, N, F] z starts from 0 to T
        args = self.args
        device = args.device
        intervene_times, la = args.intervene_times, args.intervene_lambda
        len_train = self.len_train
        
        edge_index, edge_label = get_edges_all(edge_list[1:], self.args) # [T-1, E]edges start from 1 to T for prediction
        
        def cal_y(embeddings, decoder):
            preds = torch.tensor([]).to(device)
            for t in range(len_train - 1): # [T-1]
                z = embeddings[t]
                pred = decoder(z, edge_index[t])
                preds = torch.cat([preds, pred])
            return preds

        criterion = torch.nn.BCELoss()

        def cal_loss(y, label):
            return criterion(y, label)

        cy = cal_y(cs, self.cs_decoder)
        sy = cal_y(ss, self.ss_decoder)
        causal_loss = cal_loss(cy, edge_label)
    
        if self.args.intervene_times <= 0 or self.args.intervene_lambda <= 0:
            loss = causal_loss
        else:            
            env_loss = torch.tensor([]).to(device)
            for i in range(intervene_times):
                s1 = np.random.randint(len(sy))
                s = torch.sigmoid(sy[s1]).detach()
                conf = s * cy
                env_loss = torch.cat([env_loss, cal_loss(conf, edge_label).unsqueeze(0)])
            env_mean = env_loss.mean()
            env_var = torch.var(env_loss * intervene_times)
            penalty = env_mean + env_var
            loss = causal_loss + la * penalty
            
            COLLECTOR.add('env_mean', env_mean.item())
            COLLECTOR.add('env_var', env_var.item())
            
        COLLECTOR.add('causal_loss', causal_loss.item())
        return loss
    

        
    def spectral_filter(self, z):
        # z [T, N, d]
        if not self.args.use_filt:
            return z[-1], z[-1]
        ctype = self.ctype
        time_len = z.shape[0]
        # transform into spectral domain 
        z = torch.permute(z,(1,0,2)) # [N, T, d]
        specs = rfft(z, n = self.spec_len, norm = "ortho", dim = 1) # [N, K, d]
        
        # learn causal and spurious masks
        c_spec, s_spec = self.spec_filt(specs) # [N, T, d]
        
        # transform back to time domain
        cout = irfft(c_spec, n = self.spec_len, norm = "ortho", dim = 1)[:, :time_len, :] # [N, T, d]
        if self.args.post_temporal:
            cout = self.temporal_attn(cout) # [N, T, d]
        cout = torch.permute(cout,(1,0,2)) # [T, N, d]
        
        sout = irfft(s_spec, n = self.spec_len, norm = "ortho", dim = 1)[:, :time_len, :] # [N, T, d]
        if self.args.post_temporal:
            sout = self.temporal_attn(sout) # [N, T, d]
        sout = torch.permute(sout,(1,0,2)) # [T, N, d]
        
        cout = cout[-1]
        sout = sout[-1]
        out = [cout, sout]
        return out
    
    def get_final_emb(self, edge_list):
        x_list, cs, ss = self.backbone(edge_list, self.x[:len(edge_list)])
        x_list = torch.stack(x_list, dim=0)
        time_len = len(edge_list)
        czs, szs = [], []
        for end in range(1, time_len):
            cz, sz = self.spectral_filter(x_list[:end]) # [T,N,F]
            czs.append(cz)
            szs.append(sz)
        czs = torch.stack(czs, dim = 0) # [T,N,F]        
        szs = torch.stack(szs, dim = 0) # [T,N,F]        
        return czs, szs
        
        
    def train_epoch(self, data, epoch):
        args, optimizer, model, window_size = self.args, self.optim, self, self.window_size

        model.train()
        edge_list = [data['edge_index_list'][ix] for ix in range(self.len_train)]
        cs, ss = self.get_final_emb(edge_list)
        
        loss = self.cal_loss([cs,ss], edge_list, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def test_epoch(self, data, type = 'train'):
        args, model, window_size = self.args, self, self.window_size
        train_auc_list, val_auc_list, test_auc_list = [], [], []
        
        model.eval()
        edge_list = [data['edge_index_list'][ix] for ix in range(self.len)]
        cs, ss = self.get_final_emb(edge_list)
        
        for t in range(self.len - 1):
            z = cs[t]
            edge_index, pos_edge, neg_edge = prepare(data, t + 1)[:3]
            if is_empty_edges(neg_edge):
                continue
            auc, ap = self.measure.predict(z, pos_edge, neg_edge, self.cs_decoder)
            if t < self.len_train - 1:
                train_auc_list.append(auc)
            elif t < self.len_train + self.len_val - 1:
                val_auc_list.append(auc)
            else:
                test_auc_list.append(auc)
            
        train, val, test = np.mean(train_auc_list), np.mean(val_auc_list), np.mean(test_auc_list)
        COLLECTOR.add(key = f'{type}_train_auc_list', value = train_auc_list)
        COLLECTOR.add(key = f'{type}_val_auc_list', value = val_auc_list)
        COLLECTOR.add(key = f'{type}_test_auc_list', value = test_auc_list)
        
        return train, val, test