import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,SAGEConv,GINConv,GATConv
import numpy as np

from sild.utils import DummyArgs
from torch.fft import rfft,irfft,fft,ifft

from sklearn.metrics import f1_score,accuracy_score
from sild.utils import EarlyStopping, move_to, cal_metric
from sild.utils import COLLECTOR
from tqdm import tqdm,trange
from utils import *
class SpecMask(nn.Module):
    def __init__(self, hid_dim, temporature, K_len) -> None:
        super().__init__()
        self.K_len = K_len
        self.node_spec_map = nn.Sequential(nn.Linear(self.K_len * hid_dim * 2, hid_dim *2), nn.ReLU(), nn.Linear(hid_dim*2, self.K_len)) # N, K x d x 2 - > N x f 
        self.temporature = temporature
        self.K_len = K_len 
        self.hid_dim = hid_dim
        self.spec_lin = nn.Sequential(
            nn.Linear(K_len * hid_dim * 2, K_len), 
            nn.ReLU(),
            nn.Linear(K_len, hid_dim) 
            )
        
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
        
        s_spec_real = spec_real * smask # [N, K, d] * [N, K, d]
        s_spec_imag = spec_imag * smask # [N, K, d] * [N, K, d]

        c_spec = torch.cat([c_spec_real, c_spec_imag], dim =-1).flatten(-2,-1)
        s_spec = torch.cat([s_spec_real, s_spec_imag], dim =-1).flatten(-2,-1)
        
        c_spec = self.spec_lin(c_spec) # [N, d]
        s_spec = self.spec_lin(s_spec) # [N, d]
        
        return c_spec, s_spec
    
class SGCNNet(torch.nn.Module):
    def __init__(self, args, data):
        super(SGCNNet, self).__init__()
        in_dim = args.nfeat
        hid_dim = 2 * args.nhid
        out_dim = 2 * args.nhid
        num_layers = args.n_layers
        time_length = args.length
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.time_length = time_length 
        self.spec_len = time_length
        self.K_len = 1 + self.spec_len//2
        self.args = args
        
        # dataset
        x = data['x']
        self.x = [x for _ in range(self.time_length)] if len(x.shape) <= 2 else x
        
        # spatio model
        if args.backbone in 'GCN GIN GAT'.split():
            args.static_conv = args.backbone
            self.backbone = SpaSeqNet(args)
        else:
            raise NotImplementedError()
        
        # spectral model
        if args.spec_filt == "mask":
            self.spec_filt = SpecMask(hid_dim, args.temporature, self.K_len)
        else:
            raise NotImplementedError()

        # decoder 
        self.cs_decoder = self.backbone.cs_decoder
        self.ss_decoder = self.backbone.ss_decoder
        
        self.ctype = args.ctype
        
        # optim
        self.optim = torch.optim.Adam(self.parameters(),lr = args.lr,weight_decay = args.weight_decay)
    
    def cal_loss(self, zs, data):
        cs, ss = zs # [T, N, F], [T, N, F] z starts from 0 to T
        ss = ss.detach()
        args = self.args
        device = args.device
        intervention_times, la = args.intervene_times, args.intervene_lambda
        optimizer = self.optim
        
        def cal_y(embeddings, decoder, node_mask):
            pred = decoder(embeddings)[node_mask]
            return pred

        node_mask = data['train_mask']
        node_labels = data['y'][node_mask].squeeze().to(args.device)
            
        criterion = torch.nn.CrossEntropyLoss()
        cy = cal_y(cs, self.cs_decoder, node_mask) # [N,C]
        sy = cal_y(ss, self.ss_decoder, node_mask) # [N,C]

        conf_loss = criterion(sy, node_labels)
        COLLECTOR.add(key = 'conf_loss', value = conf_loss.item())
        causal_loss = criterion(cy, node_labels)
        COLLECTOR.add(key = 'causal_loss', value = causal_loss.item())

        def intervene(cy,sy,intervention_times,label):
            select = torch.randperm(len(sy))[:intervention_times].to(sy.device) # [I, 1]
            alls = torch.sigmoid(sy).detach()[select].unsqueeze(dim=1) # [I,1,C]
            allc = cy.expand(intervention_times,*cy.shape) # [I,N,C]
            conf = allc*alls # [I,N,C]
            I,N,C = conf.shape
            alle = label.expand(intervention_times,label.shape[0]) # [I,N]
            crit = torch.nn.CrossEntropyLoss(reduction='none') 
            env_loss = crit(conf.reshape(I*N,C),alle.reshape(I*N)) # [IN,C] and [IN] -> [IN]
            env_loss = env_loss.view(intervention_times,sy.shape[0]).mean(dim=-1) # [I,N] -> [I]
            env_mean = env_loss.mean() # [1]
            env_var = torch.var(env_loss*intervention_times)
            penalty = env_mean + env_var
            COLLECTOR.add(key = 'intervene_mean', value = env_mean.item())
            COLLECTOR.add(key = 'intervene_var', value = env_var.item())
            return penalty
          
        if intervention_times > 0 and la > 0:    
            penalty = 0
            penalty_intervene = intervene(cy,sy,intervention_times,node_labels)
            penalty += la * penalty_intervene
        else:
            penalty = torch.tensor(0).to(device)

        loss = causal_loss + penalty

        COLLECTOR.add(key = 'penalty', value = penalty.item())

        loss = loss + conf_loss
        
        return loss
    
    def spectral_filter(self, z):
        if not self.args.use_filt:
            return [z[-1],z[-1]]
        # z [T, N, d]
        ctype = self.ctype
        time_len = z.shape[0]
        # transform into spectral domain 
        z = torch.permute(z,(1,0,2)) # [N, T, d]
        specs = rfft(z, n = self.spec_len, norm = "ortho", dim = 1) # [N, K, d]
        
        # learn causal and spurious masks
        c_spec, s_spec = self.spec_filt(specs) # [N, T, d]
        
        out = [c_spec, s_spec]
        return out
    
    def get_final_emb(self, edge_list):
        x_list, cs, ss = self.backbone(edge_list, self.x[:len(edge_list)])
        x_list = torch.stack(x_list, dim=0) # [T,N,F]
        cz, sz = self.spectral_filter(x_list) # [N, d]
        return cz, sz
        
    def train_epoch(self, data):
        self.train()
        
        cz, sz = self.get_final_emb(data['edge_index'])
        loss = self.cal_loss([cz,sz], data)
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        return loss.item()
    
    def test_epoch(self, data):
        model = self
        model.eval()
        
        cz, sz = self.get_final_emb(data['edge_index'])
        pred = self.cs_decoder(cz) # [N, F]
        pred = pred.argmax(dim=-1).squeeze()
        target = data["y"].squeeze()
        
        auc_list = []
        for mode in "train val test".split(): 
            node_mask = data[f'{mode}_mask']
            p = pred[node_mask]
            t = target[node_mask]
            auc = cal_metric(p, t, self.args)
            auc_list.append(auc)
        train, val, test = auc_list
        COLLECTOR.add(key = 'auc_list', value = auc_list)
        return train, val, test
    


