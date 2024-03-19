import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,SAGEConv,GINConv,GATConv
import numpy as np
from torch_geometric.utils import add_self_loops
class MultiplyPredictor(torch.nn.Module):
    def __init__(self):
        super(MultiplyPredictor, self).__init__()
    def forward(self, z, e):
        x_i = z[e[0]]
        x_j = z[e[1]]
        x = (x_i * x_j).sum(dim=1)
        return torch.sigmoid(x)
    
class StaticNet(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_dim = args.nfeat
        self.hidden_dim = 2 * args.nhid
        self.output_dim = 2 * args.nhid
        self.n_layers = args.n_layers
        self.n_heads = args.heads
        self.conv_type = args.model
        self.norm = args.norm
        self.convs = nn.ModuleList()
        
        if self.norm == 0:
            self.update_norm = lambda x : x
        elif self.norm == 1:
            self.update_norm = nn.LayerNorm(self.hidden_dim) 
        elif self.norm == 2:
            self.update_norm = nn.BatchNorm1d(self.hidden_dim)
        # self.fmask = nn.Parameter(torch.ones(self.hidden_dim), requires_grad=True)
        # self.linear = nn.Linear(self.input_dim , self.hidden_dim, bias = False)
        
        def create_conv(in_dim, out_dim):
            if self.conv_type == "GCN":
                return GCNConv(in_dim, out_dim)
            elif self.conv_type == "GIN":
                return GINConv(nn = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))
            elif self.conv_type == "GAT":
                return GATConv(in_dim, out_dim, self.n_heads, dropout = 0., concat = False)
                
        for i in range(self.n_layers):
            if i == 0:
                self.convs.append(create_conv(self.input_dim, self.hidden_dim))
            else:
                self.convs.append(create_conv(self.hidden_dim, self.hidden_dim))

    def forward(self, edge_index, x):
        # x = self.linear(x)
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            if i!= self.n_layers-1:
                if self.norm:
                    x = self.update_norm(x)
                x = F.relu(x)
        return x

class NodeClf(nn.Module):
    def __init__(self,args, isrnn = False) -> None:
        super().__init__()
        clf = nn.ModuleList()
        hid_dim = args.nhid if isrnn else args.nhid*2
        clf_layers = args.clf_layers
        num_classes = args.num_classes
        for i in range(clf_layers-1):
            clf.append(nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU()))
        clf.append(nn.Linear(hid_dim, num_classes))
        self.clf = clf
    def forward(self, x):
        for layer in self.clf:
            x = layer(x)
        return x
    
from sild.utils import DummyArgs
class SpaSeqNet(torch.nn.Module):
    def __init__(self, args):
        super(SpaSeqNet, self).__init__()
        self.static = StaticNet(DummyArgs(nfeat = args.nfeat, nhid = args.nhid, n_layers = args.n_layers, heads = args.heads, model = args.static_conv, norm = args.norm))
        hid_dim = args.nhid * 2
        self.cs_decoder = NodeClf(args, isrnn = False)
        self.ss_decoder = NodeClf(args, isrnn = False)
        
    def forward(self, edge_index_list, x_list):
        time_len = len(edge_index_list)
        x_list = [self.static(edge_index_list[i], x_list[i]) for i in range(time_len)]
        return x_list, None, None

from torch_geometric.utils import negative_sampling
from DIDA.utils.mutils import bi_negative_sampling
def seq2str(xs):
    xs = [round(float(x),4) for x in xs]
    return xs
def get_edges(pos_edge_index, args):
    device = args.device
    if args.dataset == 'yelp':
        neg_edge_index = bi_negative_sampling(pos_edge_index,
                                                args.num_nodes,
                                                args.shift,
                                                num_neg_samples=int(pos_edge_index.size(1) * args.sampling_times)
                                                )
    else:
        neg_edge_index = negative_sampling(
            pos_edge_index,
            num_neg_samples=int(pos_edge_index.size(1) *args.sampling_times) )
    edges = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    pos_y = torch.ones(pos_edge_index.size(1)).to(device)
    neg_y = torch.zeros(neg_edge_index.size(1)).to(device)
    edge_label = torch.cat([pos_y, neg_y], dim=0)
    return edges, edge_label

def get_edges_all(edge_list, args):
    edges_all = []
    edge_label_all = []
    for e in edge_list:
        edges, edge_label = get_edges(e, args)
        edges_all.append(edges)
        edge_label_all.append(edge_label)
    edge_label_all = torch.cat(edge_label_all, dim = 0)
    return edges_all, edge_label_all


import math
class RelTemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''

    def __init__(self, n_hid, max_len=50, dropout=0.2):  # original max_len=240
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x, T):
        t = torch.arange(T).to(x.device)
        enc = self.lin(self.emb(t)) # [T, d]
        enc = enc.expand_as(x) # [N,T,d]
        return x + enc
    
class PositionalEncoding(nn.Module):
    def __init__(self, n_hid, max_len=100, dropout=0.2):  # original max_len=240
        super(PositionalEncoding, self).__init__()
        self.emb = nn.Embedding(max_len, n_hid)
        
    def forward(self, x, T):
        t = torch.arange(T).to(x.device)
        enc = self.emb(t) # [T, d]
        enc = enc.expand_as(x) # [N,T,d]
        return x + enc
        

class SpecAttentionLayer(nn.Module):
    def __init__(self, 
                input_dim, 
                n_heads, 
                attn_drop, 
                residual,
                use_RTE = True,
                temporature = 1,
                norm = True,
                spec_res = True
                ):
        super(SpecAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.residual = residual
        self.temporature = temporature
        self.norm = norm
        self.input_dim = input_dim
        self.spec_res = spec_res

        hid_dim = input_dim*2
        self.update_norm = nn.LayerNorm(hid_dim)
        self.ffn_mlp = nn.Sequential(nn.Linear(hid_dim, 2*hid_dim), nn.GELU(), nn.Linear(2*hid_dim, hid_dim))

        # temporal 
        self.tenc = PositionalEncoding(hid_dim)
        self.use_RTE = use_RTE
        
        # define weights
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(hid_dim, hid_dim))

        self.xavier_init()

    def forward(self, inputs):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        input_dim = self.input_dim
        time_length = inputs.shape[1]
        
        temporal_inputs = torch.cat([inputs.real, inputs.imag],dim=-1)  # [N, T, 2*F]
        if self.use_RTE:
            temporal_inputs = self.tenc(temporal_inputs, time_length) # todo, 
        # 2: Query, Key based multi-head self attention.
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2],[0])) # [N, T, F]
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2],[0])) # [N, T, F]
        v = temporal_inputs # [N, T, F] does not transform v
        
        outputs = torch.matmul(q, k.permute(0,2,1)) # [N, T, T]
        outputs = outputs / (time_length ** 0.5)
        
        cau_att = F.softmax(outputs/self.temporature, dim=2) # [hN, T, T]
        spu_att = F.softmax(-outputs/self.temporature, dim=2) # [hN, T, T]
        
        cau_out = torch.matmul(cau_att, v)  # [hN, T, F/h]
        spu_out = torch.matmul(spu_att, v)  # [hN, T, F/h]
        
        if self.spec_res:
            cau_out = cau_out + v
        
        cau_out = torch.complex(cau_out[...,:input_dim], cau_out[...,input_dim:])
        spu_out = torch.complex(spu_out[...,:input_dim], spu_out[...,input_dim:])
        
        return cau_out, spu_out

    def xavier_init(self):
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)

class TemporalAttentionLayer(nn.Module):
    def __init__(self, 
                input_dim, 
                n_heads, 
                attn_drop, 
                residual,
                use_RTE = True,
                only_last = True
                ):
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.residual = residual

        # temporal 
        self.tenc = RelTemporalEncoding(input_dim)
        self.use_RTE = use_RTE
        self.only_last = only_last
        # define weights
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # ff
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # dropout 
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()

    def forward(self, inputs):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        temporal_inputs = inputs  # [N, T, F]
        time_length = inputs.shape[1]
        if self.use_RTE:
            temporal_inputs = self.tenc(temporal_inputs, time_length) 
        # 2: Query, Key based multi-head self attention.
        q = torch.tensordot(temporal_inputs if not self.only_last else temporal_inputs[:, [-1], :], self.Q_embedding_weights, dims=([2],[0])) # [N, T, F]
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2],[0])) # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2],[0])) # [N, T, F]
            
        # 3: Split, concat and scale.
        split_size = int(q.shape[-1]/self.n_heads)
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        
        outputs = torch.matmul(q_, k_.permute(0,2,1)) # [hN, T, T]
        outputs = outputs / (time_length ** 0.5)
        outputs = F.softmax(outputs, dim=2)

        # 5: Dropout on attention weights.
        if self.training:
            outputs = self.attn_dp(outputs)
        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0]/self.n_heads), dim=0), dim=2) # [N, T, F]
        
        # 6: Feedforward and residual
        outputs = self.feedforward(outputs)
        if self.residual:
            outputs = outputs + temporal_inputs if not self.only_last else outputs + temporal_inputs[:, [-1], :]
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)
        
def prepare(data, t):
    edge_index = data['edge_index_list'][t]
    pos_index = data['pedges'][t]
    neg_index = data['nedges'][t]
    return edge_index, pos_index, neg_index

def is_empty_edges(edges):
    return edges.shape[1]==0

from sild.utils import setup_seed
from sild.utils import EarlyStopping, move_to, cal_metric
from sild.utils import COLLECTOR
from tqdm import tqdm,trange

class Trainer:
    def __init__(self, model, data, args):
        self.model = model 
        self.earlystop = EarlyStopping('val_auc', mode="max", patience = args.patience)
        self.device = args.device
        self.metric = args.metric
        self.args = args
        self.data = data
        
    def train(self, data, *args, **kwargs):
        return self.model.train_epoch(data, *args, **kwargs)

    @torch.no_grad()
    def test(self, data, *args, **kwargs):
        return self.model.test_epoch(data,*args,**kwargs)
        
    
    def train_till_end(self, disable_progress = False):
        earlystop = self.earlystop
        with tqdm(range(self.args.max_epoch), disable=disable_progress) as bar:
            for epoch in bar:
                loss = self.train(self.data)
                train_auc,val_auc,test_auc = self.test(self.data)
                metrics = dict(zip('epoch,loss,train_auc,val_auc,test_auc'.split(','),[epoch,loss,train_auc,val_auc,test_auc]))
                bar_metrics = {m: metrics[m] for m in 'train_auc val_auc test_auc'.split()} 
                bar.set_postfix(**bar_metrics)
                for k,v in metrics.items():
                    COLLECTOR.add(key=f'{k}',value=v)
                if earlystop.step(**metrics):
                    break 
        best_metrics = earlystop.best_metrics
        for k,v in best_metrics.items():
            COLLECTOR.add(key=f'best_{k}',value=v)
        return best_metrics
    
    
    