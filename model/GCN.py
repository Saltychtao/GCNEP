"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/MichSchli/RelationPrediction

Difference compared to MichSchli/RelationPrediction
* report raw metrics instead of filtered metrics

Adopted from DGL, https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from overrides import overrides
import dgl.function as fn


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=None, activation=None,
                 self_loop=False, dropout=0.0):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        if self.bias == True:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    # define how propagation is done in subclass
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g):
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        self.propagate(g)

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)

        g.ndata['h'] = node_repr


class RGCNTransLayer(RGCNLayer):
    @overrides
    def __init__(self,in_feat,out_feat,activation,bias=None,self_loop=False,dropout=0.0):
        super(RGCNTransLayer, self).__init__(in_feat,out_feat,bias,activation=None,self_loop=self_loop,dropout=dropout)
        self.linear = nn.Linear(in_feat,out_feat)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv,stdv)

    def msg_func(self,edges):
        return {'msg':edges.src['h'] * edges.src['norm']}

    def propagate(self,g):
        g.update_all(self.msg_func,fn.sum(msg='msg',out='h'),self.apply_func)

    def apply_func(self,nodes):
        if self.activation is not None:
            return {'h': self.activation(self.linear(self.dropout(nodes.data['h'])))}
        else:
            return {'h': self.linear(nodes.data['h'])}


class BaseRGCN(nn.Module):
    def __init__(self, g,num_nodes, h_dim, out_dim,
                 pretrained=None,num_hidden_layers=1, dropout=0, use_cuda=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.pretrained = pretrained
        self.g = g

        # create rgcn layers
        self.build_model()

        # create initial features
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        return None

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self):
        if self.features is not None:
            self.g.ndata['id'] = self.features
        for layer in self.layers:
            layer(self.g)
        return self.g.ndata.pop('h')


class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim,pretrained=None):
        super(EmbeddingLayer, self).__init__()
        if pretrained is None:
            self.embedding = torch.nn.Embedding(num_nodes, h_dim,padding_idx=0)
        else:
            self.embedding = torch.nn.Embedding.from_pretrained(pretrained,freeze=False)

    def forward(self, g):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = self.embedding(node_id)


class RGCN(BaseRGCN):
    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.h_dim,pretrained=self.pretrained)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RGCNTransLayer(in_feat=self.h_dim,out_feat=self.h_dim,activation=act,self_loop=False,dropout=self.dropout)
