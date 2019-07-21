"""GCN using builtin functions that enables SPMV optimization.
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
""" Changed to use variable graphs """
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class GCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayer, self).__init__()
        
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.reset_parameters()
        self.g = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, h, g):
        self.g = g
        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h, self.weight)
        # normalization by square root of src degree
        h = h * self.g.ndata['norm']
        self.g.ndata['h'] = h
        self.g.update_all(fn.copy_src(src='h', out='m'),
                          fn.sum(msg='m', out='h'))
        h = self.g.ndata.pop('h')
        # normalization by square root of dst degree
        h = h * self.g.ndata['norm']
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(in_feats, n_hidden, activation, 0.))
        # hidden layers
        for i in range(n_layers - 1):
            if ((i % 3 == 0) and (i > 2)):
                 self.layers.append(GCNLayer(n_hidden, n_hidden, None, dropout))
            else:
                 self.layers.append(GCNLayer(n_hidden, n_hidden, activation, dropout))
        # output layer
        self.layers.append(GCNLayer(n_hidden, n_hidden, activation, 0.))
        # output linear layer
        self.fc1 = nn.Linear(n_hidden,1)
        #self.fc2 = nn.Linear(256,1)

    def forward(self, features, g):
        h = features
        for layer in self.layers:
            h = layer(h,g)
        #print("This is the output size {}".format(h.size()))
        h = self.featuresAtAgentNode(h,g)
        #print("This is the output after size {}".format(h.size()))
        h = F.relu(self.fc1(h))
        #h = F.relu(self.fc2(h))
        #print("This is the final output size {}".format(h.size()))
        return h

    def featuresAtAgentNode(self, features,g):
        agent_node = -1
        for i in range(g.number_of_nodes()):
            #if g.nodes[i].data['feat'][2] == 1:
            if g.ndata['feat'][i,2] == 1:
                agent_node = i

        return features[agent_node]

