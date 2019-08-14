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
from torch_geometric.nn import ChebConv

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.gcn1 = ChebConv(8,64,2)
        self.gcn2 = ChebConv(64,64,2)
        self.gcn3 = ChebConv(64,64,2)
        self.gcn4 = ChebConv(64,64,2)
        self.gcn5 = ChebConv(64,64,2)
        self.gcn6 = ChebConv(64,64,2)
        self.gcn7 = ChebConv(64,64,2)
        self.gcn8 = ChebConv(64,64,2)
      #  self.gcn9 = ChebConv(64,64,2)
      #  self.gcn10 = ChebConv(64,64,2)
      #  self.gcn11 = ChebConv(64,64,2)
      #  self.gcn12 = ChebConv(64,64,2)
      #  self.gcn13 = ChebConv(64,64,2)
      #  self.gcn14 = ChebConv(64,64,2)

        # output linear layer
        self.fc1 = nn.Linear(64,256)
        self.fc2 = nn.Linear(256,1)

    def forward(self, x, edges):
        y = F.relu(self.gcn1(x, edges))
        #print(y.size())
        y = F.relu(self.gcn2(y, edges))
        y = F.relu(self.gcn3(y, edges))
        y = F.relu(self.gcn4(y, edges))
        y = F.relu(self.gcn5(y, edges))
        y = F.relu(self.gcn6(y, edges))
        y = F.relu(self.gcn7(y, edges))
        y = F.relu(self.gcn8(y, edges))
    #    y = F.relu(self.gcn9(y, edges))
    #    y = F.relu(self.gcn10(y, edges))
    #    y = F.relu(self.gcn11(y, edges))
    #    y = F.relu(self.gcn12(y, edges))
    #    y = F.relu(self.gcn13(y, edges))
    #    y = F.relu(self.gcn14(y, edges))
        y = y[self.featuresAtAgentNode(x)]
        y = F.relu(self.fc1(y))
        #print(y.size())
        return self.fc2(y)


    def featuresAtAgentNode(self, features):
        agent_node = -1
        for i in range(len(features)):
            if features[i,3] == 1:
                agent_node = i

        return agent_node
    
    

    
