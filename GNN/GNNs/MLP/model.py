import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, features, hidden_channels, out_channels, num_layers, dropout):
        super(MLP, self).__init__()

        self.features = features

        self.convs = torch.nn.ModuleList()
        self.convs.append(nn.Linear(features.shape[1], hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, adj_t=None):
        x = self.features
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x)
        
        return x
    
    def get_features(self, adj_t=None, layer=0):
        x = self.features

        if layer == 0:
            return x

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x)
            if i == layer - 1:
                return x
            x = self.bns[i](x)
            x = F.relu(x)
        x = self.convs[-1](x)

        return x