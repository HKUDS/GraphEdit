import torch
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv


class SAGE(torch.nn.Module):
    def __init__(self, features, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()

        self.features = features

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(features.shape[1], hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, adj_t):
        x = self.features
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x
    
    def get_features(self, adj_t, layer=0):
        x = self.features

        if layer == 0:
            return x

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if i == layer - 1:
                return x
            x = self.bns[i](x)
            x = F.relu(x)
        x = self.convs[-1](x, adj_t)

        return x
