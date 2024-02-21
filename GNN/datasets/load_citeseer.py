import numpy as np
import torch

from torch import nn 

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


# return citeseer dataset as pytorch geometric Data object together with 60/20/20 split, and list of citeseer IDs


def get_citeseer(edge_type=None):
    # load data
    data_name = 'citeseer'
    path = './datasets/citeseer/citeseer'

    dataset = Planetoid('datasets', data_name,
                        transform=T.NormalizeFeatures())
    data = dataset[0]

    citeseer = torch.load(path + "_fixed_tfidf.pt")
    if edge_type:
        citeseer.edge_index = np.load(path + '_edges_' + edge_type + '.npy')

    data.x = torch.tensor(citeseer.x, dtype=torch.float32)
    data.edge_index = torch.tensor(citeseer.edge_index).long()
    data.y = citeseer.y
    data.num_nodes = len(citeseer.y)
    data.num_features = data.x.shape[1]
    data.num_classes = dataset.num_classes

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])

    return data
