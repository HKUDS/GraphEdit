import numpy as np
# adapted from https://github.com/jcatw/scnn
import torch
import random
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.preprocessing import normalize
import json
import pandas as pd


# return pubmed dataset as pytorch geometric Data object together with 60/20/20 split, and list of pubmed IDs

def get_pubmed(edge_type=None):
    # load data
    data_name = 'pubmed'
    path = './datasets/pubmed/pubmed'

    dataset = Planetoid('datasets', data_name, 
                        transform=T.NormalizeFeatures())
    data = dataset[0]

    pubmed = torch.load(path + "_fixed_tfidf.pt")
    if edge_type:
        pubmed.edge_index = np.load(path + '_edges_' + edge_type + '.npy')

    data.x = torch.tensor(pubmed.x)
    data.edge_index = torch.tensor(pubmed.edge_index)
    data.y = torch.tensor(pubmed.y)
    data.num_features = dataset.num_features
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