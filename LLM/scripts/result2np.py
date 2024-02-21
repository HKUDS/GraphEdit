import torch
import numpy as np
import json

data_name = 'pubmed_template_add_3'

with open('../GNN/datasets/pubmed/' + data_name + '.json') as f:
    temp = json.load(f)

with open('./result/' + data_name + '.json', 'r') as file:
    result_data = json.load(file)

ids = [item['id'] for item in temp]
result_list = [item['res'] for item in result_data]


data = torch.load("../GNN/datasets/pubmed/pubmed_fixed_tfidf.pt")
data.edge_index = np.load('../GNN/datasets/pubmed/' + data_name + '.npy')

data_edges = []
nums = 0

for index, id in enumerate(ids):
    if result_list[index] != "False":
        id = int(id)
        data_edges.append((data.edge_index[0][id], data.edge_index[1][id]))
        data_edges.append((data.edge_index[1][id], data.edge_index[0][id]))

data_edges = np.unique(data_edges, axis=0).transpose()
np.save('../../GNN/datasets/pubmed/pubmed_edges_add_3_delete', data_edges)