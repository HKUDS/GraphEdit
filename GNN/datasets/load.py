from datasets.dataset import CustomDGLDataset

def load_data(dataset, edge_type=None):
    print(f'Dataset {dataset} has load')
    dataset = dataset.lower()
    if dataset == 'cora':
        from datasets.load_cora import get_cora as get_dataset
    elif dataset == 'citeseer':
        from datasets.load_citeseer import get_citeseer as get_dataset
    elif dataset == 'pubmed':
        from datasets.load_pubmed import get_pubmed as get_dataset
    else:
        exit(f'Error: Dataset {dataset} not supported')

    data = get_dataset(edge_type)

    return data
