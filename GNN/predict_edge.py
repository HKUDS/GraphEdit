import torch
import numpy as np
import argparse
import os

from tqdm import trange, tqdm
from train_edge_predictor import EdgePredictor

def update_edge_scores(edge_scores, batch_scores, batch_pairs):
    for idx, score in enumerate(batch_scores):
        i, j = batch_pairs[idx]
        edge_scores[i, j] = score
        edge_scores[j, i] = score

def predict_batch(model, node_embeddings, batch_pairs, device):
    batch_tensors = [torch.cat([node_embeddings[i], node_embeddings[j]]) for i, j in batch_pairs]
    batch_pairs_tensor = torch.stack(batch_tensors).to(device)
    with torch.no_grad():
        batch_scores = model(batch_pairs_tensor).squeeze()
    if batch_scores.ndim == 0:
        batch_scores = batch_scores.unsqueeze(0)
    return batch_scores

def predict_edges(args):
    node_embeddings = np.load('./datasets/' + args.dataset + '/' + args.dataset + '_embs.npy')
    node_embeddings = torch.from_numpy(node_embeddings).float()
    n = node_embeddings.size(0)
    edge_scores = torch.zeros(n, n)

    if os.path.exists('./datasets/' + args.dataset + '/' + args.dataset + '_edge_scores.npy'):
        edge_scores = np.load('./datasets/' + args.dataset + '/' + args.dataset + '_edge_scores.npy')
        edge_scores = torch.from_numpy(edge_scores).float()

        top_k_edges = []
        for i in range(n):
            scores = edge_scores[i]
            scores[i] = -float('inf')
            top_k_indices = torch.topk(scores, args.top_k).indices

            for idx in top_k_indices:
                top_k_edges.append([i, idx])
                top_k_edges.append([idx, i])

        top_k_edges = np.unique(top_k_edges, axis=0).transpose()

        np.save('./datasets/' + args.dataset + '/' + args.dataset + '_edges_recon_' + str(args.top_k) + '.npy', top_k_edges)

        if args.combine:
            ori_edges = torch.load("./datasets/pubmed/pubmed_fixed_tfidf.pt")
            np.save('./datasets/pubmed/pubmed_edges_add_' + str(args.top_k) + '.npy', np.hstack((ori_edges.edge_index, top_k_edges)))

        return top_k_edges

    model = EdgePredictor(args.embedding_dim, args.hidden_dim)
    model.load_state_dict(torch.load('./datasets/' + args.dataset + '/' + args.dataset + '_edge_predictor.pth'))
    model.to(args.device)

    model.eval()

    for i in trange(n):
        batch_pairs = []
        for j in range(i + 1, n):
            batch_pairs.append((i, j))
            if len(batch_pairs) == args.batch_size:
                batch_scores = predict_batch(model, node_embeddings, batch_pairs, args.device)
                update_edge_scores(edge_scores, batch_scores, batch_pairs)
                batch_pairs = []

        if batch_pairs:
            batch_scores = predict_batch(model, node_embeddings, batch_pairs, args.device)
            update_edge_scores(edge_scores, batch_scores, batch_pairs)

    top_k_edges = []
    for i in range(n):
        scores = edge_scores[i]
        scores[i] = -float('inf')
        top_k_indices = torch.topk(scores, args.top_k).indices

        for idx in top_k_indices:
            top_k_edges.append([i, idx])
            top_k_edges.append([idx, i])

    top_k_edges = np.unique(top_k_edges, axis=0).transpose()

    np.save('./datasets/' + args.dataset + '/' + args.dataset + '_edge_scores.npy', edge_scores)
    np.save('./datasets/' + args.dataset + '/' + args.dataset + '_edges_recon_' + str(args.top_k) + '.npy', top_k_edges)
    if args.combine:
        ori_edges = torch.load("./datasets/pubmed/pubmed_fixed_tfidf.pt")
        np.save('./datasets/pubmed/pubmed_edges_add_' + str(args.top_k) + '.npy', np.hstack((ori_edges.edge_index, top_k_edges)))

    return top_k_edges


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='pubmed')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--embedding_dim', type=int, default=4096)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--combine', type=bool, default=False)
    
    args = parser.parse_args()

    predict_edges(args)
