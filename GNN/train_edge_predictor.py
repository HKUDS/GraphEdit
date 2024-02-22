import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split

import argparse
import numpy as np

from utils import time_logger

class EdgePredictor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(EdgePredictor, self).__init__()
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x

def preprocess(args):
    data = torch.load('./datasets/' + args.dataset + '/' + args.dataset + '_fixed_tfidf.pt')
    data.x = np.load('./datasets/' + args.dataset +  '/' + args.dataset + '_embs.npy')
    data.x = torch.from_numpy(data.x).float()

    train_edges = np.load('./datasets/' + args.dataset +  '/' + args.dataset + '_edges_train.npy')

    data_x = []
    data_y = []

    for i in range(len(train_edges[0])):
        paperID_0 = train_edges[0][i]
        paperID_1 = train_edges[1][i]
    
        data_x.append(torch.cat([data.x[paperID_0], data.x[paperID_1]]))
        data_y.append(int(data.y[paperID_0] == data.y[paperID_1]))

    data_x = torch.stack(data_x)
    data_y = torch.tensor(data_y, dtype=torch.float32)

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=args.seed)

    return train_x, train_y, test_x, test_y

@time_logger
def train(args):
    device = args.device
    model = EdgePredictor(args.embedding_dim, args.hidden_dim).to(device)

    train_x, train_y, test_x, test_y = preprocess(args)

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_accuracy = 0
    epochs_no_improve = 0

    num_epochs = args.epochs
    for epoch in range(num_epochs):
        for i, (node_embedding_pairs, batch_labels) in enumerate(train_loader):
            node_embedding_pairs, batch_labels = node_embedding_pairs.to(device), batch_labels.to(device)

            outputs = model(node_embedding_pairs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for node_embedding_pairs, batch_labels in test_loader:
                node_embedding_pairs, batch_labels = node_embedding_pairs.to(device), batch_labels.to(device)
                outputs = model(node_embedding_pairs)
                predicted = (outputs.squeeze() > 0.5).float()
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
            
        test_accuracy = 100 * correct / total

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item()}, Test Accuracy: {test_accuracy}%')

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            epochs_no_improve = 0
            torch.save(model.state_dict(), './datasets/' + args.dataset + '/' + args.dataset + '_edge_predictor.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.early_stop:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='pubmed')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embedding_dim', type=int, default=4096)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=10)
    
    args = parser.parse_args()

    train(args)