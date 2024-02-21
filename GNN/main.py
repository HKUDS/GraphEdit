import argparse
import pandas as pd

from GNNs.gnn_trainer import GNNTrainer

from utils import set_random_seed

from datasets.load import load_data


def run(args):  
    seeds = args.seed if args.seed is not None else range(args.runs)
    all_acc = []

    for seed in seeds:
        set_random_seed(seed)
        data= load_data(args.dataset, args.edge_type)

        trainer = GNNTrainer(args, data)
        trainer.train()
        pred, acc = trainer.eval_and_save()
        all_acc.append(acc)

        
    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        print(f"[{args.model_name}] ValACC: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f}, TestAcc: {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pubmed')
    parser.add_argument('--edge_type', type=str, default='add_3_delete')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--model_name', type=str, default='GCN')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--early_stop', type=int, default=50)
    
    args = parser.parse_args()

    run(args)