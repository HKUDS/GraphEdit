import torch
from time import time

from GNNs.gnn_utils import Evaluator, EarlyStopping
from datasets.load import load_data

from utils import time_logger

LOG_FREQ = 10

class GNNTrainer():

    def __init__(self, args, data):
        self.seed = args.seed
        self.device = args.device
        self.dataset_name = args.dataset
        self.gnn_model_name = args.model_name
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.lr = args.lr
        self.epochs = args.epochs

        self.data = data.to(self.device)

        self.features = self.data.x
        self.data.y = self.data.y.squeeze()

        try:
            if self.gnn_model_name == "GCN":
                from GNNs.GCN.model import GCN as GNN
            elif self.gnn_model_name == "SAGE":
                from GNNs.SAGE.model import SAGE as GNN
            elif self.gnn_model_name == "RevGAT":
                from GNNs.RevGAT.model import RevGAT as GNN
            elif self.gnn_model_name == "MLP":
                from GNNs.MLP.model import MLP as GNN

        except NameError:
            print(f"Model {self.gnn_model_name} is not supported!")

        self.model = GNN(features=self.features,
                         hidden_channels=self.hidden_dim,
                         out_channels=self.data.num_classes,
                         num_layers=self.num_layers,
                         dropout=self.dropout).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=0.0)
        
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        print(f"\nNumber of parameters: {trainable_params}")
        self.ckpt = f"output/{self.dataset_name}/{self.gnn_model_name}_{self.seed}.mod"

        self.stopper = EarlyStopping(
            patience=args.early_stop, path=self.ckpt) if args.early_stop > 0 else None
        self.loss_func = torch.nn.CrossEntropyLoss()

        self._evaluator = Evaluator(name=self.dataset_name)
        self.evaluator = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True),
             "y_true": labels.view(-1, 1)}
        )["acc"]
    
    @ torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        logits = self.model(self.data.edge_index)

        val_acc = self.evaluator(
            logits[self.data.val_mask], self.data.y[self.data.val_mask])
        test_acc = self.evaluator(
            logits[self.data.test_mask], self.data.y[self.data.test_mask])
        return val_acc, test_acc, logits
    
    @time_logger
    def train(self):
        # ! Training
        for epoch in range(self.epochs):
            t0, es_str = time(), ''
            self.model.train()
            
            logits = self.model(self.data.edge_index)

            loss = self.loss_func(
                logits[self.data.train_mask], self.data.y[self.data.train_mask])
            train_acc = self.evaluator(
                logits[self.data.train_mask], self.data.y[self.data.train_mask])
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            val_acc, test_acc, logits = self._evaluate()
        
            if self.stopper is not None:
                es_flag, es_str = self.stopper.step(val_acc, self.model, epoch)
                if es_flag:
                    print(f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break
            if epoch % LOG_FREQ == 0:
                print(f'Epoch: {epoch}, Time: {time()-t0:.4f}, Loss: {loss.item():.4f}, TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, ES: {es_str}')

        # ! Finished training, load checkpoints
        if self.stopper is not None:
            self.model = torch.load(self.stopper.path)

        return self.model
    
    @ torch.no_grad()
    def eval_and_save(self):
        torch.save(self.model, self.ckpt)
        val_acc, test_acc, logits = self._evaluate()

        print(f'[{self.gnn_model_name}] ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n')
        res = {'val_acc': val_acc, 'test_acc': test_acc}
        return logits, res