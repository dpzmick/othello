from ..config import ExperimentConfig, WThorConfig, DatasetConfig
from ..util import SimpleDataLoader
import sys

from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
import pyzstd
import time
import wandb
import os
import lz4.frame

class NN(nn.Module):
    def __init__(self, N1, N2):
        super().__init__()
        # player, valid moves, board state
        self.l1 = nn.Linear(1+64+128, N1)
        self.l2 = nn.Linear(N1, N2)
        self.l3 = nn.Linear(N2, 64)

    def forward(self, X):
        X = torch.relu(self.l1(X))
        X = torch.relu(self.l2(X))
        X = self.l3(X)
        return X

def trainer(model, criterion, optimizer, train, test, epochs=5, start_epoch=0, do_checkpoints=False):
    for epoch in range(start_epoch, start_epoch+epochs):
        start_ts = time.time()
        losses = 0

        for batch_X, batch_y in train:
            batch_X = batch_X.to("cuda")
            batch_y = batch_y.to("cuda")

            optimizer.zero_grad()

            valid = batch_X[:, 1:65]

            y_hat = model.forward(batch_X)
            loss = criterion(y_hat, batch_y, valid)

            loss.backward()
            optimizer.step()

            losses += loss.item()

        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch_X, batch_y in test:
                batch_X = batch_X.to("cuda")
                batch_y = batch_y.to("cuda")
                y_hat = model.forward(batch_X)
                test_correct += torch.sum(batch_y == torch.argmax(y_hat, dim=1))
                test_total += batch_y.shape[0]

        stop_ts = time.time()

        data = {"loss": losses/len(dataloader),
                "test": test_correct/test_total,
                "elapsed": stop_ts - start_ts}

        #wandb.log(data)
        print(f"epoch: {epoch+1}, {data}")

        # if epoch % 100 == 0 and DO_CHECKPOINT:
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         }, f"checkpoint_{epoch}.pt")


c = ExperimentConfig.from_experiment_dir(sys.argv[1])
batch_size = c.net_config.batch_size

if torch.cuda.is_available():
    print("using cuda")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("using mps")
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("loading cached split data")
with lz4.frame.open(c.compressed_split_filename(), mode='rb') as f:
    split = torch.load(f)
    train_inputs = split['train_inputs']
    train_policy = split['train_policy']
    test_inputs = split['test_inputs']
    test_policy = split['test_policy']

print(f"Loaded data to torch.")
print(f"  Train input  dtype={train_inputs.dtype}  shape={train_inputs.shape}")
print(f"  Train policy dtype={train_policy.dtype}    shape={train_policy.shape}")
print(f"  Test input   dtype={test_inputs.dtype}  shape={test_inputs.shape}")
print(f"  Test policy  dtype={test_policy.dtype}    shape={test_policy.shape}")

print(f"There will be {train_inputs.shape[0]/batch_size} batches")

dataset = TensorDataset(train_inputs, train_policy)
dataloader = SimpleDataLoader(dataset, batch_size)
#dataloader = PrefetchDataLoader(dataset, batch_size)

dataset_test = TensorDataset(test_inputs, test_policy)
dataloader_test = SimpleDataLoader(dataset_test, int(test_policy.shape[0]/8))
#dataloader_test = PrefetchDataLoader(dataset_test, int(test_policy.shape[0]/8))

net = NN(c.net_config.N1, c.net_config.N2)
net.to(device)
#wandb.watch(net, log_freq=5, log='all')

optim = torch.optim.Adam(net.parameters(), lr=0.001, amsgrad=True)

# checkpoint = torch.load("checkpoint_700.pt")
# net.load_state_dict(checkpoint['model_state_dict'])
# optim.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
epoch = 0

# maybe don't need this if we do the dedup

# def loss(y_hat, batch_y, valid):
#     invalid = -1 * valid + 1.0
#     ce = torch.nn.functional.cross_entropy(y_hat, batch_y)
#     wrong = torch.mean(invalid * y_hat)
#     return ce+wrong

def loss(y_hat, batch_y, _valid):
    ce = torch.nn.functional.cross_entropy(y_hat, batch_y)
    return ce

print('starting training')
trainer(net, loss, optim, dataloader, dataloader_test, epochs=8192, start_epoch=epoch)
