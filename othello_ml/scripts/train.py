from ..config import open_config
from ..util import BoardDirLoader, SimpleDataLoader
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
    def __init__(self, input_shape, N1, N2):
        super().__init__()
        # player, valid moves, board state
        self.l1 = nn.Linear(input_shape, N1)
        self.l2 = nn.Linear(N1, N2)
        self.l3 = nn.Linear(N2, 64)

    def forward(self, X):
        X = torch.relu(self.l1(X))
        X = torch.relu(self.l2(X))
        X = self.l3(X)
        return X

def loss_with_invalid(y_hat, batch_y, valid):
    # gets lower as we select better
    ce = torch.nn.functional.cross_entropy(y_hat, batch_y)

    # needs to get lower as we select better
    #
    # outputs like this should be penalized:
    # valid: [1 0 1 0 0 0 1 ...]
    # output: [0 1 0 0 ...]
    #
    # but this should be rewarded
    # valid: [1 0 1 0 0 0 1 ...]
    # output: [1 0 0 0 ...]

    # this doens't work
    invalid = -1 * valid + 1.0
    wrong = torch.mean(invalid * y_hat)
    return ce+wrong

def loss_without_invalid(y_hat, batch_y, _valid):
    ce = torch.nn.functional.cross_entropy(y_hat, batch_y)
    return ce

def trainer(model, criterion, optimizer, X_train, y_train, X_test, y_test, epochs=5, start_epoch=0, do_checkpoints=False):
    for epoch in range(start_epoch, start_epoch+epochs):
        start_ts = time.time()

        losses = 0
        train_total = 0

        for batch_X, batch_y in zip(X_train, y_train):
            batch_X = batch_X.to("cuda")
            batch_y = batch_y.to("cuda") # FIXME do these transfers async as well

            optimizer.zero_grad()

            valid = batch_X[:, 1:65]

            y_hat = model.forward(batch_X)
            loss = criterion(y_hat, batch_y, valid)

            loss.backward()
            optimizer.step()

            losses += loss.item()
            train_total += batch_y.shape[0]

        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch_X, batch_y in zip(X_test, y_test):
                batch_X = batch_X.to("cuda")
                batch_y = batch_y.to("cuda")
                y_hat = model.forward(batch_X)
                test_correct += torch.sum(batch_y == torch.argmax(y_hat, dim=1))
                test_total += batch_y.shape[0]

        stop_ts = time.time()

        data = {"loss": losses/train_total,
                "test": test_correct/test_total,
                "elapsed": stop_ts - start_ts}

        wandb.log(data)
        print(f"epoch: {epoch+1}, {data}")

c = open_config(sys.argv[1])

batch_size      = c["settings"]["batch_size"]
boards_dir      = c["files"]["boards_dir"]
input_shape     = c["settings"]["model_params"]["input_shape"]
boards_per_file = c["settings"]["boards_per_file"]

wandb.init(project='othello', name=c["name"], config=c["settings"])

if torch.cuda.is_available():
    print("using cuda")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("using mps")
    device = torch.device("mps")
else:
    device = torch.device("cpu")

with lz4.frame.open(c["files"]["split_filename"], mode='rb') as f:
    split = torch.load(f)
    train_policy = split['train_policy']
    test_policy = split['test_policy']

    train_indices = split['train_idx']
    test_indices  = split['test_idx']

board_dataloader = BoardDirLoader(boards_dir, boards_per_file, input_shape, train_indices, batch_size)
policy_dataloader = SimpleDataLoader(train_policy, batch_size)

board_dataloader_test = BoardDirLoader(boards_dir, boards_per_file, input_shape, test_indices, int(test_policy.shape[0]/8))
policy_dataloader_test = SimpleDataLoader(test_policy, int(test_policy.shape[0]/8))

# load the model and loss function from the config file
net  = getattr(sys.modules[__name__], c["settings"]["model_name"])(**c["settings"]["model_params"])
loss = eval(c["settings"]["loss_variant"])

net.to(device)
wandb.watch(net, log_freq=5, log='all')

optim = torch.optim.Adam(net.parameters(), lr=0.001, amsgrad=True, weight_decay=float(c["settings"]["weight_decay"]))

print('starting training')
trainer(net, loss, optim, board_dataloader, policy_dataloader, board_dataloader_test, policy_dataloader_test, epochs=c["settings"]["train_epochs"], start_epoch=0)
