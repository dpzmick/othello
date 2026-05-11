from ..config import open_config
from ..util import BoardDirLoader, SimpleDataLoader
from ..util.tracer import Tracer
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

class NN_with_norm(nn.Module):
    def __init__(self, input_shape, N1, N2):
        super().__init__()
        # player, valid moves, board state
        self.l1 = nn.Linear(input_shape, N1)
        self.l1_norm = nn.BatchNorm1d(N1)
        self.l2 = nn.Linear(N1, N2)
        self.l2_norm = nn.BatchNorm1d(N2)
        self.l3 = nn.Linear(N2, 64)

        # NOTE: pretty sure including the current player as input is not very
        # helpful

    def forward(self, X):
        X = torch.relu(self.l1_norm(self.l1(X)))
        X = torch.relu(self.l2_norm(self.l2(X)))
        X = self.l3(X)
        return X

class CNN(nn.Module):
    def __init__(self, input_shape, N1, N2):
        super().__init__()
        # HACK: the input we're provided includes an extra
        # value for current player. We are going to ignore that

        # input is input_channels worth of 8x8 boards
        n_boards = (input_shape - 1)

        # we will ignore that the first input is the valid inputs layer
        # and just treat each 8x8 input as a separate channel
        n_channels = n_boards // 64

        # input: (n_channel, 8, 8)
        # output: (N1, 8, 8)
        self.l1 = nn.Conv2d(n_channels, N1, kernel_size=(3,3), padding=1)
        self.l1_norm = nn.BatchNorm2d(N1)

        # input: (N1, 8, 8)
        # output: (N2, 8, 8)
        self.l2 = nn.Conv2d(N1, N2, kernel_size=(3,3), padding=1)
        self.l2_norm = nn.BatchNorm2d(N2)

        # flattened, we'll have N2*8*8 inputs
        # then run through linear layer to select the place to play

        self.l3 = nn.Linear(N2*8*8, 64)

        # save N2 for reshaping
        self.N2 = N2

    def forward(self, X):
        # HACK: chop off the first entry of every input in the batch
        # we're removing the "whose turn is it" input
        X = X[:,1:]

        batch_size = X.shape[0]
        n_channels = X.shape[1] // 64
        X = torch.reshape(X, (batch_size, n_channels, 8, 8) )

        X = torch.relu(self.l1_norm(self.l1(X)))
        X = torch.relu(self.l2_norm(self.l2(X)))

        # flatten out and apply a final linear layer
        # assert X.shape[0] == batch_size # slow
        X = torch.reshape( X, (batch_size, self.N2*8*8) )
        X = self.l3(X)

        return X

# def loss_with_invalid(y_hat, batch_y, valid):
#     # gets lower as we select better
#     ce = torch.nn.functional.cross_entropy(y_hat, batch_y)

#     # needs to get lower as we select better
#     #
#     # outputs like this should be penalized:
#     # valid: [1 0 1 0 0 0 1 ...]
#     # output: [0 1 0 0 ...]
#     #
#     # but this should be rewarded
#     # valid: [1 0 1 0 0 0 1 ...]
#     # output: [1 0 0 0 ...]

#     # this doens't work
#     invalid = -1 * valid + 1.0
#     wrong = torch.mean(invalid * y_hat)
#     return ce+wrong

def loss_without_invalid(y_hat, batch_y, _valid):
    # NOTE: per-row one-hot targets from data_gen_policy.c. Duplicate boards
    # are not pre-aggregated into soft policy distributions; cross-entropy
    # with N one-hot rows is equivalent (up to a 1/N scalar) to one row
    # with the empirical soft target, so the network still learns the right
    # distribution. Pre-deduping with sample weights would reduce per-epoch
    # compute -- see the matching note in apps/data_gen_policy.c.
    ce = torch.nn.functional.cross_entropy(y_hat, batch_y)
    return ce

def train_step(model, optimizer, batch_X, batch_y):
    optimizer.zero_grad()
    y_hat = model.forward(batch_X)
    #loss = criterion(y_hat, batch_y, valid)
    loss = loss_without_invalid(y_hat, batch_y, None) # ignoring config

    loss.backward()
    optimizer.step()

# NotImplementedError: could not find kernel for aten._foreach_maximum_.List at dispatch key DispatchKey.Meta
# train_step = torch.compile(train_step) # cannot compile this, something about max not being supported?
# just compiling the model isn't really interesting at all but I guess that's all we're gonna get

def trainer(t, model, criterion, optimizer, X_train, y_train, X_test, y_test, checkpoint_dir, epochs=5, start_epoch=0):
    for epoch in range(start_epoch, start_epoch+epochs):
        start_ts = time.time()

        # Accumulate loss as a tensor on the device. .item() forces a
        # GPU->CPU sync that would otherwise bottleneck the per-batch loop;
        # we sync once per epoch at the end instead.
        losses_t = torch.zeros((), device=device)
        train_total = 0

        with t.start("train"):
            for batch_X, batch_y in zip(X_train, y_train):
                # Data already lives on device thanks to the loader setup.
                valid = batch_X[:, 1:65]

                optimizer.zero_grad()
                y_hat = model.forward(batch_X)
                loss = criterion(y_hat, batch_y, valid)

                loss.backward()
                optimizer.step()

                losses_t = losses_t + loss.detach()
                train_total += batch_y.shape[0]

        losses = losses_t.item()

        train_end  = time.time()
        test_start = time.time()

        test_correct = 0
        test_total = 0
        with t.start("test"):
            with torch.no_grad(): # GPU run is much faster when grad is disabled
                for batch_X, batch_y in zip(X_test, y_test):
                    # Data already on device.
                    y_hat = model.forward(batch_X)
                    test_correct += torch.sum(batch_y == torch.argmax(y_hat, dim=1))
                    test_total += batch_y.shape[0]

        stop_ts = time.time()

        data = {"loss": losses/train_total,
                "test": test_correct/test_total,
                "elapsed": stop_ts - start_ts,
                "train_elapsed": train_end - start_ts,
                "test_elapsed": stop_ts - train_end}

        # Save a checkpoint every 16 epochs. Originally was every 64 when one
        # run was ~8 hours on the training cluster; for the local M1 runs
        # (minutes per 128 epochs) we want more granularity.
        if epoch % 16 == 0:
            path = f'{checkpoint_dir}/checkpoint_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, path)

            with open(f'{checkpoint_dir}/LATEST', "w") as f:
                f.write(path)


        wandb.log(data)
        print(f"epoch: {epoch+1}, {data}")

    # Always save a final checkpoint so short runs leave behind something
    # useful (the epoch-interval rule above might miss the last epoch).
    final_path = f'{checkpoint_dir}/final.pt'
    torch.save({
        'epoch': start_epoch + epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, final_path)
    with open(f'{checkpoint_dir}/LATEST', "w") as f:
        f.write(final_path)

if __name__ == "__main__":
    c = open_config(sys.argv[1])
    t = Tracer(c["files"]["trace_file"], enabled=False)

    batch_size      = c["settings"]["batch_size"]
    boards_dir      = c["files"]["boards_dir"]
    checkpoint_dir  = c["files"]["checkpoint_dir"]
    input_shape     = c["settings"]["model_params"]["input_shape"]
    boards_per_file = c["settings"]["boards_per_file"]

    # have 4 cores, but using 2 of them for IO threads
    torch.set_num_threads(2)

    wandb.init(project='othello', name=c["name"], config=c["settings"])

    if torch.cuda.is_available():
        print("using cuda")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("using mps")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    with t.start('load_split_file'):
        with lz4.frame.open(c["files"]["split_filename"], mode='rb') as f:
            # weights_only=False because the split file contains numpy arrays
            # alongside torch tensors; we're loading our own file so the
            # arbitrary-code-execution risk is moot. (Could be cleaned up by
            # converting numpy arrays to torch tensors in make_split.py.)
            split = torch.load(f, weights_only=False)
            train_policy = split['train_policy']
            test_policy = split['test_policy']

            train_indices = split['train_idx']
            test_indices  = split['test_idx']

    print(f'{train_policy.shape} entries in train_policy')
    print(f'{test_policy.shape} entries in test_policy')

    # Move policy targets to device once. Cheap and avoids the per-batch
    # host->device copy that the SimpleDataLoader would otherwise need.
    train_policy = train_policy.to(device)
    test_policy = test_policy.to(device)

    with t.start('create_loaders'):
        board_dataloader = BoardDirLoader('train', boards_dir, boards_per_file, input_shape, train_indices, batch_size, t, device=device)
        policy_dataloader = SimpleDataLoader(train_policy, batch_size)

        board_dataloader_test = BoardDirLoader('test', boards_dir, boards_per_file, input_shape, test_indices, batch_size, t, device=device)
        policy_dataloader_test = SimpleDataLoader(test_policy, batch_size)

    # load the model and loss function from the config file
    with t.start('create model'):
        net  = getattr(sys.modules[__name__], c["settings"]["model_name"])(**c["settings"]["model_params"])
        loss = eval(c["settings"]["loss_variant"])

    # net = torch.compile(net) # doens't work with batch norm!
    net.to(device)
    wandb.watch(net, log_freq=5, log='all')

    optim = torch.optim.Adam(net.parameters(), lr=0.001, amsgrad=True, weight_decay=float(c["settings"]["weight_decay"]))

    # load checkpoint
    try:
        with open(checkpoint_dir + "/LATEST", "r") as f:
            latest_fname = f.readline()
            print(f"Reading {latest_fname} as checkpoint file")
            cp = torch.load(latest_fname)

            net.load_state_dict(cp['model_state_dict'])
            optim.load_state_dict(cp['optimizer_state_dict'])

            start_epoch = cp['epoch']
    except FileNotFoundError:
        print('no checkpoints found')
        start_epoch = 0

    print('starting training')
    trainer(t, net, loss, optim, board_dataloader, policy_dataloader, board_dataloader_test, policy_dataloader_test, checkpoint_dir, epochs=c["settings"]["train_epochs"], start_epoch=start_epoch)

    t.close()
