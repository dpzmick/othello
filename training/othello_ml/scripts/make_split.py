from ..config import open_config
import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

import lz4.frame
import numpy as np
import os
import pyzstd
import time

if __name__ == "__main__":
    np.random.seed(12345)
    c = open_config(sys.argv[1])

    with open(c["files"]["ids_filename"],  'rb') as f:
        decompressed_data = pyzstd.decompress(f.read())
        ids = np.frombuffer(decompressed_data, dtype=np.uint64)

    with open(c["files"]["policy_filename"],  'rb') as f:
        decompressed_data = pyzstd.decompress(f.read())
        policy = np.frombuffer(decompressed_data, dtype=np.float32).reshape( (len(ids), 64) )

    unique_ids = np.unique(ids)

    print(f"There will be {len(ids)} elements in the dataset")
    print(f"With {len(unique_ids)} unique games")

    n_train = int(c["settings"]["train_perc"] * len(unique_ids))

    # shuffle the game ids and select test/train using these ids
    np.random.shuffle(unique_ids)
    train_ids, test_ids = unique_ids[0:n_train], unique_ids[n_train:]

    print(f"Using {len(train_ids)} games to train and {len(test_ids)} games to test")

    # select all the idx in the ids array that map to training and testing data
    # note that keeping the boards roughly in the original order is somewhat important
    # we can load the first board file, process the whole thing, then second, etc
    train_indices = np.where(np.isin(ids, train_ids))[0]
    test_indices = np.where(np.isin(ids, test_ids))[0]

    # we can generate the policy dataset directly because it is small enough to fit in ram
    train_policy, test_policy = policy[train_indices], policy[test_indices]

    # convert policy vectors to index of best board position, saves memory and
    # is what CrossEntropyLoss expects
    train_policy = np.argmax(train_policy, axis=1)
    test_policy = np.argmax(test_policy, axis=1)

    # store as int for crossentropyloss
    train_policy = torch.from_numpy(train_policy)
    test_policy = torch.from_numpy(test_policy)

    # save the policies and ordered ids/idx
    with lz4.frame.open(c["files"]["split_filename"], mode='wb') as f:
        torch.save({
            'train_ids':    train_ids,
            'train_idx':    train_indices,
            'train_policy': train_policy,
            'test_ids':     test_ids,
            'test_idx':     test_indices,
            'test_policy':  test_policy,
            }, f)
