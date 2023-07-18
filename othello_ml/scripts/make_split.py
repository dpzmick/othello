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
    c = open_config(sys.argv[1])

    with open(c["files"]["ids_filename"],  'rb') as f:
        decompressed_data = pyzstd.decompress(f.read())
        ids = np.frombuffer(decompressed_data, dtype=np.uint64)

    with open(c["files"]["boards_filename"],  'rb') as f:
        decompressed_data = pyzstd.decompress(f.read())
        inputs = np.frombuffer(decompressed_data, dtype=np.float32).reshape( (len(ids), 1+64+128) )

    with open(c["files"]["policy_filename"],  'rb') as f:
        decompressed_data = pyzstd.decompress(f.read())
        policy = np.frombuffer(decompressed_data, dtype=np.float32).reshape( (len(ids), 64) )

    print(f"Read {len(ids)} boards from the input files")

    np.random.seed(12345)
    unique_ids = np.unique(ids)
    np.random.shuffle(unique_ids)

    # unique_ids = unique_ids[0:500] # FIXME make configurable

    print(f"With {len(unique_ids)} unique games")

    n_train = int(c["settings"]["train_perc"] * len(unique_ids))
    train_ids, test_ids = unique_ids[0:n_train], unique_ids[n_train:]

    print(f"Using {len(train_ids)} games to train and {len(test_ids)} games to test")

    train_indices = np.where(np.isin(ids, train_ids))[0]
    test_indices = np.where(np.isin(ids, test_ids))[0]

    train_inputs, test_inputs = inputs[train_indices], inputs[test_indices]
    train_policy, test_policy = policy[train_indices], policy[test_indices]

    # convery policy vectors to index of best option
    # saves memory and is what CrossEntropyLoss expects
    train_policy = np.argmax(train_policy, axis=1)
    test_policy = np.argmax(test_policy, axis=1)

    print(f"-> {train_inputs.shape[0]} boards to train and {test_inputs.shape[0]} boards to test")

    train_inputs = torch.from_numpy(train_inputs)
    train_policy = torch.from_numpy(train_policy)

    test_inputs = torch.from_numpy(test_inputs)
    test_policy = torch.from_numpy(test_policy)

    with lz4.frame.open(c["files"]["split_filename"], mode='wb') as f:
        torch.save({
            'train_inputs': train_inputs,
            'train_policy': train_policy,
            'test_inputs':  test_inputs,
            'test_policy':  test_policy,
            }, f)
