from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import random
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyzstd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import matplotlib

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1+64+128, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 64)

    def forward(self, X):
        X = torch.relu(self.l1(X))
        X = torch.relu(self.l2(X))
        X = torch.relu(self.l3(X))
        return X

    def forward_inference(self, X):
        X = torch.relu(self.l1(X))
        X = torch.relu(self.l2(X))
        X = torch.relu(self.l3(X))
        return torch.sigmoid(X)

    def get_l1(self, X):
        X = torch.relu(self.l1(X))
        return X

    def get_l2(self, X):
        X = torch.relu(self.l1(X))
        X = torch.relu(self.l2(X))
        return X

with open("../ids.dat.zst", 'rb') as f:
    decompressed_data = pyzstd.decompress(f.read())
    ids = np.frombuffer(decompressed_data, dtype=np.uint64)

with open("../input.dat.zst", 'rb') as f:
    decompressed_data = pyzstd.decompress(f.read())
    inputs = np.frombuffer(decompressed_data, dtype=np.float32).reshape( (len(ids), 1+64+128) )

with open("../policy.dat.zst", 'rb') as f:
    decompressed_data = pyzstd.decompress(f.read())
    policy = np.frombuffer(decompressed_data, dtype=np.float32).reshape( (len(ids), 64) )

n_boards = inputs.shape[0]

inputs = torch.tensor(inputs, device="cuda")
policy = torch.tensor(policy, device="cuda")

net = NN()
net.to("cuda")
net.load_state_dict(torch.load("out_8192_8192_128_256_8192_train_with_valid_and_fixed_data_and_dedup_then_with_dup_v2_final_for_real_this_time.torch", map_location="cuda"))

y_hat = net.forward_inference(inputs)

top_values, top_indices = torch.topk(y_hat, k=3, dim=1)

# how many select anything
row_has_value_above_threshold = torch.any(top_values > 0.5, dim=1)
count = torch.sum(row_has_value_above_threshold)

print(f"{count} of {n_boards} actually select something ({count/n_boards * 100}%)")
