from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
#import othello
import pickle
import random
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(128, 512)
        self.l_last = nn.Linear(512, 1)


    def forward(self, X):
        X = torch.relu(self.l1(X))
        return torch.sigmoid(self.l_last(X))

pred = np.fromfile("../stash-data-gen/training.pred", dtype=np.float32)
games = np.fromfile("../stash-data-gen/training.boards", dtype=np.float32)
games = games.reshape( (len(pred), 128) )

fig = go.Figure()
fig.add_trace(go.Histogram(x=pred, nbinsx=100))
fig.write_html("out.html")
#fig.show()

# dataset = TensorDataset(games, pred)
# dataloader = DataLoader(dataset, batch_size=128)

# net = NN()
# net.load_state_dict(torch.load("from_gpu_3.torch", map_location=torch.device("cpu")))

# print(net.state_dict()['l1.weight'].shape)

# import sys
# sys.exit(0)

# pred = np.zeros(len(pred))
# expect = np.zeros(len(pred))

# with torch.no_grad():
#     idx = 0
#     for X, y in dataloader:
#         y_hat = net.forward(X).flatten()

#         expect[idx:idx+len(y_hat)] = y.detach().cpu().numpy()
#         pred[idx:idx+len(y_hat)] = y_hat.detach().cpu().numpy()

#         idx += len(y_hat)

# fig = go.Figure()
# # fig.add_trace(go.Scatter(
# #     x=np.arange(len(pred)),
# #     y=expect - pred,
# #     mode='markers',
# # ))
# fig.add_trace(go.Histogram(
#     x=expect-pred,
#     nbinsx=100,
# ))
# # fig.write_html("out.html")
# fig.show()
