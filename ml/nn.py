from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import othello
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

if torch.cuda.is_available():
    iters = 8192
else:
    iters = 16

# plot_epoch = 4
# fig = make_subplots(rows=int(iters/plot_epoch), cols=2, shared_xaxes=True)

def trainer(model, criterion, optimizer, dataloader, epochs=5):
    global fig

    # an epoch is a full pass over the input data set
    # but each optim step will only consider BATCH_SIZE worth of data
    for epoch in range(epochs):
        losses = 0
        idx = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            y_hat = model.forward(batch_X).flatten()
            loss = criterion(batch_y, y_hat) # calculate loss using the provided criterion function

            # if epoch % plot_epoch == 0 and idx==0:
            #     a=y_hat.cpu().detach().numpy()
            #     b=batch_y.cpu().detach().numpy()

            #     fig.add_trace(
            #         go.Histogram(x=a, nbinsx=100, marker=dict(color='red')),
            #         row=int(epoch/plot_epoch) + 1, col=1)

            #     fig.add_trace(
            #         go.Histogram(x=b, nbinsx=100, marker=dict(color='blue')),
            #         row=int(epoch/plot_epoch) + 1, col=2)

            loss.backward()

            idx += 1

            optimizer.step()
            losses += loss.item() # add loss to running total? Why track _running_ total

        print(f"epoch: {epoch+1}, loss: {losses / len(dataloader)}")

def print_game(game):
    g = np.zeros((8,8), dtype=np.int8)
    game = np.array(game, dtype=np.int8)

    white = game[0:64].reshape( (8,8) )
    black = game[64:128].reshape( (8,8) )

    g |= white
    g |= -1 * black

    print(g)

pred = np.fromfile("data_gen/training.pred", dtype=np.float32)
games = np.fromfile("data_gen/training.boards", dtype=np.float32)
games = games.reshape( (len(pred), 128) )

# pred = pred[0:1000]
# games = games[0:1000,:]

print(f"Loaded {len(pred)} total boards")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

games = torch.tensor(games)
pred = torch.tensor(pred)

games = games.to(device)
pred = pred.to(device)

if torch.cuda.is_available():
    device = torch.device("cuda")

batch_size = 128
train_size = int(len(pred)*0.8)
test_size  = int(len(pred)-train_size)
print(f"There will be {train_size/batch_size} batches")

dataset = TensorDataset(games, pred)
train, test = torch.utils.data.random_split(dataset, (train_size, test_size))
dataloader = DataLoader(train, batch_size=batch_size)

net = NN()
net.load_state_dict(torch.load("out.torch", map_location=device))

net = net.to(device)
optim = torch.optim.Adam(net.parameters(), lr=0.001)

# MSE is correct, or at least more correct than CrossEntropy.
# we are predicting _values_, not probabilty of class membership
#
# we are really predicting the number of trials (out of 1000) that will result
# in winning, where this is stored as W/1000
loss = nn.MSELoss()
trainer(net, loss, optim, dataloader, epochs=iters)

torch.save(net.state_dict(), "out.torch")

pred = np.zeros(len(test))
expect = np.zeros(len(test))

with torch.no_grad():
    testloader = DataLoader(test, batch_size=batch_size)

    idx = 0
    for X, y in testloader:
        y_hat = net.forward(X).flatten()

        expect[idx:idx+len(y_hat)] = y.detach().cpu().numpy()
        pred[idx:idx+len(y_hat)] = y_hat.detach().cpu().numpy()

        idx += len(y_hat)

fig = go.Figure()
# fig.add_trace(go.Scatter(
#     x=np.arange(len(pred)),
#     y=expect - pred,
#     mode='markers',
# ))
fig.add_trace(go.Histogram(
    x=expect-pred,
    nbinsx=100,
))
fig.write_html("out.html")
#fig.show()
