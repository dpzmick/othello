from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# hack around pytorch dataloader being slow for tensors in ram
class SimpleDataLoader(object):
    def __init__(self, dset, batch_size):
        iterable = []
        for i in range(0, int(len(dset)/batch_size)+1):
            st = i * batch_size
            ed = min((i+1) * batch_size, len(dset))

            if ed <= st:
                break

            x = dset[st:ed]

            if len(x[1]):
                iterable.append(x)

        self._inner = iterable

    def __iter__(self):
        return iter(self._inner)

    def __len__(self):
        return len(self._inner)


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(128, 512)
        self.l2 = nn.Linear(512, 512)
        self.l_last = nn.Linear(512, 1)

    def forward(self, X):
        X = torch.relu(self.l1(X))
        X = torch.relu(self.l2(X))
        return self.l_last(X)

iters = 64

plot_epoch = int(iters/8)
fig = make_subplots(rows=int(iters/plot_epoch), cols=2, shared_xaxes=True)

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

pred = np.fromfile("../stash-data-gen/training.pred", dtype=np.float32)
games = np.fromfile("../stash-data-gen/training.boards", dtype=np.float32)
games = games.reshape( (len(pred), 128) )

pred[pred==0.5] = 0.0

# games = games[0:100,:]
# pred = pred[0:100]

print(f"Loaded {len(pred)} total boards")

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print('using cuda')
# else:
#     device = torch.device("cpu")

# no point, just use cpu
device = torch.device("cpu")

games = torch.tensor(games)
pred = torch.tensor(pred)

games = games.to(device)
pred = pred.to(device)

#train_size = 1024
train_size = int(len(pred)*0.8)
test_size  = int(len(pred)-train_size)
batch_size = int(train_size/16)
print(f"There will be {train_size/batch_size} batches for {train_size} training data")

dataset = TensorDataset(games, pred)
train, test = torch.utils.data.random_split(dataset, (train_size, test_size))
#dataloader = DataLoader(train, batch_size=batch_size)
dataloader = SimpleDataLoader(train, batch_size)

net = NN()
#net.load_state_dict(torch.load("out.torch", map_location=device))

net = net.to(device)
optim = torch.optim.Adam(net.parameters(), lr=0.01)

loss = nn.MSELoss()
trainer(net, loss, optim, dataloader, epochs=iters)
fig.write_html("out.html")

torch.save(net.state_dict(), "out2.torch")

# # run over a bunch of test data
# fig = make_subplots(rows=1, cols=2, shared_xaxes=True)
# with torch.no_grad():
#     print(f"there are {len(test)} test samples")
#     dataloader = DataLoader(test, batch_size=len(test))
#     test_X, test_y = next(iter(dataloader))

#     y_hat = net.forward(test_X).flatten()

#     a=y_hat.cpu().detach().numpy()
#     b=test_y.cpu().detach().numpy()

#     fig.add_trace(
#         go.Histogram(x=a, nbinsx=100, marker=dict(color='red')),
#         row=1, col=1)

#     fig.add_trace(
#         go.Histogram(x=b, nbinsx=100, marker=dict(color='red')),
#         row=1, col=2)

# fig.write_html("res.html")
