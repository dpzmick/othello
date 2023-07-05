from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyzstd
import time

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
        self.l1 = nn.Linear(1+64+128, 1024)
        self.l2 = nn.Linear(1024, 1024)
        self.l3 = nn.Linear(1024, 64)

    def forward(self, X):
        X = torch.relu(self.l1(X))
        X = torch.relu(self.l2(X))
        X = torch.relu(self.l3(X))
        return X


def trainer(model, criterion, optimizer, train, test, epochs=5):
    _losses = []
    _corrects = []

    for epoch in range(epochs):
        start_ts = time.time()
        losses = 0
        idx = 0
        for batch_X, batch_y in train:
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
                y_hat = model.forward(batch_X)
                test_correct += torch.sum(batch_y == torch.argmax(y_hat, dim=1))
                test_total += batch_y.shape[0]

        _losses.append(losses/len(dataloader))
        _corrects.append(test_correct.cpu().detach().numpy()/test_total)
        stop_ts = time.time()
        print(f"epoch: {epoch+1}, loss: {losses / len(dataloader)}, test result {test_correct/test_total}, took {stop_ts - start_ts}")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        x = list(range(len(_losses)))
        fig.add_trace(go.Scatter(x=x, y=_losses, name='loss'), secondary_y=False)
        fig.add_trace(go.Scatter(x=x, y=_corrects, name='correct'), secondary_y=True)
        fig.write_html("res2.html")

train_perc = 0.8

with open("../sym_ids_withdups.dat.zst", 'rb') as f:
    decompressed_data = pyzstd.decompress(f.read())
    ids = np.frombuffer(decompressed_data, dtype=np.uint64)

with open("../sym_input_withdups.dat.zst", 'rb') as f:
    decompressed_data = pyzstd.decompress(f.read())
    inputs = np.frombuffer(decompressed_data, dtype=np.float32).reshape( (len(ids), 1+64+128) )

with open("../sym_policy_withdups.dat.zst", 'rb') as f:
    decompressed_data = pyzstd.decompress(f.read())
    policy = np.frombuffer(decompressed_data, dtype=np.float32).reshape( (len(ids), 64) )

print(f"Read {len(ids)} boards from the input files")

np.random.seed(12345)
unique_ids = np.unique(ids)
np.random.shuffle(unique_ids)

#unique_ids = unique_ids[0:500]

print(f"With {len(unique_ids)} unique games")

n_train = int(train_perc * len(unique_ids))
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

# if torch.cuda.is_available():
#     print("using cuda")
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     print("using mps")
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")

device = torch.device("cpu")

train_inputs = torch.from_numpy(train_inputs).to(device)
train_policy = torch.from_numpy(train_policy).to(device)

test_inputs = torch.from_numpy(test_inputs).to(device)
test_policy = torch.from_numpy(test_policy).to(device)

# the board dataset size is about 11gigs
# the policy sizes are about 120megs
#
# macbook has 16gigs of ram, but to use MPS data has to be pinned (probably)
# but mps is broken anyway...

print(f"Loaded data to torch.")
print(f"  Train input  dtype={train_inputs.dtype}  shape={train_inputs.shape}")
print(f"  Train policy dtype={train_policy.dtype}    shape={train_policy.shape}")
print(f"  Test input   dtype={test_inputs.dtype}  shape={test_inputs.shape}")
print(f"  Test policy  dtype={test_policy.dtype}    shape={test_policy.shape}")

batch_size = 2048
print(f"There will be {train_inputs.shape[0]/batch_size} batches")

dataset = TensorDataset(train_inputs, train_policy)
dataloader = SimpleDataLoader(dataset, batch_size)

dataset_test = TensorDataset(test_inputs, test_policy)
dataloader_test = SimpleDataLoader(dataset_test, int(test_policy.shape[0]/8))

net = NN()
net.to(device)
#net.load_state_dict(torch.load("out_8192_8192_128_256_train_with_valid_and_fixed_data_and_dedup.torch", map_location=device))

optim = torch.optim.Adam(net.parameters(), lr=0.001, amsgrad=True)

# custom loss than penalizes playing in invalid places
# maybe don't need this if we do the dedup

# def loss(y_hat, batch_y, valid):
#     invalid = -1 * valid + 1.0
#     ce = torch.nn.functional.cross_entropy(y_hat, batch_y)
#     wrong = torch.mean(invalid * y_hat)
#     return ce+wrong

def loss(y_hat, batch_y, _valid):
    ce = torch.nn.functional.cross_entropy(y_hat, batch_y)
    return ce

trainer(net, loss, optim, dataloader, dataloader_test, epochs=8192)

torch.save(net.state_dict(), "out.torch")
