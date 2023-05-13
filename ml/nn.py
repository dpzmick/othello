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
        self.l1 = nn.Linear(1+64+128, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 64)

    def forward(self, X):
        X = torch.relu(self.l1(X))
        X = torch.relu(self.l2(X))
        X = torch.relu(self.l3(X))
        return X


def trainer(model, criterion, optimizer, train, test, epochs=5):
    _losses = []
    _corrects = []

    scaler = torch.cuda.amp.GradScaler()

    # an epoch is a full pass over the input data set
    # but each optim step will only consider BATCH_SIZE worth of data
    for epoch in range(epochs):
        losses = 0
        idx = 0
        for batch_X, batch_y in train:
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                y_hat = model.forward(batch_X)
                assert y_hat.shape == batch_y.shape

                y_indices = torch.argmax(batch_y, dim=1) # CrossEntropyLoss expects argmax to be efficient. Whoops
                loss = criterion(y_hat, y_indices) # calculate loss using the provided criterion function

            scaler.scale(loss).backward()
            scaler.step(optimizer)

            #loss.backward()
            #optimizer.step()
            losses += loss.item()

            scaler.update()


        test_correct = 0
        test_total = 0
        for batch_X, batch_y in test:
            with torch.no_grad():
                y_hat = model.forward(batch_X)
                test_correct += torch.sum(torch.argmax(batch_y, dim=1) == torch.argmax(y_hat, dim=1))
                test_total += batch_y.shape[0]

        _losses.append(losses/len(dataloader))
        _corrects.append(test_correct.cpu().detach().numpy()/test_total)
        print(f"epoch: {epoch+1}, loss: {losses / len(dataloader)}, test result {test_correct/test_total}")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        x = list(range(len(_losses)))
        fig.add_trace(go.Scatter(x=x, y=_losses, name='loss'), secondary_y=False)
        fig.add_trace(go.Scatter(x=x, y=_corrects, name='correct'), secondary_y=True)
        fig.write_html("res2.html")

    return _losses, _corrects

train_perc = 0.8

ids = np.fromfile("../ids.dat", dtype=np.uint64)
inputs = np.fromfile("../input.dat", dtype=np.float32).reshape( (len(ids), 1+64+128) )
policy = np.fromfile("../policy.dat", dtype=np.float32).reshape( (len(ids), 64) )

# note the games are in order so slicing off the last few will produce boards
# which came from games the NN has never seen
np.random.seed(12345)
indices = np.random.permutation(len(ids))

n_train = int(train_perc * len(ids))
train_indicies, test_indicies = indices[0:n_train], indices[n_train:]

train_inputs, test_inputs = inputs[train_indicies], inputs[test_indicies]
train_policy, test_policy = policy[train_indicies], policy[test_indicies]

if torch.cuda.is_available():
    print("using cuda")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("using mps")
    device = torch.device("mps")
else:
    device = torch.device("cpu")

train_inputs = torch.tensor(train_inputs, device=device)
train_policy = torch.tensor(train_policy, device=device)

test_inputs = torch.tensor(test_inputs, device=device)
test_policy = torch.tensor(test_policy, device=device)

#batch_size = int(n_train/16)
batch_size = 512
print(f"There will be {n_train/batch_size} batches for {n_train} training data")

dataset = TensorDataset(train_inputs, train_policy)
dataloader = SimpleDataLoader(dataset, batch_size)

dataset_test = TensorDataset(test_inputs, test_policy)
dataloader_test = SimpleDataLoader(dataset_test, int(test_policy.shape[0]/8))

net = NN()
net.to(device)
# net = torch.nn.DataParallel(net, device_ids=[0 for _ in range(2)]) # use giant batches then let torch split up on gpu
# net.load_state_dict(torch.load("out2.torch", map_location=device))

optim = torch.optim.Adam(net.parameters(), lr=0.001, amsgrad=True)
loss = nn.CrossEntropyLoss()
losses, correct = trainer(net, loss, optim, dataloader, dataloader_test, epochs=8192)

torch.save(net.state_dict(), "out.torch")
