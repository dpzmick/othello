from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(128, 512)
        self.l_last = nn.Linear(512, 1)

    def forward(self, X):
        X = torch.relu(self.l1(X))
        X = self.l_last(X)
        return torch.sigmoid(X)


net = NN()
net.load_state_dict(torch.load("from_gpu_final.torch", map_location=torch.device("cpu")))

print("static const float l1_weights[] = {")
for ws in net.state_dict()["l1.weight"]:
    for w in ws:
        print(f"  {w}f,")
print("};")

print("static const float l1_biases[] = {")
for b in net.state_dict()["l1.bias"]:
    print(f"  {b}f,")
print("};")

print("static const float l2_weights[] = {")
for ws in net.state_dict()["l_last.weight"]:
    for w in ws:
        print(f"  {w}f,")
print("};")

print("static const float l2_biases[] = {")
for b in net.state_dict()["l_last.bias"]:
    print(f"  {b}f,")
print("};")
