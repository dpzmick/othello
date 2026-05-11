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

with open("../sym_ids.dat.zst", 'rb') as f:
    decompressed_data = pyzstd.decompress(f.read())
    ids = np.frombuffer(decompressed_data, dtype=np.uint64)

with open("../sym_input.dat.zst", 'rb') as f:
    decompressed_data = pyzstd.decompress(f.read())
    inputs = np.frombuffer(decompressed_data, dtype=np.float32).reshape( (len(ids), 1+64+128) )

with open("../sym_policy.dat.zst", 'rb') as f:
    decompressed_data = pyzstd.decompress(f.read())
    policy = np.frombuffer(decompressed_data, dtype=np.float32).reshape( (len(ids), 64) )

n_boards = inputs.shape[0]

net = NN()
net.load_state_dict(torch.load("out_8192_8192_128_256_8192_train_with_valid_and_fixed_data_and_dedup_then_with_dup_v2_final_for_real_this_time.torch"))


app = Dash()

def selector():
    return dcc.Dropdown(list(range(100)), 1, id='my-input')

@app.callback(
    Output(component_id='display', component_property='children'),
    Input(component_id='my-input', component_property='value')
)
def display_board(i=0):
    inpt = inputs[i,:]
    pl   = policy[i,:].reshape((8,8))

    with torch.no_grad():
        t = torch.tensor(inpt)
        l1 = net.get_l1(t)
        l2 = net.get_l2(t)
        est = net.forward_inference(t).numpy().reshape((8,8))

    player = inpt[1]
    valid  = inpt[1:65].reshape((8,8)) # valid moves
    white  = inpt[1+64:1+64+64].reshape((8,8))
    black  = inpt[1+64+64:1+64+64+64].reshape((8,8)) * -1

    board = white + black

    def make_cell(x,y):
        if board[x,y] > 0:
            s = 'W'
        elif board[x,y] < 0:
            s = 'B'
        else:
            s = ' '

        style = {'height': '15px', 'width': '15px'}
        if pl[x,y] > 0:
            style["border"] = "3pt solid red"

        cmap = matplotlib.colormaps['OrRd']
        color = cmap(est[x,y], bytes=True)
        style["background-color"] = f"rgb({color[0]}, {color[1]}, {color[2]})"

        return html.Td(s, style=style)

    table = html.Table(
        id='table',
        children=[html.Tr(children=[make_cell(x,y) for x in range(8)]) for y in range(8)],
        style={'table-layout': 'fixed'})

    return html.Div(id='display', children=[
        table,
        html.Div(children=[
            dcc.Graph(figure=px.imshow(l1.reshape((32,16))), style={"display": "inline-block", "width": "30%"}),
            dcc.Graph(figure=px.imshow(l2.reshape((32,16))), style={"display": "inline-block", "width": "30%"}),
            dcc.Graph(figure=px.imshow(est), style={"display": "inline-block", "width": "30%"}),
        ]),
    ])

app.layout = html.Div(children=[
    selector(),
    html.Br(),
    display_board(),
])
app.run_server(host='0.0.0.0')
