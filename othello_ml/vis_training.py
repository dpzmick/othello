from plotly import graph_objects as go
import glob
import numpy as np

def print_game(game):
    g = np.zeros((8,8), dtype=np.int8)
    game = np.array(game, dtype=np.int8)

    white = game[0:64].reshape( (8,8) )
    black = game[64:128].reshape( (8,8) )

    g |= white
    g |= -1 * black

    print(g)

preds = np.fromfile("data_gen/training.pred", dtype=np.float32)
#boards = np.fromfile("data_gen/training.boards", dtype=np.float32)
#boards = boards.reshape( (len(preds), 128) )

#print_game(boards[12,:])

fig = go.Figure()
fig.add_trace(go.Histogram(x=preds, nbinsx=100))
fig.show()
