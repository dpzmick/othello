from ..config import open_config
from ..util import BoardDirLoader, SimpleDataLoader
from ..util.tracer import Tracer

import sys
import lz4.frame
import torch

c = open_config(sys.argv[1])

batch_size      = c["settings"]["batch_size"]
boards_dir      = c["files"]["boards_dir"]
input_shape     = c["settings"]["model_params"]["input_shape"]
boards_per_file = c["settings"]["boards_per_file"]

with lz4.frame.open(c["files"]["split_filename"], mode='rb') as f:
    split = torch.load(f)
    train_policy = split['train_policy']
    test_policy = split['test_policy']

    train_indices = split['train_idx']
    test_indices  = split['test_idx']

t = Tracer("out.trace")

board_dataloader = BoardDirLoader('train', boards_dir, boards_per_file, input_shape, train_indices, batch_size, t)
policy_dataloader = SimpleDataLoader(train_policy, batch_size)

for i in range(0,3):
    print(i)
    for batch_x, batch_Y in zip(board_dataloader, policy_dataloader):
        pass

print('done')
