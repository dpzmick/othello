"""Write a small experiment config for local-on-Mac end-to-end smoke
testing of the policy training pipeline.

Run:
    uv run python -m othello_ml.scripts.make_local_config

Then follow the three commands it prints.
"""

import os

from ..config import make_config, setup


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

config = make_config(
    experiment_root=os.path.join(REPO_ROOT, 'experiments'),
    experiment_name='local_reg',
    wthor_dir=os.path.join(REPO_ROOT, 'wthor_files'),
    debug=False,                   # all 46 wthor files
    include_flips=True,
    model_style='small',           # 1024 -> 1024 MLP
    board_lookback=0,              # Othello is Markovian; no history
    train_epochs=32,
    batch_size=4096,               # MLP is tiny; bigger batch = better GPU util
    weight_decay=1e-4,             # cheap L2; model was overfitting hard
)

setup(config)

experiment_dir = config['experiment_dir']
print(f"experiment dir: {experiment_dir}")
print(f"config:         {experiment_dir}config.toml")
print()
print("next, run in sequence:")
print(f"  ./build/apps/data_gen_policy {experiment_dir}config.toml")
print(f"  uv run python -m othello_ml.scripts.make_split {experiment_dir}")
print(f"  WANDB_MODE=disabled uv run python -m othello_ml.scripts.train {experiment_dir}")
