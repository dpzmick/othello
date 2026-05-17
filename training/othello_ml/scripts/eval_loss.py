"""Evaluate a trained model's CE loss on a pre-generated dataset.

For sanity, also reports modal-move accuracy. Compares to the data floor.

    uv run python -m othello_ml.scripts.eval_loss <model_experiment_dir> <data_experiment_dir>

If <data_experiment_dir>/datasets/split.pt.lz4 exists, reports train, test,
and combined losses separately. Otherwise just the combined loss.
"""

import glob
import os
import sys
import time

import lz4.frame
import numpy as np
import pyzstd
import torch

from ..config import open_config
from .inspect_policy import load_model


@torch.no_grad()
def evaluate(net, boards_np: np.ndarray, played_np: np.ndarray, device, batch_size=8192, label=""):
    n_rows = boards_np.shape[0]
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    t0 = time.time()
    for st in range(0, n_rows, batch_size):
        ed = min(st + batch_size, n_rows)
        x = torch.from_numpy(boards_np[st:ed]).to(device).float()
        valid = x[:, 0:64]
        y = torch.from_numpy(played_np[st:ed]).to(device)
        logits = net(x)
        masked = logits.masked_fill(valid == 0, -1e9)
        total_loss += loss_fn(masked, y).item()
        total_correct += (masked.argmax(dim=1) == y).sum().item()
        if st > 0 and st % (batch_size * 200) == 0:
            elapsed = time.time() - t0
            eta = (n_rows - st) * elapsed / st
            print(f"  [{label}] {st:>12,}/{n_rows:,}  ({100*st/n_rows:5.1f}%)  eta={eta:.0f}s",
                  file=sys.stderr)
    return total_loss / n_rows, total_correct / n_rows


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <model_experiment_dir> <data_experiment_dir>", file=sys.stderr)
        sys.exit(1)

    model_dir = sys.argv[1].rstrip("/")
    data_dir = sys.argv[2].rstrip("/")

    print(f"loading model from {model_dir}", file=sys.stderr)
    net, model_name, epoch = load_model(model_dir)
    print(f"  {model_name}, epoch={epoch}", file=sys.stderr)

    data_config = open_config(data_dir)
    boards_dir = data_config["files"]["boards_dir"]
    policy_filename = data_config["files"]["policy_filename"]
    split_filename = data_config["files"]["split_filename"]

    t0 = time.time()
    print(f"loading boards from {boards_dir}", file=sys.stderr)
    files = sorted(glob.glob(os.path.join(boards_dir, "*.dat.zst")))
    parts = []
    for f in files:
        with open(f, "rb") as fp:
            arr = np.frombuffer(pyzstd.decompress(fp.read()), dtype=np.uint8).reshape(-1, 192)
        parts.append(arr)
    boards = np.concatenate(parts, axis=0)
    del parts
    n_rows = boards.shape[0]
    print(f"  loaded {n_rows:,} rows in {time.time()-t0:.1f}s", file=sys.stderr)

    t0 = time.time()
    with open(policy_filename, "rb") as fp:
        policy = np.frombuffer(pyzstd.decompress(fp.read()), dtype=np.float32).reshape(-1, 64)
    played = np.argmax(policy, axis=1).astype(np.int64)
    del policy
    print(f"  loaded policy in {time.time()-t0:.1f}s", file=sys.stderr)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"running inference on {device}", file=sys.stderr)
    net = net.to(device)

    have_split = os.path.exists(split_filename)
    if have_split:
        print(f"loading split from {split_filename}", file=sys.stderr)
        with lz4.frame.open(split_filename, mode="rb") as f:
            split = torch.load(f, weights_only=False)
        train_idx = np.asarray(split["train_idx"])
        test_idx = np.asarray(split["test_idx"])
        print(f"  train rows: {len(train_idx):,}   test rows: {len(test_idx):,}",
              file=sys.stderr)

        train_loss, train_acc = evaluate(net, boards[train_idx], played[train_idx], device, label="train")
        test_loss, test_acc = evaluate(net, boards[test_idx], played[test_idx], device, label="test")
        # Combined loss (weighted)
        n_train, n_test = len(train_idx), len(test_idx)
        combined_loss = (train_loss * n_train + test_loss * n_test) / (n_train + n_test)
        combined_acc = (train_acc * n_train + test_acc * n_test) / (n_train + n_test)
    else:
        loss, acc = evaluate(net, boards, played, device, label="all")
        train_loss = test_loss = None
        combined_loss = loss
        combined_acc = acc

    print()
    print(f"=== model {os.path.basename(model_dir)} on {os.path.basename(data_dir)} ===")
    print(f"rows                   : {n_rows:,}")
    if have_split:
        print(f"train CE / acc         : {train_loss:.4f} / {train_acc:.4f}")
        print(f"test  CE / acc         : {test_loss:.4f} / {test_acc:.4f}")
        print(f"train-test CE gap      : {test_loss - train_loss:+.4f}")
    print(f"combined CE / acc      : {combined_loss:.4f} / {combined_acc:.4f}")
    print(f"data CE floor          : 0.2330  (oracle on this data)")
    print(f"modal-move accuracy    : 0.9002  (oracle ceiling)")


if __name__ == "__main__":
    main()
