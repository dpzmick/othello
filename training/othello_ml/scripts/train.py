from ..config import open_config
from ..util import BoardDirLoader, SimpleDataLoader
from ..util.tracer import Tracer
import sys

from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
import pyzstd
import time
import wandb
import os
import lz4.frame

class NN(nn.Module):
    def __init__(self, input_shape, N1, N2):
        super().__init__()
        # player, valid moves, board state
        self.l1 = nn.Linear(input_shape, N1)
        self.l2 = nn.Linear(N1, N2)
        self.l3 = nn.Linear(N2, 64)

    def forward(self, X):
        X = torch.relu(self.l1(X))
        X = torch.relu(self.l2(X))
        X = self.l3(X)
        return X


class NN_single(nn.Module):
    """Single-hidden-layer MLP -- input -> N1 -> 64. Tests whether the middle
    layer in NN is doing useful work or just acting as a pass-through."""
    def __init__(self, input_shape, N1):
        super().__init__()
        self.l1 = nn.Linear(input_shape, N1)
        # name the output l3 so dump_weights / nn_policy.c keep using a
        # consistent naming convention (even though there's no l2 here).
        self.l3 = nn.Linear(N1, 64)

    def forward(self, X):
        X = torch.relu(self.l1(X))
        X = self.l3(X)
        return X


class NN_no_valid(nn.Module):
    """NN that ignores the valid-moves input plane. Inspect-policy saliency
    showed near-zero gradient on the valid plane under loss_masked_to_valid,
    so the network shouldn't need it as input -- legality is enforced at
    argmax time by C inference. Dropping it shrinks L1 by 33% (real RAM on
    Playdate). loss_masked_to_valid still receives the valid plane via the
    batch slice [:, 0:64] in the trainer."""
    def __init__(self, input_shape, N1, N2):
        super().__init__()
        # input_shape ignored: nn_format_input still produces 192 bytes
        # (valid + my + opp) but we slice off the valid plane in forward.
        del input_shape
        self.l1 = nn.Linear(128, N1)
        self.l2 = nn.Linear(N1, N2)
        self.l3 = nn.Linear(N2, 64)

    def forward(self, X):
        X = X[:, 64:]  # drop valid-moves plane; keep my + opp (+ lookback if any)
        X = torch.relu(self.l1(X))
        X = torch.relu(self.l2(X))
        X = self.l3(X)
        return X

class NN_with_norm(nn.Module):
    def __init__(self, input_shape, N1, N2):
        super().__init__()
        # player, valid moves, board state
        self.l1 = nn.Linear(input_shape, N1)
        self.l1_norm = nn.BatchNorm1d(N1)
        self.l2 = nn.Linear(N1, N2)
        self.l2_norm = nn.BatchNorm1d(N2)
        self.l3 = nn.Linear(N2, 64)

        # NOTE: pretty sure including the current player as input is not very
        # helpful

    def forward(self, X):
        X = torch.relu(self.l1_norm(self.l1(X)))
        X = torch.relu(self.l2_norm(self.l2(X)))
        X = self.l3(X)
        return X

class CNN(nn.Module):
    def __init__(self, input_shape, N1, N2):
        super().__init__()
        # HACK: the input we're provided includes an extra
        # value for current player. We are going to ignore that

        # input is input_channels worth of 8x8 boards
        n_boards = (input_shape - 1)

        # we will ignore that the first input is the valid inputs layer
        # and just treat each 8x8 input as a separate channel
        n_channels = n_boards // 64

        # input: (n_channel, 8, 8)
        # output: (N1, 8, 8)
        self.l1 = nn.Conv2d(n_channels, N1, kernel_size=(3,3), padding=1)
        self.l1_norm = nn.BatchNorm2d(N1)

        # input: (N1, 8, 8)
        # output: (N2, 8, 8)
        self.l2 = nn.Conv2d(N1, N2, kernel_size=(3,3), padding=1)
        self.l2_norm = nn.BatchNorm2d(N2)

        # flattened, we'll have N2*8*8 inputs
        # then run through linear layer to select the place to play

        self.l3 = nn.Linear(N2*8*8, 64)

        # save N2 for reshaping
        self.N2 = N2

    def forward(self, X):
        # HACK: chop off the first entry of every input in the batch
        # we're removing the "whose turn is it" input
        X = X[:,1:]

        batch_size = X.shape[0]
        n_channels = X.shape[1] // 64
        X = torch.reshape(X, (batch_size, n_channels, 8, 8) )

        X = torch.relu(self.l1_norm(self.l1(X)))
        X = torch.relu(self.l2_norm(self.l2(X)))

        # flatten out and apply a final linear layer
        # assert X.shape[0] == batch_size # slow
        X = torch.reshape( X, (batch_size, self.N2*8*8) )
        X = self.l3(X)

        return X

# def loss_with_invalid(y_hat, batch_y, valid):
#     # gets lower as we select better
#     ce = torch.nn.functional.cross_entropy(y_hat, batch_y)

#     # needs to get lower as we select better
#     #
#     # outputs like this should be penalized:
#     # valid: [1 0 1 0 0 0 1 ...]
#     # output: [0 1 0 0 ...]
#     #
#     # but this should be rewarded
#     # valid: [1 0 1 0 0 0 1 ...]
#     # output: [1 0 0 0 ...]

#     # this doens't work
#     invalid = -1 * valid + 1.0
#     wrong = torch.mean(invalid * y_hat)
#     return ce+wrong

def loss_without_invalid(y_hat, batch_y, _valid):
    # NOTE: per-row one-hot targets from data_gen_policy.c. Duplicate boards
    # are not pre-aggregated into soft policy distributions; cross-entropy
    # with N one-hot rows is equivalent (up to a 1/N scalar) to one row
    # with the empirical soft target, so the network still learns the right
    # distribution. Pre-deduping with sample weights would reduce per-epoch
    # compute -- see the matching note in apps/data_gen_policy.c.
    ce = torch.nn.functional.cross_entropy(y_hat, batch_y)
    return ce


def loss_masked_to_valid(y_hat, batch_y, valid):
    # Cross-entropy restricted to legal moves: softmax denominator only sums
    # over valid squares so the network competes among legal moves instead of
    # against illegal ones. Without this, the easy way to lower CE is to copy
    # the valid-moves input plane into the output and the policy never has to
    # learn from board state. valid is (B, 64) float 0/1 from the input batch.
    masked = y_hat.masked_fill(valid == 0, -1e9)
    return torch.nn.functional.cross_entropy(masked, batch_y)

def train_step(model, optimizer, batch_X, batch_y):
    optimizer.zero_grad()
    y_hat = model.forward(batch_X)
    #loss = criterion(y_hat, batch_y, valid)
    loss = loss_without_invalid(y_hat, batch_y, None) # ignoring config

    loss.backward()
    optimizer.step()

# NotImplementedError: could not find kernel for aten._foreach_maximum_.List at dispatch key DispatchKey.Meta
# train_step = torch.compile(train_step) # cannot compile this, something about max not being supported?
# just compiling the model isn't really interesting at all but I guess that's all we're gonna get

def trainer(t, model, criterion, optimizer, X_train, y_train, X_test, y_test, checkpoint_dir, epochs=5, start_epoch=0, patience=8, profile_dir=None):
    # Track best test accuracy and bail out if it stops improving for
    # `patience` epochs. best.pt is what dump_weights.py picks up via LATEST.
    best_test_acc            = -1.0
    best_epoch               = -1
    epochs_since_improvement = 0

    # Optional torch.profiler capture. Schedule = 2 wait + 3 warmup + 10 active,
    # so we get a Chrome trace of 10 representative training steps after the
    # first few have warmed up the loaders/compile/cudnn-autotune.
    prof = None
    if profile_dir is not None:
        os.makedirs(profile_dir, exist_ok=True)
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=2, warmup=3, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
            record_shapes=True,
        )
        prof.start()
        print(f"[profile] capturing to {profile_dir}")

    for epoch in range(start_epoch, start_epoch+epochs):
        start_ts = time.time()

        # Accumulate loss as a tensor on the device. .item() forces a
        # GPU->CPU sync that would otherwise bottleneck the per-batch loop;
        # we sync once per epoch at the end instead.
        losses_t = torch.zeros((), device=device)
        train_total = 0

        with t.start("train"):
            for batch_X, batch_y in zip(X_train, y_train):
                # Data already lives on device thanks to the loader setup.
                # Valid-moves plane is now at [0:64] (canonicalized input,
                # no leading player byte).
                valid = batch_X[:, 0:64]

                optimizer.zero_grad()
                y_hat = model.forward(batch_X)
                loss = criterion(y_hat, batch_y, valid)

                loss.backward()
                optimizer.step()

                losses_t = losses_t + loss.detach()
                train_total += batch_y.shape[0]

                if prof is not None:
                    prof.step()

        losses = losses_t.item()

        train_end  = time.time()
        test_start = time.time()

        test_correct = 0
        test_total = 0
        with t.start("test"):
            with torch.no_grad(): # GPU run is much faster when grad is disabled
                for batch_X, batch_y in zip(X_test, y_test):
                    # Data already on device.
                    valid = batch_X[:, 0:64]
                    y_hat = model.forward(batch_X)
                    # Argmax over legal moves only, matching C inference. Without
                    # the mask, networks trained with loss_masked_to_valid look
                    # spuriously worse because illegal logits aren't suppressed.
                    y_hat = y_hat.masked_fill(valid == 0, -1e9)
                    test_correct += torch.sum(batch_y == torch.argmax(y_hat, dim=1))
                    test_total += batch_y.shape[0]

        stop_ts = time.time()

        data = {"loss": losses/train_total,
                "test": test_correct/test_total,
                "elapsed": stop_ts - start_ts,
                "train_elapsed": train_end - start_ts,
                "test_elapsed": stop_ts - train_end}

        # Periodic checkpoint every 16 epochs (was every 64 from the cluster
        # days; smaller granularity for fast local runs).
        if epoch % 16 == 0:
            path = f'{checkpoint_dir}/checkpoint_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': (model._orig_mod.state_dict()
                                     if hasattr(model, '_orig_mod')
                                     else model.state_dict()),
                'optimizer_state_dict': optimizer.state_dict(),
                }, path)

        # Early-stopping checkpoint: save whenever test accuracy hits a new
        # high. LATEST points here so dump_weights.py picks up the best model
        # rather than the final-epoch one (which may be past the peak).
        epoch_test_acc = (test_correct / test_total).item() if hasattr(test_correct, 'item') else (test_correct / test_total)
        if epoch_test_acc > best_test_acc:
            best_test_acc            = epoch_test_acc
            best_epoch               = epoch
            epochs_since_improvement = 0
            best_path = f'{checkpoint_dir}/best.pt'
            torch.save({
                'epoch': epoch,
                'test_acc': epoch_test_acc,
                'model_state_dict': (model._orig_mod.state_dict()
                                     if hasattr(model, '_orig_mod')
                                     else model.state_dict()),
                'optimizer_state_dict': optimizer.state_dict(),
                }, best_path)
            with open(f'{checkpoint_dir}/LATEST', "w") as f:
                f.write(best_path)
        else:
            epochs_since_improvement += 1

        wandb.log(data)
        print(f"epoch: {epoch+1}, {data}")

        if epochs_since_improvement >= patience:
            print(f"early stop: no test-accuracy improvement for {patience} epochs "
                  f"(best {best_test_acc:.4f} at epoch {best_epoch+1})")
            break

    # Always save the final epoch alongside best.pt (LATEST stays pointing
    # at best). Useful for inspecting overfitting trajectory or resuming.
    final_path = f'{checkpoint_dir}/final.pt'
    torch.save({
        'epoch': start_epoch + epochs - 1,
        'model_state_dict': (model._orig_mod.state_dict()
                                     if hasattr(model, '_orig_mod')
                                     else model.state_dict()),
        'optimizer_state_dict': optimizer.state_dict(),
        }, final_path)
    print(f"\nbest test accuracy: {best_test_acc:.4f} at epoch {best_epoch+1}")
    print(f"LATEST -> {checkpoint_dir}/best.pt")

    if prof is not None:
        prof.stop()
        print(f"[profile] trace written to {profile_dir}")

if __name__ == "__main__":
    c = open_config(sys.argv[1])
    t = Tracer(c["files"]["trace_file"], enabled=False)

    batch_size      = c["settings"]["batch_size"]
    boards_dir      = c["files"]["boards_dir"]
    checkpoint_dir  = c["files"]["checkpoint_dir"]
    input_shape     = c["settings"]["model_params"]["input_shape"]
    boards_per_file = c["settings"]["boards_per_file"]

    # have 4 cores, but using 2 of them for IO threads
    torch.set_num_threads(2)

    wandb.init(project='othello', name=c["name"], config=c["settings"])

    if torch.cuda.is_available():
        print("using cuda")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("using mps")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    with t.start('load_split_file'):
        with lz4.frame.open(c["files"]["split_filename"], mode='rb') as f:
            # weights_only=False because the split file contains numpy arrays
            # alongside torch tensors; we're loading our own file so the
            # arbitrary-code-execution risk is moot. (Could be cleaned up by
            # converting numpy arrays to torch tensors in make_split.py.)
            split = torch.load(f, weights_only=False)
            train_policy = split['train_policy']
            test_policy = split['test_policy']

            train_indices = split['train_idx']
            test_indices  = split['test_idx']

    print(f'{train_policy.shape} entries in train_policy')
    print(f'{test_policy.shape} entries in test_policy')

    # Move policy targets to device once. Cheap and avoids the per-batch
    # host->device copy that the SimpleDataLoader would otherwise need.
    train_policy = train_policy.to(device)
    test_policy = test_policy.to(device)

    with t.start('create_loaders'):
        board_dataloader = BoardDirLoader('train', boards_dir, boards_per_file, input_shape, train_indices, batch_size, t, device=device)
        policy_dataloader = SimpleDataLoader(train_policy, batch_size)

        board_dataloader_test = BoardDirLoader('test', boards_dir, boards_per_file, input_shape, test_indices, batch_size, t, device=device)
        policy_dataloader_test = SimpleDataLoader(test_policy, batch_size)

    # load the model and loss function from the config file
    with t.start('create model'):
        net  = getattr(sys.modules[__name__], c["settings"]["model_name"])(**c["settings"]["model_params"])
        loss = eval(c["settings"]["loss_variant"])

    net.to(device)
    net = torch.compile(net)
    wandb.watch(net, log_freq=5, log='all')

    # fused=True dispatches a single CUDA kernel for the param update across
    # all tensors (multi-tensor apply). Only on CUDA; MPS/CPU need the default.
    use_fused = torch.cuda.is_available()
    optim = torch.optim.Adam(net.parameters(), lr=0.001, amsgrad=True,
                             weight_decay=float(c["settings"]["weight_decay"]),
                             fused=use_fused)

    # load checkpoint
    try:
        with open(checkpoint_dir + "/LATEST", "r") as f:
            latest_fname = f.readline()
            print(f"Reading {latest_fname} as checkpoint file")
            cp = torch.load(latest_fname)

            net.load_state_dict(cp['model_state_dict'])
            optim.load_state_dict(cp['optimizer_state_dict'])

            start_epoch = cp['epoch']
    except FileNotFoundError:
        print('no checkpoints found')
        start_epoch = 0

    # When config.settings.profile is true, torch.profiler captures a short
    # trace to <experiment_dir>/profile/ for one-off perf investigations.
    profile_dir = None
    if c["settings"].get("profile", False):
        profile_dir = os.path.join(c["experiment_dir"], "profile")

    epochs = c["settings"]["train_epochs"]
    print(f'starting training (epochs={epochs}, profile={"on" if profile_dir else "off"})')
    trainer(t, net, loss, optim, board_dataloader, policy_dataloader, board_dataloader_test, policy_dataloader_test, checkpoint_dir, epochs=epochs, start_epoch=start_epoch, profile_dir=profile_dir)

    t.close()
