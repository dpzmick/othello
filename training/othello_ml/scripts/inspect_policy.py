"""Visualize what the policy net actually predicts on real board positions.

Run:
    uv run python -m othello_ml.scripts.inspect_policy <experiment_dir>

Writes per-position PNGs into <experiment_dir>/inspect/policy/.

For each curated position we show:
  - The board (my=●, opp=○, valid moves=✕)
  - Policy logits over the 64 squares as a heatmap (with the argmax starred)
  - Saliency maps: gradient of the chosen-move logit w.r.t. each input plane
    (valid plane, my plane, opp plane), reshaped to 8x8. Shows which input
    squares the net leaned on for the prediction.

The input layout matches libcomputer/nn_policy.c::nn_format_input exactly:
  [0:64)    valid-move plane
  [64:128)  my-pieces plane   (whoever is to move)
  [128:192) opp-pieces plane
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import toml
import torch

import othello_ml.scripts.train as train_module


def compute_valid_moves(my: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """8x8 mask of valid moves for the player whose pieces are `my`."""
    valid = np.zeros((8, 8), dtype=np.int8)
    empty = (my + opp == 0)
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for y in range(8):
        for x in range(8):
            if not empty[y, x]:
                continue
            for dy, dx in dirs:
                cy, cx = y + dy, x + dx
                if not (0 <= cx < 8 and 0 <= cy < 8):
                    continue
                if not opp[cy, cx]:
                    continue
                cy += dy
                cx += dx
                while 0 <= cx < 8 and 0 <= cy < 8:
                    if my[cy, cx]:
                        valid[y, x] = 1
                        break
                    if not opp[cy, cx]:
                        break
                    cy += dy
                    cx += dx
                if valid[y, x]:
                    break
    return valid


def apply_move(my: np.ndarray, opp: np.ndarray, x: int, y: int):
    """Return (new_my, new_opp) after `my` plays at (x, y). Assumes legal."""
    my = my.copy()
    opp = opp.copy()
    my[y, x] = 1
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for dy, dx in dirs:
        flips = []
        cy, cx = y + dy, x + dx
        while 0 <= cx < 8 and 0 <= cy < 8 and opp[cy, cx]:
            flips.append((cy, cx))
            cy += dy
            cx += dx
        if 0 <= cx < 8 and 0 <= cy < 8 and my[cy, cx]:
            for (fy, fx) in flips:
                my[fy, fx] = 1
                opp[fy, fx] = 0
    return my, opp


def build_input(my: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """192-d input vector matching nn_format_input."""
    valid = compute_valid_moves(my, opp).astype(np.float32)
    return np.concatenate([valid.flatten(), my.astype(np.float32).flatten(),
                           opp.astype(np.float32).flatten()])


def load_model(experiment_dir: str):
    cfg = toml.load(os.path.join(experiment_dir, "config.toml"))
    settings = cfg["settings"]
    model_name = settings["model_name"]
    if model_name not in ("NN", "NN_single"):
        raise RuntimeError(f"unsupported model_name={model_name!r}")
    Model = getattr(train_module, model_name)
    net = Model(**settings["model_params"])

    latest_path = os.path.join(experiment_dir, "checkpoints", "LATEST")
    if os.path.exists(latest_path):
        with open(latest_path) as f:
            ckpt_path = f.read().strip()
    else:
        # Modal runs only ship best.pt; LATEST may be absent.
        ckpt_path = os.path.join(experiment_dir, "checkpoints", "best.pt")
    cp = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    state = cp["model_state_dict"]
    # torch.compile prefixes keys with "_orig_mod."; strip if present.
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k[len("_orig_mod."):]: v for k, v in state.items()}
    net.load_state_dict(state)
    net.eval()
    return net, model_name, cp.get("epoch", -1)


def plot_position(net, my: np.ndarray, opp: np.ndarray, title: str, save_path: str):
    inp_np = build_input(my, opp)
    x = torch.from_numpy(inp_np).unsqueeze(0).requires_grad_(True)
    logits = net(x).squeeze(0)  # (64,)

    valid = inp_np[0:64].reshape(8, 8)
    n_valid = int(valid.sum())
    if n_valid == 0:
        print(f"  {title}: no valid moves, skipping")
        return

    # Pick the argmax over valid squares as the "chosen" move for saliency.
    masked = logits.detach().numpy().copy()
    masked[valid.flatten() == 0] = -np.inf
    chosen_idx = int(np.argmax(masked))
    chosen_x, chosen_y = chosen_idx % 8, chosen_idx // 8

    net.zero_grad()
    logits[chosen_idx].backward()
    sal = x.grad.squeeze(0).numpy()
    sal_valid = sal[0:64].reshape(8, 8)
    sal_my    = sal[64:128].reshape(8, 8)
    sal_opp   = sal[128:192].reshape(8, 8)

    logits_2d = logits.detach().numpy().reshape(8, 8)
    logits_show = np.where(valid > 0, logits_2d, np.nan)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"{title}  (chose ({chosen_x},{chosen_y}))", fontsize=14)

    # Board
    ax = axes[0, 0]
    ax.set_title("board")
    ax.set_xlim(-0.5, 7.5); ax.set_ylim(7.5, -0.5)
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    for yy in range(8):
        for xx in range(8):
            if my[yy, xx]:
                ax.plot(xx, yy, "o", markersize=22, color="black", mec="black")
            elif opp[yy, xx]:
                ax.plot(xx, yy, "o", markersize=22, color="white", mec="black")
            elif valid[yy, xx]:
                ax.plot(xx, yy, "x", markersize=14, color="red", mew=2)
    ax.plot(chosen_x, chosen_y, "*", markersize=28, color="gold", mec="black")

    # Logits
    ax = axes[0, 1]
    im = ax.imshow(logits_show, cmap="viridis")
    ax.set_title(f"policy logits ({n_valid} valid)")
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.plot(chosen_x, chosen_y, "*", markersize=20, color="red", mec="white")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # Softmax (valid only)
    ax = axes[0, 2]
    valid_mask = valid > 0
    masked_log = np.where(valid_mask, logits_2d, -1e9)
    e = np.exp(masked_log - np.max(masked_log[valid_mask]))
    e = np.where(valid_mask, e, 0)
    probs = e / e.sum()
    probs_show = np.where(valid_mask, probs, np.nan)
    im = ax.imshow(probs_show, cmap="viridis", vmin=0)
    ax.set_title("softmax over valid moves")
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    fig.colorbar(im, ax=ax, fraction=0.046)

    # Summed |saliency|
    ax = axes[0, 3]
    sal_total = np.abs(sal_valid) + np.abs(sal_my) + np.abs(sal_opp)
    im = ax.imshow(sal_total, cmap="hot")
    ax.set_title("|saliency| summed over planes")
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    fig.colorbar(im, ax=ax, fraction=0.046)

    # Per-plane saliency
    vmax = max(np.abs(sal_valid).max(), np.abs(sal_my).max(), np.abs(sal_opp).max())
    for ax, plane, label in [
        (axes[1, 0], sal_valid, "saliency: valid plane"),
        (axes[1, 1], sal_my,    "saliency: my plane"),
        (axes[1, 2], sal_opp,   "saliency: opp plane"),
    ]:
        im = ax.imshow(plane, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(label)
        ax.set_xticks(range(8)); ax.set_yticks(range(8))
        fig.colorbar(im, ax=ax, fraction=0.046)

    # Per-plane saliency at the chosen square highlighted on opp plane,
    # for a quick "what's around the chosen move" view.
    ax = axes[1, 3]
    ax.set_title("saliency vs board overlay")
    ax.set_xlim(-0.5, 7.5); ax.set_ylim(7.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.grid(True, alpha=0.3)
    im = ax.imshow(sal_total, cmap="hot", alpha=0.85,
                   extent=(-0.5, 7.5, 7.5, -0.5))
    for yy in range(8):
        for xx in range(8):
            if my[yy, xx]:
                ax.plot(xx, yy, "o", markersize=14, color="black", mec="white")
            elif opp[yy, xx]:
                ax.plot(xx, yy, "o", markersize=14, color="white", mec="black")
    ax.plot(chosen_x, chosen_y, "*", markersize=22, color="lime", mec="black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  wrote {save_path}")


def starting_position():
    """Standard Othello starting layout, Black to move. 'my' = black."""
    my  = np.zeros((8, 8), dtype=np.int8)
    opp = np.zeros((8, 8), dtype=np.int8)
    # Engine convention (libothello/othello.c:31-32):
    #   white at (3,3),(4,4); black at (3,4),(4,3); black to move.
    my[4, 3] = 1; my[3, 4] = 1       # black
    opp[3, 3] = 1; opp[4, 4] = 1     # white
    return my, opp


def play_moves(my, opp, moves):
    """Apply a sequence of (x,y) moves, alternating side-to-move each step."""
    for (mx, my_) in moves:
        my, opp = apply_move(my, opp, mx, my_)
        my, opp = opp, my  # swap perspective: now the other player is "my"
    return my, opp


def curated_positions():
    """List of (name, my, opp) test positions for visualization."""
    out = []

    # 1. Opening: just the start.
    my, opp = starting_position()
    out.append(("01_start_black_to_move", my, opp))

    # 2. After a couple of standard opening moves.
    my, opp = starting_position()
    my, opp = play_moves(my, opp, [(3, 2), (2, 2), (3, 1)])
    out.append(("02_after_3_moves", my, opp))

    # 3. A corner-pressure midgame. Construct a position where opp threatens
    #    a corner and my has options. Built by playing a short sequence.
    my, opp = starting_position()
    my, opp = play_moves(my, opp, [
        (3, 2), (2, 4), (5, 4), (5, 3),
        (2, 3), (4, 2), (5, 2), (2, 5),
        (1, 4), (4, 5),
    ])
    out.append(("03_midgame_corner_pressure", my, opp))

    return out


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <experiment_dir>", file=sys.stderr)
        sys.exit(1)
    experiment_dir = sys.argv[1].rstrip("/")

    net, model_name, epoch = load_model(experiment_dir)
    print(f"loaded {model_name} from {experiment_dir} (epoch {epoch})")

    out_dir = os.path.join(experiment_dir, "inspect", "policy")
    os.makedirs(out_dir, exist_ok=True)

    for name, my, opp in curated_positions():
        title = f"{os.path.basename(experiment_dir)} / {name}"
        save_path = os.path.join(out_dir, f"{name}.png")
        plot_position(net, my, opp, title, save_path)


if __name__ == "__main__":
    main()
