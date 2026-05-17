"""Quick look at what the trained policy MLP learned.

Run:
    uv run python -m othello_ml.scripts.inspect_weights <experiment_dir>

Writes a handful of PNGs into <experiment_dir>/inspect/.

What it shows:
  - L1 "receptive fields": for top-K hidden neurons (sorted by weight L2
    norm), reshape the 192 incoming weights into 3 separate 8x8 heatmaps
    (valid-moves plane, my-pieces plane, opp-pieces plane). Strong + or -
    weights on specific squares hint at what the neuron specialized in.
  - Per-layer weight distribution: histograms of l1, l2, l3 weight values.
    Tells you if weight decay actually shrunk the weights, and if any layer
    saturated.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import toml
import torch


def load_state(experiment_dir: str):
    config = toml.load(os.path.join(experiment_dir, "config.toml"))
    with open(os.path.join(experiment_dir, "checkpoints", "LATEST")) as f:
        ckpt_path = f.read().strip()
    cp = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    return config, cp, ckpt_path


def plot_receptive_fields(state, input_shape, out_dir, top_k=12):
    """For top_k hidden neurons (by L2 norm), plot their L1 input weights.

    Input layout from nn_format_input (canonicalized, 192 floats):
      [0:64)   valid-move plane
      [64:128) my-pieces plane
      [128:192) opp-pieces plane
    """
    assert input_shape == 192, (
        f"this script assumes the canonicalized 192-dim input layout; got {input_shape}"
    )

    w = state["l1.weight"].cpu().numpy()  # (1024, 192)
    norms = np.linalg.norm(w, axis=1)
    idxs = np.argsort(-norms)[:top_k]

    plane_names = ["valid moves", "my pieces", "opp pieces"]
    plane_slices = [slice(0, 64), slice(64, 128), slice(128, 192)]

    cols = 3
    rows = top_k
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    for row, neuron_idx in enumerate(idxs):
        neuron_weights = w[neuron_idx]
        vmax = float(np.abs(neuron_weights).max())
        for col, (name, sl) in enumerate(zip(plane_names, plane_slices)):
            plane = neuron_weights[sl].reshape(8, 8)
            ax = axes[row, col]
            ax.imshow(plane, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(name, fontsize=9)
            if col == 0:
                ax.set_ylabel(
                    f"n={neuron_idx}\n‖w‖={norms[neuron_idx]:.2f}",
                    fontsize=8,
                )

    fig.suptitle(f"L1 receptive fields (top {top_k} neurons by ‖w‖)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, "l1_receptive_fields.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


def plot_weight_distributions(state, out_dir):
    """Histogram of weight values per layer."""
    layer_keys = [k for k in ["l1.weight", "l2.weight", "l3.weight"] if k in state]
    layers = []
    for key in layer_keys:
        out_features, in_features = state[key].shape
        layers.append((key, f"{key.split('.')[0].upper()}: {in_features} -> {out_features}"))

    fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 3.5))
    if len(layers) == 1:
        axes = [axes]
    for ax, (key, title) in zip(axes, layers):
        flat = state[key].cpu().numpy().ravel()
        ax.hist(flat, bins=100, color="steelblue", alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.axvline(0.0, color="k", linestyle="--", linewidth=0.6)
        ax.set_yscale("log")
        ax.set_xlabel(
            f"weight\nmean={flat.mean():+.4f}  std={flat.std():.4f}\n"
            f"min={flat.min():+.4f}  max={flat.max():+.4f}",
            fontsize=8,
        )
    fig.suptitle("Per-layer weight distributions (log y)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "weight_distributions.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


def print_summary_stats(state, ckpt_path):
    print(f"checkpoint: {ckpt_path}")
    for key in ["l1.weight", "l1.bias", "l2.weight", "l2.bias", "l3.weight", "l3.bias"]:
        if key not in state:
            continue   # NN_single has no l2
        t = state[key]
        flat = t.cpu().numpy().ravel()
        shape_str = str(tuple(t.shape))
        print(
            f"  {key:14s} shape={shape_str:>15}  "
            f"mean={flat.mean():+.4f}  std={flat.std():.4f}  "
            f"|w|>0.1: {(np.abs(flat) > 0.1).mean():.2%}"
        )


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <experiment_dir>", file=sys.stderr)
        sys.exit(1)

    experiment_dir = sys.argv[1].rstrip("/")
    config, cp, ckpt_path = load_state(experiment_dir)

    settings = config["settings"]
    if settings["model_name"] not in ("NN", "NN_single"):
        print(f"only NN / NN_single supported here; got {settings['model_name']!r}", file=sys.stderr)
        sys.exit(1)

    state = cp["model_state_dict"]
    input_shape = settings["model_params"]["input_shape"]

    out_dir = os.path.join(experiment_dir, "inspect")
    os.makedirs(out_dir, exist_ok=True)

    print_summary_stats(state, ckpt_path)
    plot_receptive_fields(state, input_shape, out_dir)
    plot_weight_distributions(state, out_dir)
    print(f"\nOutputs: {out_dir}")


if __name__ == "__main__":
    main()
