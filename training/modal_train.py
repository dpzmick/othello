"""Run the Othello training pipeline on Modal.

Pipeline: data_gen_policy (CPU) -> make_split (CPU) -> train (A100) ->
dump_weights (CPU). All steps share a Modal Volume so we don't re-upload
the dataset between runs.

Setup (do once):
    uv run modal token new
    uv run modal secret create wandb-secret WANDB_API_KEY=<your key>

Run a fresh experiment:
    uv run --project training modal run training/modal_train.py

Skip data prep (re-train on existing data):
    uv run --project training modal run training/modal_train.py --run-name <existing>

Fetch trained weights back:
    uv run modal volume get othello-data <run-name>_small_full/weights_policy.c libcomputer/weights_policy.c
    cmake --build build -j && ./build/apps/thunderdome

Clean up old runs (we don't auto-prune):
    uv run modal volume ls othello-data
    uv run modal volume rm othello-data <run-name>_small_full --recursive
"""

import os
import secrets
import time

import modal

# Repo root sits one level above this file (training/modal_train.py).
# add_local_dir below uses this so the upload is invariant to cwd.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

APP_NAME    = "othello-train"
VOLUME_NAME = "othello-data"


def make_run_name() -> str:
    """Memorable, sortable, unique. timestamp + 4 hex chars."""
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(2)}"


app    = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Image: Debian + C toolchain for data_gen_policy + Python deps for training.
# `copy_local_dir` happens at image-bake time so the C build can run during
# image construction; cached layer is reused when source is unchanged.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "cmake",
        "build-essential",
        "pkg-config",
        "libzstd-dev",
    )
    # Explicit pip deps rather than from pyproject -- some Modal versions
    # have shifted the pyproject-loading API; this works on all of them.
    .pip_install(
        "numpy>=1.26",
        "torch>=2.2",
        "wandb>=0.16",
        "pyzstd>=0.15",
        "lz4>=4.3",
        "toml>=0.10",
        "tqdm>=4.66",
    )
    # Bake the source into the image so the C build can happen at image
    # build time and the binaries persist across function invocations.
    # `copy=True` evaluates at image-bake time (vs the runtime overlay you
    # get with copy=False / the default).
    .add_local_dir(
        REPO_ROOT,
        remote_path="/workspace",
        copy=True,
        ignore=[
            ".git",
            ".direnv",
            ".venv",
            "build",
            "emcc-build",
            "experiments",
            "wandb",
            "rip",
            "ml",
            "attic",
            "training/.venv",
            "training/__pycache__",
            "web_game/node_modules",
            "web_game/dist",
        ],
    )
    .run_commands(
        "cd /workspace && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release",
        "cd /workspace && cmake --build build -j --target data_gen_policy",
    )
)


def _build_config(run_name: str, model_style: str = "small", weight_decay: float = 1e-4,
                  train_epochs: int = 64, profile: bool = False):
    """Write an experiment config inside the container.

    Sits beside data_gen_policy's expectations; matches make_local_config.py
    but with paths pointing into the Modal Volume. `model_style` picks the
    architecture (see othello_ml/config.py's model_styles dict).
    """
    import sys
    sys.path.insert(0, "/workspace/training")
    from othello_ml.config import make_config, setup

    config = make_config(
        experiment_root="/data",
        experiment_name=run_name,
        wthor_dir="/workspace/wthor_files",
        debug=False,          # all 46 WTHOR files
        include_flips=True,   # 4-fold rotation augmentation
        model_style=model_style,
        board_lookback=0,
        train_epochs=train_epochs,
        batch_size=4096,
        weight_decay=weight_decay,
        profile=profile,
    )
    setup(config)
    return config


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=4.0,
    timeout=900,
)
def prepare_data(run_name: str, model_style: str = "small", weight_decay: float = 1e-4,
                 train_epochs: int = 64, profile: bool = False):
    """Generate the policy dataset and the train/test split.

    Idempotent: if split.pt.lz4 already exists for this run, we skip both
    steps. Run name is suffixed by make_config (e.g. ``..._small_full``),
    so the on-disk dir matches what the trainer expects.
    """
    import subprocess
    from pathlib import Path

    config = _build_config(run_name, model_style=model_style, weight_decay=weight_decay,
                           train_epochs=train_epochs, profile=profile)
    experiment_dir = Path(config["experiment_dir"])
    split_file     = Path(config["files"]["split_filename"])

    if split_file.exists():
        print(f"[prepare_data] {split_file} already exists, skipping")
        return config["name"]

    print(f"[prepare_data] generating dataset for {experiment_dir}")
    subprocess.run(
        [
            "/workspace/build/apps/data_gen_policy",
            f"{experiment_dir}/config.toml",
        ],
        check=True,
    )

    print(f"[prepare_data] writing train/test split")
    subprocess.run(
        [
            "python",
            "-m",
            "othello_ml.scripts.make_split",
            str(experiment_dir),
        ],
        cwd="/workspace/training",
        check=True,
    )

    volume.commit()
    return config["name"]


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="L4",
    timeout=7200,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(run_name: str, model_style: str = "small", weight_decay: float = 1e-4,
          train_epochs: int = 64, profile: bool = False):
    """Train the policy MLP on A100. wandb runs in online mode via the secret."""
    import subprocess

    config = _build_config(run_name, model_style=model_style, weight_decay=weight_decay,
                           train_epochs=train_epochs, profile=profile)
    experiment_dir = config["experiment_dir"]

    # No WANDB_MODE => wandb defaults to online, using WANDB_API_KEY from the
    # secret. The run name in wandb matches config["name"].
    subprocess.run(
        [
            "python",
            "-m",
            "othello_ml.scripts.train",
            experiment_dir,
        ],
        cwd="/workspace/training",
        check=True,
    )

    volume.commit()
    return config["name"]


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=2.0,
    timeout=300,
)
def dump_weights(run_name: str, model_style: str = "small", weight_decay: float = 1e-4,
                 train_epochs: int = 64, profile: bool = False):
    """Export the trained .pt to weights_policy.c on the volume."""
    import shutil
    import subprocess
    from pathlib import Path

    config = _build_config(run_name, model_style=model_style, weight_decay=weight_decay,
                           train_epochs=train_epochs, profile=profile)
    experiment_dir = Path(config["experiment_dir"])

    subprocess.run(
        [
            "python",
            "-m",
            "othello_ml.scripts.dump_weights",
            str(experiment_dir),
        ],
        cwd="/workspace/training",
        check=True,
    )

    # dump_weights.py writes to /workspace/libcomputer/weights_policy.c.
    # Stash a copy in the run's volume dir so the user can fetch it.
    src = Path("/workspace/libcomputer/weights_policy.c")
    dst = experiment_dir / "weights_policy.c"
    shutil.copy(src, dst)
    print(f"[dump_weights] wrote {dst}")
    volume.commit()
    return str(dst)


@app.local_entrypoint()
def main(run_name: str = "", model_style: str = "small", skip_prepare: bool = False,
         weight_decay: float = 1e-4, train_epochs: int = 64, profile: bool = False):
    """Run the full pipeline. Random run name unless one is supplied."""
    if not run_name:
        run_name = make_run_name()
    print(f"=== Modal run: {run_name} (model_style={model_style}, wd={weight_decay}, "
          f"epochs={train_epochs}, profile={profile}) ===")

    # Match make_config's naming exactly so console output matches the volume.
    from othello_ml.config import make_run_dir_name, LOSS_TAGS  # noqa: F401
    actual = make_run_dir_name(run_name, model_style, "loss_masked_to_valid", debug=False)

    if not skip_prepare:
        prepare_data.remote(run_name, model_style, weight_decay, train_epochs, profile)
    print(f"[main] experiment name (on volume): {actual}")

    train.remote(run_name, model_style, weight_decay, train_epochs, profile)
    dump_weights.remote(run_name, model_style, weight_decay, train_epochs, profile)

    print("\n=== Done ===")
    print(f"Fetch locally:")
    print(f"  uv run modal volume get {VOLUME_NAME} {actual}/weights_policy.c libcomputer/weights_policy.c")
    print(f"  cmake --build build -j && ./build/apps/thunderdome")


@app.function(image=image, cpu=0.5, timeout=14400)
def sweep_orchestrator(style_list: list[str], base_name: str, weight_decay: float = 1e-4):
    """Runs on Modal. Owns the parallel child runs across model styles so
    they survive a laptop disconnect under `modal run --detach`.

    Each style's prepare_data, train, and dump_weights run in their own
    containers; this orchestrator just spawns them and waits. Per-style
    failures are caught so one bad style (e.g. a train timeout) doesn't
    kill the rest of the sweep."""
    print(f"=== Sweep: {base_name} across {style_list} (wd={weight_decay}) ===")

    handles = []
    for style in style_list:
        run_name = base_name   # make_config builds the full dir name
        actual   = _build_config(run_name, model_style=style, weight_decay=weight_decay)["name"]
        print(f"[sweep] spawning prepare_data for {style}")
        pd_handle = prepare_data.spawn(run_name, style, weight_decay)
        handles.append((style, run_name, actual, pd_handle))

    train_handles = []
    for style, run_name, actual, pd_h in handles:
        try:
            pd_h.get()
        except Exception as e:
            print(f"[sweep] prepare_data FAILED for {style}: {e}")
            continue
        t_h = train.spawn(run_name, style, weight_decay)
        train_handles.append((style, run_name, actual, t_h))
        print(f"[sweep] training spawned: {style}")

    dump_handles = []
    for style, run_name, actual, t_h in train_handles:
        try:
            t_h.get()
        except Exception as e:
            print(f"[sweep] train FAILED for {style}: {e}")
            continue
        d_h = dump_weights.spawn(run_name, style, weight_decay)
        dump_handles.append((style, actual, d_h))
        print(f"[sweep] dump spawned: {style}")

    for style, actual, d_h in dump_handles:
        try:
            d_h.get()
            print(f"[sweep] done: {style} -> {actual}")
        except Exception as e:
            print(f"[sweep] dump_weights FAILED for {style}: {e}")

    print("\n=== Sweep complete ===")


@app.local_entrypoint()
def sweep(styles: str = "small,medium,large", weight_decay: float = 1e-4):
    """Kick off a sweep across model sizes.

    The orchestration runs ON Modal (not on your laptop), so the whole
    sweep survives a `--detach` + laptop sleep.

    Detached overnight run:

        uv run --project training modal run --detach training/modal_train.py::sweep

    Custom styles:

        uv run --project training modal run --detach training/modal_train.py::sweep \\
            --styles "tiny,single_256,single_1024,single_2048,medium,large"

    Available MLP styles (see othello_ml/config.py model_styles dict):
        tiny, single_256, single_1024, single_2048,
        small, small_norm, medium, medium_norm, large, large_norm
    (CNN styles will fail until C inference supports them.)
    """
    style_list = [s.strip() for s in styles.split(",")]
    base_name  = make_run_name()
    print(f"Triggering sweep_orchestrator for {style_list} (base={base_name}, wd={weight_decay})")
    # .spawn() is fire-and-forget -- submits the orchestrator to Modal and
    # returns immediately. Survives local CLI disconnect. .remote() would
    # block here and risk cancellation on detach.
    handle = sweep_orchestrator.spawn(style_list, base_name, weight_decay)
    print(f"Spawned: {handle.object_id}")
    print("\nDone. The orchestrator is running independently on Modal.")
    print("Watch progress at:")
    print("  https://modal.com/apps/dpzmick")
    print("  https://wandb.ai/dpzmick/othello")
