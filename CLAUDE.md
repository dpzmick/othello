# Project orientation

Othello AI targeting Playdate hardware (Cortex-M7, 168MHz, 16MB RAM). C engine
+ training pipeline. Currently focused on the policy network. Long-term plan
is net-as-policy-prior for MCTS rollouts.

For full context — what we've tried, results, why decisions were made — read
`JOURNAL.md`. **Always check the journal before suggesting experiments**, and
**add a new entry at the bottom after any non-trivial experiment** (new
architecture, new loss, new data treatment, eval methodology change). Keep
entries terse: date, what, why, what we learned. Don't journal trivial
cleanups or bug fixes; do journal anything that changes how a future
decision should be made.

## Layout

- `libothello/` — board, move legality, bitboard rules. Pure C, no deps.
- `libcomputer/` — MCTS, policy NN inference. `nn_policy.c` does the NN
  forward pass; it `#include`s `weights_policy.c` (compile-time weights
  baked in via `training/othello_ml/scripts/dump_weights.py`).
- `libcommon/` — hashing, error reporting.
- `apps/` — CLI binaries. `thunderdome.c` is the NN-vs-MCTS evaluator.
  `data_gen_policy.c` reads WTHOR files and emits training data with full
  D4 augmentation.
- `training/` — Python project root. `pyproject.toml` and `uv.lock` live
  here, plus the `othello_ml` package (`scripts/train.py` is the trainer,
  `config.py` defines run configurations and model styles) and
  `modal_train.py` (Modal-based remote training on L4; the sweep
  orchestrator runs on Modal so it survives laptop sleep).
- `experiments/<run_name>/` — one directory per trained model. Each has
  `config.toml`, `weights_policy.c`, `checkpoints/best.pt`. Naming:
  `<timestamp>_<hex>_<style>_<lossmask|lossall>_full`. Gitignored.
- `wthor_files/` — raw WTHOR `.wtb` games (input data).
- `scripts/` — dev shell tooling. `thunderdome_sweep.sh` builds + runs
  thunderdome across every experiment in parallel.
- `attic/` — older code/scripts kept for reference, no longer used.

## Commands

Everything that needs the C toolchain or Python deps lives behind
`nix develop`. Always wrap build/run commands accordingly.

Build:
```
nix develop --command bash -c "cmake -B build -S . && cmake --build build -j"
```

Run thunderdome on whatever weights are currently in `libcomputer/weights_policy.c`:
```
./build/apps/thunderdome
```

Sweep thunderdome across all trained NN-arch runs (builds one binary per run,
runs them in parallel on all cores):
```
./scripts/thunderdome_sweep.sh
```

Dump weights from a trained checkpoint into `libcomputer/weights_policy.c` and
into the experiment dir. `--project training` points uv at the python project
without changing cwd, so paths like `experiments/<run>` stay repo-relative:
```
nix develop --command bash -c "uv run --project training python -m othello_ml.scripts.dump_weights experiments/<run>"
```

Visualize a model's policy + saliency on curated positions
(writes to `<run>/inspect/policy/`):
```
nix develop --command bash -c "uv run --project training python -m othello_ml.scripts.inspect_policy experiments/<run>"
```

Launch a Modal sweep (detached so it survives laptop sleep):
```
nix develop --command bash -c 'uv run --project training modal run --detach training/modal_train.py::sweep --styles "tiny,small,medium,large"'
```

Fetch a finished Modal run to local + clean up the Modal volume:
```
# Download (config.toml, weights_policy.c, checkpoints/best.pt only — skip datasets/)
RUN=<run_name>
mkdir -p experiments/$RUN/checkpoints
nix develop --command bash -c "
  uv run --project training modal volume get othello-data '$RUN/config.toml' experiments/$RUN/config.toml &&
  uv run --project training modal volume get othello-data '$RUN/weights_policy.c' experiments/$RUN/weights_policy.c &&
  uv run --project training modal volume get othello-data '$RUN/checkpoints/best.pt' experiments/$RUN/checkpoints/best.pt
"

# Clean up
nix develop --command bash -c "uv run --project training modal volume rm othello-data '$RUN' --recursive"
```

## Conventions

- **All build/Python commands run inside `nix develop`.** The bare environment
  doesn't have cmake, uv, the right Python, etc.
- **Don't commit datasets or large run artifacts.** `experiments/<run>/`
  contains weights + config + best.pt only; intermediate checkpoints and
  datasets stay on Modal or get regenerated.
- **C inference path supports `NN` and `NN_single` only.** `CNN` runs train
  fine but `dump_weights.py` skips them (would need a new C generator).
- **Loss variant** is selected by name string in `config.toml`:
  `loss_masked_to_valid` (current default, masks illegal moves before
  softmax) or `loss_without_invalid` (legacy, softmax over all 64).
  See `training/othello_ml/scripts/train.py`.
- **Model style suffix** in run names: `_lossmask_full` vs `_lossall_full`.
  Lets you compare loss variants in the same `experiments/` directory.

## Things to avoid

- Don't suggest pre-training infrastructure or feature flags. Single
  developer, no production constraints.
- Don't optimize prematurely. Playdate inference cost matters; everything
  else (training time, dataset size, eval time) is cheap.
- Don't add comments restating what the code does. Comments here are for
  *why* — invariants, surprising decisions, gotchas. The existing code
  follows this style; match it.
- Don't recommend things from memory without verifying the current code.
  Particularly anything about architecture, loss, or data layout — those
  have all changed recently. Read the source.

## When the user runs an experiment

1. **Before**: skim `JOURNAL.md`'s open questions and recent entries to see
   if this has been tried, or if results from earlier inform the setup.
2. **After**: add a journal entry. Include the run name, what changed,
   what the result was (numerically), and what it means for next steps.
   If the result confirms or kills an open question, update the
   "Open questions" section.
