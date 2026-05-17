# Research Journal

Running log of experiments, motivations, and findings. Newest entries at the
bottom. For day-to-day code or tool notes use the relevant source file; this
file is for *why we did it* and *what we learned*.

---

## Project goal

A policy network strong enough to drive MCTS rollouts on Playdate (Cortex-M7,
168MHz, 16MB RAM). Long-term plan: net as policy prior + MCTS at inference.
Right now we're benchmarking nets in isolation (NN vs vanilla MCTS) to compare
architectures and training recipes; we'll add net-guided MCTS evaluation when
the underlying MCTS gets reworked.

## Evaluation methodology

Thunderdome (`apps/thunderdome.c`) plays each NN against a 2000-rollout vanilla
MCTS over 1000 trials, alternating which color the NN plays each game to wash
out color-asymmetric strength. Reports overall winrate with a Wilson 95% CI
plus per-color split. CI half-width is ±0.03 at 1000 trials.

Sweep script: `thunderdome_sweep.sh` builds one binary per run (each binary has
that run's weights baked into `libcomputer/weights_policy.c`) and runs all 7 in
parallel on 8 physical cores. ~5 min wall time for the whole sweep.

## Data

- Source: WTHOR human games, `wthor_files/WTH_1977.wtb` ... `WTH_2025.wtb` (49 files).
- Targets: the move the human played, given the board state.
- Input layout (canonicalized in `nn_format_input`, `libcomputer/nn_policy.c`):
  - `[0:64)`   valid-move plane
  - `[64:128)` "my" pieces (player to move)
  - `[128:192)` "opp" pieces
  No explicit current-player byte; everything is from the side-to-move view.
- Augmentation: D4 (8 dihedral symmetries) in `apps/data_gen_policy.c`. See
  the entry on switching from C4 to full D4 below.

## Architectures

Defined in `othello_ml/scripts/train.py`:
- `NN`: input → N1 → N2 → 64, two ReLU hidden layers.
- `NN_single`: input → N1 → 64, one hidden layer. Used to test whether the
  second layer is doing useful work.
- `CNN`: two 3×3 conv layers + `Linear(N2*8*8, 64)` head. Head breaks
  translation invariance, which is correct for Othello (corners ≠ centers).
  Has not been benchmarked recently; on the to-do list.

Style table is `othello_ml/config.py::model_styles`. NN styles in current use:
tiny (256/128), small (1024/1024), medium (2048/1024), large (2048/2048).
NN_single styles: single_256, single_1024, single_2048.

## Open questions

- **CNN benchmark.** Not run with current loss + D4. Expected to outperform
  MLPs at fixed param count given Othello's spatial structure.
- **Net-guided MCTS eval.** Need to rework the MCTS implementation before
  this can ship; current thunderdome only measures NN-alone strength.
- **Value head.** Pure policy is myopic. Adding a value head lets us prune
  search more aggressively.
- **Drop valid plane from input.** Once `loss_masked_to_valid` is the
  default, the input valid plane is mostly redundant (legality is enforced
  externally at inference). Drop it for a 33% smaller L1.
- **Pre-aggregate duplicate boards.** Same board appearing in N games
  contributes N one-hot targets; could be one soft target with weight N
  for the same gradient signal at 1/N the per-epoch compute. See note in
  `apps/data_gen_policy.c`. (Only ~3% of positions repeat — savings
  small.)
- **Split by row, not by game.** Now that `board_lookback=0` there's no
  temporal leakage from per-row splits. See `othello_ml/scripts/make_split.py`.

---

## Entries

### 2026-05-11 — initial scale-and-shape sweep (sweep base `32e2`)

Trained 6 styles in parallel on Modal: tiny, single_256, single_1024, single_2048,
medium, large. Plus a one-off small (`1b55`) baseline. All with the original
`loss_without_invalid` and C4 (4 rotations) augmentation.

Thunderdome results (1000 trials, 95% CI):

| Model | NN winrate | white | black |
|---|---|---|---|
| medium (N1=2048,N2=1024) | 0.550 [0.519, 0.581] | 0.552 | 0.548 |
| large (N1=2048,N2=2048) | 0.538 [0.507, 0.569] | 0.524 | 0.552 |
| 1b55 small (N1=1024,N2=1024) | 0.488 [0.457, 0.519] | 0.478 | 0.498 |
| single_2048 (N1=2048) | 0.481 [0.450, 0.512] | 0.428 | 0.534 |
| tiny (N1=256,N2=128) | 0.427 [0.397, 0.458] | 0.382 | 0.472 |
| single_1024 (N1=1024) | 0.433 [0.403, 0.464] | 0.376 | 0.490 |
| single_256 (N1=256) | 0.392 [0.362, 0.423] | 0.366 | 0.418 |

Takeaways:
- Width saturates: medium → large gives nothing. The architecture or
  training signal is the bottleneck, not capacity.
- Two layers beat one at every matched width. The middle layer matters.
- Mild color asymmetry (4-10 pt) on most models, especially the
  single-layer variants.
- 55% vs 2000-rollout MCTS is a real but modest result. A strong Othello
  engine should be >95% against this MCTS. Lots of headroom.

### 2026-05-12 — policy visualization (`othello_ml/scripts/inspect_policy.py`)

Built a viz that, for a given trained model, plots on curated positions:
the board, the 64 output logits as a heatmap, softmax over valid moves, and
saliency (∂top_logit/∂input) per input plane. Renders into
`<experiment_dir>/inspect/policy/`.

Findings on `32e2-medium`:
1. **Opening logits are not D4-symmetric.** The 4 valid starting moves for
   Black should be equivalent under D4. The trained net assigns very
   different scores: (3,2) and (4,5) ≈ 11, (2,3) and (5,4) ≈ 0. Diagnosis:
   data gen uses C4 only (`N_SYMMETRIES 4`); under C4, the 4 starting
   moves are not in the same orbit because 90° rotation also swaps the
   colors of the starting diamond. Fix: full D4 augmentation (8 transforms).
2. **The valid-moves plane dominates saliency by ~10×** over the my/opp
   planes. The network learned to copy the valid plane into the output to
   suppress illegal-square logits — that's most of what its capacity
   bought. The remaining ~10% signal is the actual board-reading policy.

### 2026-05-12 — masked-to-valid loss (`loss_masked_to_valid`)

Changed `othello_ml/scripts/train.py::loss_masked_to_valid` and made it the
default. Mechanism: before softmax, set illegal-square logits to -1e9 so the
softmax denominator only sums over legal moves. Cross-entropy then becomes
−log p[played | legal moves]. Illegal-square logits are excluded from the
loss entirely, so the network no longer has incentive to suppress them by
copying the valid plane.

Also updated the test-accuracy metric in `trainer()` to argmax over legal
moves (matches C inference behavior; otherwise lossmask runs look spuriously
worse because illegal logits aren't suppressed).

Run dirs now include a loss tag: `<timestamp>_<hex>_<style>_<lossmask|lossall>_full`.
See `othello_ml/config.py::LOSS_TAGS`.

Retrained medium only (`20260512_203506_4b3b_medium_lossmask_full`):

| Model | winrate | white | black |
|---|---|---|---|
| medium lossall (old) | 0.550 [0.519, 0.581] | 0.552 | 0.548 |
| medium lossmask (new) | 0.541 [0.510, 0.572] | 0.542 | 0.540 |

Raw winrate: statistically tied.

But the visualizations are *very* different:
- Saliency on valid plane no longer dominates — all 3 planes now ±0.6 on
  the same scale (opening) or ±1.7 (corner-pressure midgame).
- Logit magnitudes shrunk ~5× (no more "shout illegal moves down" pressure).
- Color symmetry improved (0.542/0.540 vs 0.552/0.548).
- Opening D4-asymmetry persists — confirms it's a data problem, not loss.

Interpretation: loss change is a necessary cleanup that removes the legality
shortcut, but doesn't auto-translate to stronger play. The 55% ceiling has
a different cause (likely architecture/data, not capacity). What it *enables*
is real comparisons of architectural changes — capacity freed from the
shortcut is now usable for actual policy work, so CNNs / smaller nets should
now show their true relative merits.

### 2026-05-13 — full D4 in data gen

`apps/data_gen_policy.c::transform_xy` now defines 8 transforms (rotations +
reflections). `N_SYMMETRIES = 8`. Each WTHOR game now contributes 8 training
samples instead of 4 — dataset doubles, training time doubles, expected
improvement is mostly in opening symmetry and sample efficiency for spatial
patterns.

Launched sweep `20260513_225809_5118` with all 7 styles + new loss + D4.

### 2026-05-14 — sweep `5118` results (loss_masked_to_valid + D4)

`large` style timed out on Modal (>3600s train; double dataset + 64 epochs
overran the function timeout). The other 6 finished. Note: Modal dump_weights
silently failed for the three `single_*` runs — only `best.pt` shipped; ran
`dump_weights.py` locally (extended to fall back to `best.pt` when
`checkpoints/LATEST` is absent).

Thunderdome results (1000 trials, 95% CI), with the prior 32e2/1b55 sweep
(lossall + C4) for comparison:

| Model | new (lossmask + D4) | prior (lossall + C4) | Δ |
|---|---|---|---|
| tiny       | 0.501 [0.470, 0.532] | 0.427 | +0.074 |
| small      | 0.529 [0.498, 0.560] | 0.488 | +0.041 |
| medium     | 0.526 [0.495, 0.557] | 0.550 | −0.024 |
| single_256 | 0.439 [0.409, 0.470] | 0.392 | +0.047 |
| single_1024| 0.466 [0.435, 0.497] | 0.433 | +0.033 |
| single_2048| 0.492 [0.461, 0.523] | 0.481 | +0.011 |
| large      | (timed out)          | 0.538 | — |

Color symmetry (white vs black winrate) also tightened on most models. E.g.
single_2048: 0.428/0.534 → 0.502/0.482; medium: 0.552/0.548 → 0.530/0.522.

Takeaways:
- **Smaller models gained the most.** tiny +7.4pt, single_256 +4.7pt,
  small +4.1pt. Capacity that was being spent on the legality shortcut and
  the C4-asymmetric opening is now spent on actual play.
- **Medium slipped slightly.** −2.4pt, CIs overlap so statistically tied.
  Reading: medium was already large enough that the shortcut wasn't a
  bottleneck for it — and the harder loss + 2× dataset may need more
  epochs (it stopped at epoch 38, vs tiny's 63 and small's 47).
- **The 55% ceiling persists.** Best model (small at 0.529) is below the
  prior best (medium at 0.550). Loss + D4 cleanup didn't break the ceiling
  — it redistributed strength across model sizes. Confirms ceiling cause
  is architectural (MLP can't exploit spatial structure) or signal-related
  (pure policy-from-human-moves is myopic), not capacity or shortcut.
- **Opening D4 symmetry: fixed.** Medium logit range across the 4
  starting moves: 1.18–1.27 (spread 0.09), vs 0 vs 11 (spread 11) under
  C4. Tiny: 1.70–1.76 (spread 0.06). Effectively D4-symmetric.
- **Saliency: balanced.** my/opp planes now dominate over valid plane on
  all 6 models (was inverted ~10× under old loss).

For `large`: needs longer Modal timeout (>3600s) or fewer epochs to
re-attempt. Not currently the bottleneck (medium already saturating).
Likely punted until we have something new to test at that scale (CNN,
value head, or net-guided MCTS).

Next priorities given the unbroken ceiling:
1. CNN benchmark (spatial-structure win at fixed param count).
2. Drop input valid plane (now redundant under lossmask) — free 33% of L1.
3. Value head — pure-policy training is myopic; mate-search via MCTS
   probably needs the value signal.

### 2026-05-14 — data floor analysis (`othello_ml/scripts/data_floor.py`)

Computed the lower bound on test CE loss: for each unique (canonicalized,
D4-augmented) board, compute the entropy of the empirical distribution of
human moves at that board. Average weighted by row count is the floor
(oracle's best possible CE). Modal-move accuracy is the top-1 ceiling.

Dataset stats (WTHOR + full D4, 47 files):

```
rows (post-D4)        : 62,096,552
unique positions      : 43,031,612
avg rows per position : 1.44
positions seen 1x     : 96.7%
cross-entropy floor   : 0.2330
modal-move accuracy   : 0.9002
rows by k distinct played moves: k=1 70.2%, k=2 4.2%, k=3 4.3%, k=4 5.4%, ...
```

Othello's legal state space is ~10²⁸; WTHOR samples ~3M position-instances
out of that tree, so positions appearing once is expected (a game-tree
sample inherits the tree's branching). What this confirms is that **the
target distribution itself is low-entropy** — CE floor 0.233 (vs
uniform-on-64 = 4.16) and a perfect memorizer hits 90% top-1. Tons of
headroom; signal noise is not the bottleneck.

### 2026-05-14 — model loss vs floor (`othello_ml/scripts/eval_loss.py`)

Ran the sweep-5118 `medium` model over the full 62M-row dataset
(mixed train+test, 80/20 split done by-game during training):

```
=== medium_lossmask on WTHOR(D4) ===
avg CE loss     : 1.1563
top-1 accuracy  : 0.5606
floor (oracle)  : 0.2330
gap to floor    : +0.92
```

The model is nowhere near fitting the data — gap of 0.92 CE on data it
mostly saw during training. Top-1 of 56% vs 90% ceiling. This rules out
"data is too noisy to learn from" — the data floor is low, the model just
can't reach it.

Probable cause: an MLP must learn each (position → move) pair as an
independent fact through ~3M random connections per neuron. With weight
decay = 1e-4 it pays a penalty for memorization. With most positions
appearing once, there's no redundant signal to drive generalization either.
Weight-sharing (CNN) is the natural fix: a corner-recognizer learned at
(0,0) transfers to (0,7), (7,0), (7,7) at zero generalization cost.

Updates the open questions: the bottleneck is now confirmed to be
**model fit, not data signal.** "Add a value head" is still useful long
term, but it's not what'll break the 55% ceiling.

Open question for follow-up: split by *row* instead of *game* now that
`board_lookback=0`. Should give a cleaner train/test loss comparison since
both will sample from the same per-row distribution. Won't change the floor
or the conclusion above (medium fails on its own training rows), but lets
us measure overfitting cleanly.

### 2026-05-14 — train vs test loss for medium

Regenerated the data + split-by-game (seed 12345 in `make_split.py`,
deterministic from the WTHOR IDs) and evaluated `medium_lossmask` on each:

```
train CE / acc : 1.1557 / 0.5609
test  CE / acc : 1.1589 / 0.5595
train-test gap : +0.0032
```

Train == test. **No overfitting at all** — the model isn't memorizing the
training set. It's pure underfitting. CE 1.16 vs floor 0.23 on its own
training data; top-1 56% vs 90% ceiling.

This kills the "CNN for generalization" framing — the MLP isn't even
hitting the train ceiling, so generalization isn't the binding constraint
yet. Most likely culprit: `weight_decay = 1e-4` (configured in
`modal_train.py`) is preventing the per-position memorization that's the
only way to drive train loss down on a dataset where 96.7% of positions
appear once. With weight sharing inactive (MLP) and weight decay active,
each position needs its own dedicated parameter mass and the regularizer
punishes that directly.

Cheapest next test: retrain medium with `weight_decay=0`. If train loss
drops, the architecture is fine and the training recipe was the bottleneck.
If train loss is still ~1.16, optimization or activation matters and we
have to dig in.

### 2026-05-14 — medium retrained with `weight_decay=0` (run `37b0`)

`20260514_223601_37b0_medium_lossmask_full`. Modal trainer log:
- train loss drove to 0.0002 by epoch 32 (vs 1.16 with wd=1e-4)
- early stop at epoch 32, best test accuracy = 0.5943 at epoch 24

Eval on `best.pt` (epoch 24, the saved checkpoint):

```
                       wd=1e-4 (5118)   wd=0 (37b0)
train CE / acc       : 1.1557 / 0.5609  1.0089 / 0.6135
test  CE / acc       : 1.1589 / 0.5595  1.0627 / 0.5943
train-test CE gap    : +0.0032          +0.0538
thunderdome winrate  : 0.526            0.715 [0.686, 0.742]
as white / as black  : 0.530 / 0.522    0.700 / 0.730
```

**+19pt winrate from removing weight decay.** The previous "55% ceiling"
was a hyperparameter mistake, not a fundamental architecture or signal
limit. With wd=0:
- The MLP *can* memorize (Modal log shows loss → 0.0002 at epoch 32)
- Early-stop saves at epoch 24 before runaway overfitting
- Top-1 accuracy moved modestly (+3.5pt) but winrate moved a lot (+19pt)

The accuracy vs winrate gap is interesting: the model's whole logit
distribution improved (CE down ~0.10), so even when the top-1 prediction
is wrong, the runner-up is much closer to the human choice. MCTS sees
this as much-improved playstyle.

This re-orders priorities:
1. **Re-sweep all styles with wd=0** to find the new best. Particularly
   curious if `large` recovers from its prior tie with `medium`, and
   whether `tiny` retains its post-cleanup boost.
2. **Scan weight_decay** (wd=0 overfits given time; we got lucky with
   early-stop. Optimal is probably 1e-6 to 1e-5.)
3. **CNN benchmark** — still worth it for spatial inductive bias, but the
   immediate gap from MLP-as-trained → MLP-tuned was bigger than expected.

### 2026-05-15 — full wd=0 sweep (run `6835`, on L4)

All 7 styles, `weight_decay=0`, on L4 GPU (~$0.80/hr vs A100's ~$2.78/hr;
small MLPs don't use the A100 compute and were always loader-bound).
Bumped train timeout to 7200s; `large` finished cleanly this time.
Orchestrator now catches per-style failures so a single bad run doesn't
kill the rest.

| Model | wd=1e-4 (5118) | wd=0 (6835) | Δ | white / black |
|---|---|---|---|---|
| tiny        | 0.501 | 0.540 [0.509, 0.571] | +0.039 | 0.536 / 0.544 |
| small       | 0.529 | **0.750 [0.722, 0.776]** | +0.221 | 0.690 / 0.810 |
| medium      | 0.526 | 0.691 [0.662, 0.719] | +0.165 | 0.684 / 0.698 |
| large       | timeout | 0.702 [0.673, 0.730] | — | 0.720 / 0.684 |
| single_256  | 0.439 | 0.574 [0.543, 0.604] | +0.135 | 0.580 / 0.568 |
| single_1024 | 0.466 | 0.683 [0.654, 0.711] | +0.217 | 0.636 / 0.730 |
| single_2048 | 0.492 | 0.672 [0.642, 0.700] | +0.180 | 0.674 / 0.670 |

Takeaways:
- **`small` (1024/1024) is the new best at 0.750.** Beats `medium` and
  `large`. Wider isn't strictly better past `small`.
- **Removing wd helps every size**, but the gain is U-shaped vs capacity:
  +4pt at tiny (was already capacity-bound), +22pt at small (the sweet
  spot), +16-18pt at medium/large.
- **Single-hidden-layer is now competitive.** `single_1024` (0.683) is a
  hair below `medium` (0.691); the middle layer's value is much smaller
  than the prior wd=1e-4 sweep suggested. At matched width the NN still
  beats NN_single but not by much. Useful for Playdate inference cost.
- **`small`'s color asymmetry is large** (white 0.690 vs black 0.810,
  +12pt for black). Other models are tighter. Worth a look in viz to see
  if it's a positional bias or a quirk of best.pt selection.
- The prior "55% ceiling" wasn't fundamental — it was wd=1e-4. The
  apparent new ceiling sits at ~75% and may also be soft.

Action items shifted:
1. **Weight-decay scan** — wd=0 is fine with early-stop but a small wd
   (1e-6, 1e-5) might smooth things further. Cheap to test.
2. **More epochs / longer patience** — `small` topping the ranking
   suggests medium/large might still have headroom if we trained longer.
3. **CNN** — still worth trying. With MLP-tuning largely tapped, the
   spatial inductive bias is what's left to test before adding a value
   head or net-guided MCTS.

### 2026-05-16 — drop the valid-moves input plane (`small_no_valid`, run `ecb7`)

Hypothesis from prior viz: under `loss_masked_to_valid` the valid-moves
input plane has near-zero saliency — the network learns from my/opp and the
legality mask is enforced externally at argmax. Adding 64 input features the
net doesn't need should be neutral-to-bad (extra params absorbing noise).

Implementation: new `NN_no_valid` model class that slices `X[:, 64:]` in
forward; L1 sized 128→1024 instead of 192→1024. Data layout unchanged
(loaders still emit 192-byte uint8 batches; loss_masked_to_valid still reads
the valid plane out of `batch[:, 0:64]`). `dump_weights.py` and
`nn_policy.c` learned a new `NN_POLICY_L1_INPUT_SIZE` macro and slice the
formatted input by `(INPUT_SHAPE - L1_INPUT_SIZE)` bytes before the L1
matmul.

Thunderdome (1000 trials):

| Model | winrate | white | black |
|---|---|---|---|
| small (wd=0, valid plane in) | 0.750 [0.722, 0.776] | 0.690 | 0.810 |
| small_no_valid (wd=0)        | **0.809 [0.783, 0.832]** | 0.788 | 0.830 |

**+5.9pt winrate from removing 64 input features.** Color asymmetry also
shrank (0.788/0.830 vs 0.690/0.810). The valid plane wasn't neutral — it
was actively hurting. Plausible mechanism: the extra parameters from
valid→L1 weren't dead, they were absorbing useful capacity that should've
been spent on board-reading; or they were introducing a subtle leak between
"squares I might play" and the policy output that distorted move ranking.

This is also a Playdate win: L1 weight matrix is 33% smaller (128×1024 vs
192×1024) = 256KB saved per matrix in fp32.

Locked in as the default `libcomputer/weights_policy.c`.

Two infra issues found and fixed along the way:
- **torch.compile state_dict prefix.** `torch.compile(net)` wraps the
  module in `OptimizedModule` and prefixes all state keys with
  `_orig_mod.`. Broke `dump_weights.py`, `inspect_policy.py`, and Modal
  `dump_weights` step. Fix: strip the prefix on load (back-compat); save
  the unwrapped state going forward.
- **Modal `dump_weights` failed for the original `ecb7` run** because the
  image was baked before the prefix-strip landed. Workaround: download
  `best.pt`, run `dump_weights.py` locally.

