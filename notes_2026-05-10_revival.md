# Session summary — 2026-05-10 (project revival)

## What we did

**Build system reproducible.** Added `flake.nix` providing cmake / emscripten / node 20 / uv / dev tools on darwin + linux. Python deps moved to `pyproject.toml` + `uv`. Native build, wasm build, web dev server, and Python pipeline all work from a single `nix develop`. Cleaned 3GB of stale build artifacts.

**Got the web game compiling again.** Bumped svelte 3→5, vite 4→6, vite-plugin-svelte 2→5. 0 audit vulnerabilities now. Fixed hardcoded Linux path in `vite.config.js`, deleted dead `Cell.svelte` stub, migrated `main.js` to Svelte 5's `mount()`. Removed broken MCTS+NN dropdown option (mode 2 was unimplemented and aborted after applying human's move).

**Real bug fixes in C:**
- `nn.c`: `relu(hidden1, 512)` was being called twice instead of `relu(hidden2, 512)` — silent inference bug.
- `nn.c`: `nn_select_move` was scoring every candidate on the *unchanged* starting board (forgot to use `updated_game->white/black`) — meaning the NN player was effectively returning the first move every time.
- `data_gen_policy.c`: id assignment incremented per-move-pair instead of per-game, causing train/test split to scatter moves of the same game across both sets (data leakage). Fixed by hoisting id allocation outside the move loop.
- `thunderdome.c`: MCTS hash table sized for 8192 entries but used 2000 trials/move — overflowed and aborted. Bumped to 65536.

**Repo cleanup.** Moved dead python and superseded code to `attic/old_python/`, `attic/ml_2023/`, `attic/old_apps/`, `attic/old_configs/`. Marked `libcomputer/nn.c` with a header comment explaining it's the dead value-net architecture still used as the NN-player slot.

**ML pipeline simplification.**
- Replaced `BoardDirLoader`'s linux-only `/dev/shm` + `fallocate` prefetch architecture with a simple in-memory loader (~200 LOC deleted).
- Fixed `.to("cuda")` hardcoding in `train.py` so MPS actually works on M1.
- Fixed `torch.load(weights_only=False)` for PyTorch 2.6+.
- Added always-save-final + every-16-epoch checkpoints (was every 64, no final).

**Architecture decisions (with rationale):**
- `board_lookback=0` going forward. Othello is Markovian; lookback added input cost for no information gain.
- Documented in two source comments that one-hot duplicates + cross-entropy ≡ soft-target distribution. So the policy training is correct as-is, with the caveat that pre-aggregating would save compute.

**Big training speedup (~4.5×).** Identified MPS bottlenecks: per-batch host↔device copies, per-batch `.item()` sync, small batch. Fixes: pre-move training data to device once, accumulate loss as a tensor (sync once per epoch), batch_size 512→4096. **Per-epoch went from ~5s to ~1.1s**, outlier epochs (the random 200s ones) disappeared.

**Trained and deployed a policy MLP.**
- 1024×1024 MLP on 10 WTHOR files, 32 epochs in 45s
- Peak test top-1 accuracy: **44.6% at epoch 20**
- Wrote `dump_weights.py` to export `.pt` → `weights_policy.c`
- Wrote `nn_policy.c` for C inference (forward pass + invalid-move masking + argmax)
- Wired into thunderdome (replaces the dead `nn_select_move`)
- **Result: 38–12 (NN wins 24%) vs MCTS-2000** — up from 0–50 with the dead value-net MLP.

## Things we noted but didn't act on

- **8-fold dihedral symmetry augmentation.** Currently only horizontal+vertical flip; missing 6 of 8 symmetries. Free 4× data.
- **Input canonicalization.** Drop the player byte; swap white/black planes per side-to-move. Standard practice for two-player games.
- **Logit masking during training**, not just inference. Currently invalid-move logits compete in the cross-entropy loss for no reason.
- **Dedup with sample weights** in data-gen. Reduces training compute substantially; `notes.org` has it listed as a planned flag.
- **`valid_moves_vectorized_intel.h` is misnamed** — it's the generic GCC vector path.
- **Cosmetic:** misleading "branch-free relu" comment in `nn.c`.
- **wandb wired up but disabled** for these runs. Real experiment tracking should turn it on.
- **Data-gen perf is fine** at ~550k rows/sec; potential 5-10× speedup if it ever matters.

## What's next, in priority order

**Phase: improve the policy network (Milestone 2).**

1. **Cheap improvements to the current MLP, in one combined run:**
   - `weight_decay=1e-4` (model is overfitting)
   - Try `model_style='small_norm'` (NN_with_norm, has BatchNorm)
   - Early stopping by best-test checkpoint
   - Train on full 46 WTHOR files (was 10)
   - Re-deploy, re-thunderdome, see if win rate climbs
2. **Input canonicalization + 8-fold symmetry.** Two changes to `data_gen_policy.c` and `nn_format_input`. Get a fair "what can a small MLP really do?" number.
3. **Move to CNN.** Train `cnn_small` on the same setup. Will require writing C inference for the CNN (conv2d, batchnorm, flatten, linear). Bigger code change but expected to be the biggest single quality jump.

**Phase: NN-guided MCTS (Milestone 3).**

4. Add value head to the network (joint policy + value training).
5. Implement PUCT in `mcts.c` using the policy as prior and value as rollout replacement. Wire mode=2 in `wasm_wrapper.c`.

**Phase: Playdate (Milestone 4).**

6. Int8 quantization, fixed-point inference, opening book, port and ship.

**Smaller things worth doing whenever:**
- Add color rotation + paired matchups to thunderdome (variance reduction for comparing close model variants).
- Add a wandb MCP server so I can query training metrics directly.
- The two cosmetic C cleanups (rename `valid_moves_vectorized_intel.h`, fix the misleading relu comment) whenever we're touching those files.

---

Recommended next session start: **(1) above** — cheap MLP improvements bundled in one run. Concrete, measurable against the 24% baseline, and fast on M1 (full WTHOR train should be a few minutes with our optimizations).
