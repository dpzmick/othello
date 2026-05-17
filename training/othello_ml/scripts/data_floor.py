"""Compute the cross-entropy floor of the WTHOR policy targets.

Lower bound on test CE loss: even an oracle that knows the exact empirical
distribution of human moves at each board cannot beat -E[log p_pos[played]].

Reads a dataset produced by data_gen_policy. Usage:
    uv run python -m othello_ml.scripts.data_floor <experiment_dir>
"""

import glob
import os
import sys
import time

import numpy as np
import pyzstd

from ..config import open_config


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <experiment_dir>", file=sys.stderr)
        sys.exit(1)

    experiment_dir = sys.argv[1].rstrip('/')
    config = open_config(experiment_dir)
    boards_dir = config['files']['boards_dir']
    policy_filename = config['files']['policy_filename']

    t0 = time.time()
    print(f"loading boards from {boards_dir}", file=sys.stderr)
    files = sorted(glob.glob(os.path.join(boards_dir, '*.dat.zst')))
    parts = []
    for f in files:
        with open(f, 'rb') as fp:
            arr = np.frombuffer(pyzstd.decompress(fp.read()), dtype=np.uint8).reshape(-1, 192)
        # only need my/opp planes for hashing positions
        parts.append(arr[:, 64:192].copy())
    boards = np.concatenate(parts, axis=0)
    del parts
    print(f"  loaded {len(boards):,} rows in {time.time()-t0:.1f}s", file=sys.stderr)

    t0 = time.time()
    print(f"loading policy from {policy_filename}", file=sys.stderr)
    with open(policy_filename, 'rb') as fp:
        policy = np.frombuffer(pyzstd.decompress(fp.read()), dtype=np.float32).reshape(-1, 64)
    played = np.argmax(policy, axis=1).astype(np.uint8)
    n_rows = len(played)
    print(f"  loaded policy in {time.time()-t0:.1f}s", file=sys.stderr)

    # Pack the 128 board bits into 2 uint64s (lo, hi).
    t0 = time.time()
    print("packing positions...", file=sys.stderr)
    packed = np.packbits(boards, axis=1).view(np.uint64).reshape(n_rows, 2)
    lo = np.ascontiguousarray(packed[:, 0])
    hi = np.ascontiguousarray(packed[:, 1])
    del boards, packed
    print(f"  packed in {time.time()-t0:.1f}s", file=sys.stderr)

    # Sort lexicographically by (lo, hi, played). lexsort takes keys in
    # reverse priority — last key is primary.
    t0 = time.time()
    print("sorting...", file=sys.stderr)
    order = np.lexsort((played, hi, lo))
    lo = lo[order]
    hi = hi[order]
    mv = played[order]
    del order
    print(f"  sorted in {time.time()-t0:.1f}s", file=sys.stderr)

    t0 = time.time()
    print("aggregating...", file=sys.stderr)
    # (position, move) group boundaries
    pm_change = np.empty(n_rows, dtype=bool)
    pm_change[0] = True
    pm_change[1:] = (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1]) | (mv[1:] != mv[:-1])
    pm_starts = np.flatnonzero(pm_change)
    pm_counts = np.diff(np.concatenate((pm_starts, np.array([n_rows], dtype=pm_starts.dtype))))

    # Position-only group boundaries (a subset of the above).
    p_change = np.empty(n_rows, dtype=bool)
    p_change[0] = True
    p_change[1:] = (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])
    p_starts = np.flatnonzero(p_change)
    p_counts = np.diff(np.concatenate((p_starts, np.array([n_rows], dtype=p_starts.dtype))))
    n_positions = len(p_starts)
    del lo, hi, mv, pm_change, p_change

    # For each (pos, move) group, look up its position's total row count.
    p_idx = np.searchsorted(p_starts, pm_starts, side='right') - 1
    totals_for_pm = p_counts[p_idx]

    # Cross-entropy contribution: each row in a (pos, move) group contributes
    # -log(c/T). Sum across the group is c * -log(c/T).
    log_p = np.log(pm_counts.astype(np.float64) / totals_for_pm.astype(np.float64))
    loss_sum = -(pm_counts * log_p).sum()
    floor = loss_sum / n_rows

    # Modal-move accuracy: per position, take the max count over its
    # (pos, move) groups. reduceat needs the index in pm-space of the first
    # pm group belonging to each position.
    first_pm_per_p = np.searchsorted(pm_starts, p_starts)
    modal_counts = np.maximum.reduceat(pm_counts, first_pm_per_p)
    modal_acc = modal_counts.sum() / n_rows

    # Distinct moves per position
    pm_count_per_p = np.diff(np.concatenate(
        (first_pm_per_p, np.array([len(pm_counts)], dtype=first_pm_per_p.dtype))
    ))
    # Histogram of "distinct moves" weighted by the # of rows at that position
    n_distinct_dist = np.bincount(pm_count_per_p, weights=p_counts, minlength=2).astype(np.int64)
    n_rows_at_unique_pos = (p_counts == 1).sum()
    print(f"  aggregated in {time.time()-t0:.1f}s", file=sys.stderr)

    print()
    print(f"=== WTHOR policy-target data floor ===")
    print(f"rows (post-D4)        : {n_rows:,}")
    print(f"unique positions      : {n_positions:,}")
    print(f"avg rows per position : {n_rows / n_positions:.2f}")
    print(f"positions seen 1x     : {n_rows_at_unique_pos:,} "
          f"({100*n_rows_at_unique_pos/n_positions:.1f}%)")
    print()
    print(f"cross-entropy floor   : {floor:.4f}")
    print(f"   (best possible test CE; oracle picks empirical distribution)")
    print()
    print(f"modal-move accuracy   : {modal_acc:.4f}")
    print(f"   (top-1 accuracy ceiling: always pick the most-common human move)")
    print()
    print("rows at positions with k distinct played moves:")
    for k in range(1, 11):
        n = n_distinct_dist[k]
        if n > 0:
            print(f"  k={k:2d}: {n:>12,} rows  ({100*n/n_rows:5.1f}%)")
    rest = n_distinct_dist[11:].sum()
    if rest > 0:
        print(f"  k>10: {rest:>12,} rows  ({100*rest/n_rows:5.1f}%)")


if __name__ == "__main__":
    main()
