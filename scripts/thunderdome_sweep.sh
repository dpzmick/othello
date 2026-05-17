#!/usr/bin/env bash
# Sweep thunderdome across all NN-arch runs in experiments/.
# Two phases:
#   1. Sequential: cp each run's weights into libcomputer, rebuild thunderdome,
#      move the binary aside (one binary per run, weights baked in).
#   2. Parallel: launch all binaries at once, one per CPU. Collect results.
set -euo pipefail

# Paths below are repo-root relative; cd up from scripts/ so the script can
# be invoked from anywhere.
cd "$(dirname "$0")/.."

RUNS=(
    "experiments/20260515_205758_6835_tiny_lossmask_full|6835-tiny"
    "experiments/20260515_205758_6835_small_lossmask_full|6835-small"
    "experiments/20260515_205758_6835_medium_lossmask_full|6835-medium"
    "experiments/20260515_205758_6835_large_lossmask_full|6835-large"
    "experiments/20260515_205758_6835_single_256_lossmask_full|6835-single-256"
    "experiments/20260515_205758_6835_single_1024_lossmask_full|6835-single-1024"
    "experiments/20260515_205758_6835_single_2048_lossmask_full|6835-single-2048"
)

WORKDIR=/tmp/thunderdome_sweep
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

# --- phase 1: build a binary per run -------------------------------------
echo "Building $(echo "${#RUNS[@]}") binaries..."
for entry in "${RUNS[@]}"; do
    run_dir="${entry%%|*}"
    label="${entry##*|}"
    bin="$WORKDIR/thunderdome_${label}"

    if [[ ! -f "${run_dir}/weights_policy.c" ]]; then
        echo "  SKIP ${label}: no weights_policy.c"
        continue
    fi

    cp "${run_dir}/weights_policy.c" libcomputer/weights_policy.c
    nix develop --command bash -c "cmake --build build -j --target thunderdome" \
        >> "$WORKDIR/build.log" 2>&1
    cp ./build/apps/thunderdome "$bin"
    echo "  built ${label}"
done

# --- phase 2: run all in parallel ---------------------------------------
echo
echo "Running ${#RUNS[@]} binaries in parallel..."
declare -a pids labels
for entry in "${RUNS[@]}"; do
    label="${entry##*|}"
    bin="$WORKDIR/thunderdome_${label}"
    [[ -x "$bin" ]] || continue
    "$bin" > "$WORKDIR/out_${label}.txt" 2>&1 &
    pids+=($!)
    labels+=("$label")
done

start=$(date +%s)
for pid in "${pids[@]}"; do wait "$pid"; done
elapsed=$(( $(date +%s) - start ))
echo "Done in ${elapsed}s."
echo

# --- phase 3: collate -----------------------------------------------------
printf "%-18s | %-44s | %-12s | %-12s\n" \
       "label" "result" "as white" "as black"
printf "%-18s-+-%-44s-+-%-12s-+-%-12s\n" \
       "------------------" "--------------------------------------------" \
       "------------" "------------"

for label in "${labels[@]}"; do
    out=$(cat "$WORKDIR/out_${label}.txt")
    result_line=$(echo "$out" | grep "NN winrate" || echo "?")
    as_white=$(echo "$out" | grep "as white" | awk -F'= ' '{print $2}')
    as_black=$(echo "$out" | grep "as black" | awk -F'= ' '{print $2}')
    printf "%-18s | %-44s | %-12s | %-12s\n" \
           "$label" "$result_line" "$as_white" "$as_black"
done

echo
echo "Per-run outputs: $WORKDIR/out_*.txt"
