"""Summarize a torch.profiler Chrome trace.

For each event category (kernel / memcpy / cpu_op / etc.) report total
duration and the top operators by self-time. Useful for answering
'where is the GPU spending its time' without firing up perfetto.

    uv run python -m othello_ml.scripts.analyze_trace <trace.json>
"""

import collections
import json
import sys


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <trace.json>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        trace = json.load(f)

    events = trace.get("traceEvents", trace) if isinstance(trace, dict) else trace
    print(f"loaded {len(events):,} events")

    # Per-category totals (cat is e.g. "kernel", "cpu_op", "gpu_memcpy").
    by_cat = collections.defaultdict(lambda: [0.0, 0])  # [total_us, count]
    # Per-(cat, name) totals.
    by_name = collections.defaultdict(lambda: [0.0, 0])

    min_ts = None
    max_ts = None

    for ev in events:
        if ev.get("ph") != "X":
            continue
        dur = ev.get("dur")
        if dur is None:
            continue
        cat = ev.get("cat", "?")
        name = ev.get("name", "?")
        ts = ev.get("ts", 0)
        if min_ts is None or ts < min_ts:
            min_ts = ts
        end = ts + dur
        if max_ts is None or end > max_ts:
            max_ts = end

        by_cat[cat][0] += dur
        by_cat[cat][1] += 1
        by_name[(cat, name)][0] += dur
        by_name[(cat, name)][1] += 1

    span_us = (max_ts - min_ts) if (min_ts is not None and max_ts is not None) else 0
    print(f"trace span: {span_us/1000:.1f} ms")
    print()

    print("=== totals by category ===")
    rows = sorted(by_cat.items(), key=lambda kv: -kv[1][0])
    for cat, (total, n) in rows:
        print(f"  {cat:<20} {total/1000:>10.2f} ms  ({n:>7,} events, avg {total/max(n,1):.1f} us)")

    print()
    cats_of_interest = ["kernel", "gpu_memcpy", "cpu_op", "cuda_runtime", "python_function", "user_annotation"]
    for cat in cats_of_interest:
        rows = [(name, total, n) for (c, name), (total, n) in by_name.items() if c == cat]
        if not rows:
            continue
        rows.sort(key=lambda r: -r[1])
        print(f"=== top {cat} by total time ===")
        for name, total, n in rows[:15]:
            print(f"  {total/1000:>9.2f} ms  ({n:>6,}x, avg {total/n:>7.1f} us)  {name}")
        print()


if __name__ == "__main__":
    main()
