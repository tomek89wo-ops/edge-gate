"""Reproduce the headline number: how often does pure noise look like a find?

Run:  python examples/noise_sweep.py

The README and the `overlapping_windows` demo both claim that sweeping seeds
0-59 through one fixed construction of PURE NOISE produces the overlapping-
windows artefact in a specific number of runs. That claim was, until this
file existed, only a sentence. A number you cannot re-run is a number you
should not believe, least of all in a repository whose entire subject is
refusing to believe numbers. So here is the script that computes it.

The construction is copied from `examples/demo.py::overlapping_windows` and
must stay identical to it: two independent white-noise series, one smoothed
over 60 days as the "signal", the other over 20 days as the "forward return".
Nothing links them. Any significance is manufactured by bootstrapping
overlapping observations as if they were independent.

A run counts as the artefact when BOTH hold:
  * naive reading is significant   - p < 0.05  with block = 1
  * honest reading is not          - p > 0.15  with block = 20 (the horizon)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge_gate import block_bootstrap

DAYS = pd.bdate_range("1990-01-01", periods=9000)
SEEDS = range(60)
HORIZON = 20
P_NAIVE_MAX = 0.05
P_HONEST_MIN = 0.15


def noise_pair(seed: int):
    """Identical to demo.overlapping_windows, parameterised by seed."""
    rng = np.random.default_rng(seed)
    sig = pd.Series(rng.normal(size=len(DAYS))).rolling(60).mean().bfill()
    ret = pd.Series(rng.normal(size=len(DAYS))).rolling(HORIZON).mean().bfill()
    return sig.to_numpy(), ret.to_numpy()


def main() -> int:
    print("=" * 78)
    print("NOISE SWEEP - how often does pure noise pass a naive screen?")
    print("=" * 78)
    print(f"  construction : two independent white-noise series, smoothed")
    print(f"  seeds        : {SEEDS.start}-{SEEDS.stop - 1} ({len(SEEDS)} runs)")
    print(f"  artefact     : p < {P_NAIVE_MAX} at block=1 AND "
          f"p > {P_HONEST_MIN} at block={HORIZON}")
    print("")

    artefacts, naive_hits = [], []
    for seed in SEEDS:
        sig, ret = noise_pair(seed)
        legs = sig * ret
        p_naive = block_bootstrap(legs, block=1)
        p_honest = block_bootstrap(legs, block=HORIZON)
        naive = p_naive < P_NAIVE_MAX
        artefact = naive and p_honest > P_HONEST_MIN
        if naive:
            naive_hits.append(seed)
        if artefact:
            artefacts.append(seed)
        print(f"  seed {seed:>2}  p(block=1) = {p_naive:.4f}  "
              f"p(block={HORIZON}) = {p_honest:.4f}  "
              f"{'ARTEFACT' if artefact else ('naive-only' if naive else '-')}")

    n = len(SEEDS)
    print("")
    print("-" * 78)
    print(f"  significant under the NAIVE reading : {len(naive_hits)} of {n}")
    print(f"  of those, dissolved by the block    : {len(artefacts)} of {n}")
    print(f"  seeds : {artefacts}")
    print("-" * 78)
    print(f"  HEADLINE: the artefact appears in {len(artefacts)} of {n} runs "
          f"of pure noise.")
    print("")
    print("  Nothing in this sweep contains a relationship. Every hit above is")
    print("  a false positive produced by resampling overlapping windows one")
    print("  point at a time - the single most common way a backtest lies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
