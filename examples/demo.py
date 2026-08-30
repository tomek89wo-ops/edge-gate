"""Four candidates. A naive screen promotes all four. The gate keeps one.

Run:  python examples/demo.py

Every fake below is modelled on a real rejection, so the failure modes are the
ones you will actually meet, not textbook curiosities.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge_gate import (benjamini_hochberg, block_bootstrap, decades, describe,
                       halves, verdict)

RNG = np.random.default_rng(20260826)
DAYS = pd.bdate_range("1990-01-01", periods=9000)


def overlapping_windows():
    """Pure noise that looks significant because the windows overlap.

    A slow signal against a 20-day forward return, both sampled daily. There is
    no relationship here whatsoever - both series are white noise passed
    through a moving average. The apparent significance comes entirely from
    resampling overlapping observations as if they were independent.

    The seed is fixed at 4 so the demo is reproducible, but this is NOT a
    cherry-picked freak: sweeping seeds 0-59 with identical construction, 41
    of 60 runs are significant at block=1, and the artefact (p < 0.05 at
    block=1, p > 0.15 at block=20) appears in 29 of 60. Run it yourself with
    `python examples/noise_sweep.py` - the claim ships as a script, not as
    this sentence.
    """
    rng = np.random.default_rng(4)
    sig = pd.Series(rng.normal(size=len(DAYS))).rolling(60).mean().bfill()
    ret = pd.Series(rng.normal(size=len(DAYS))).rolling(20).mean().bfill()
    return sig.to_numpy(), ret.to_numpy()


def price_in_disguise():
    """Simpson's paradox: significant overall, opposite sign in both halves.

    Modelled on a commodity-inventory series where the whole sample gave
    -0.211 at p = 0.0000 and both halves were +0.10. The relationship is a
    level shift between two periods, not a relationship inside either.
    """
    n = len(DAYS)
    half = n // 2
    sig = np.concatenate([RNG.normal(0.0, 1.0, half) - 2.0,
                          RNG.normal(0.0, 1.0, n - half) + 2.0])
    ret = np.concatenate([0.15 * RNG.normal(size=half) + 0.04,
                          0.15 * RNG.normal(size=n - half) - 0.04])
    ret = ret + 0.02 * (sig - sig.mean()) / sig.std()
    return sig, ret


def arbitraged_away():
    """Real once, gone now. Strength decays monotonically decade over decade.

    Modelled on the 10Y-3M yield curve, which passed significance, multiple
    testing, orthogonality AND the split-half test, and still had nothing left
    in its most recent decade.
    """
    sig = RNG.normal(size=len(DAYS))
    strength = np.interp(np.arange(len(DAYS)), [0, len(DAYS) - 1], [0.55, 0.01])
    ret = strength * sig + np.sqrt(np.clip(1 - strength ** 2, 0, 1)) * \
        RNG.normal(size=len(DAYS))
    return sig, ret * 0.01


def genuine():
    """A modest, stable relationship that holds throughout the sample."""
    sig = RNG.normal(size=len(DAYS))
    ret = 0.10 * sig + RNG.normal(0, 0.5, len(DAYS))
    return sig, ret * 0.01


CANDIDATES = {
    "overlapping_windows": (overlapping_windows, 20),
    "price_in_disguise": (price_in_disguise, 1),
    "arbitraged_away": (arbitraged_away, 1),
    "genuine": (genuine, 1),
}


def main() -> int:
    rows, results = [], []
    for name, (build, horizon) in CANDIDATES.items():
        sig, ret = build()
        legs = sig * ret                      # per-day contribution of the leg
        p_naive = block_bootstrap(legs, block=1)
        p_honest = block_bootstrap(legs, block=horizon)
        h = halves(sig, ret)
        d = decades(sig, ret, DAYS)
        results.append({"name": name, "p": p_honest, "halves": h, "decades": d})
        rows.append((name, p_naive, p_honest, h.get("consistent"),
                     d.get("expired")))

    print("=" * 78)
    print("NAIVE READING — one p-value per candidate, no block, no correction")
    print("=" * 78)
    naive_keep = [n for n, pn, _, _, _ in rows if pn < 0.05]
    for n, pn, _, _, _ in rows:
        print(f"  {n:<22} p = {pn:.4f}   {'SIGNIFICANT' if pn < 0.05 else '-'}")
    print(f"\n  A naive pipeline promotes: {', '.join(naive_keep) or 'none'}")

    print("")
    print("=" * 78)
    print("THE GATE — block = horizon, BH across all candidates, halves, decades")
    print("=" * 78)
    print(f"  {'candidate':<22} {'p (block)':>10} {'BH':>5} {'halves':>10} "
          f"{'decades':>10}")
    keep = benjamini_hochberg([r["p"] for r in results])
    for (n, _, ph, cons, exp), k in zip(rows, keep):
        print(f"  {n:<22} {ph:>10.4f} {'keep' if k else 'drop':>5} "
              f"{('consistent' if cons else 'FLIPS') if cons is not None else 'n/a':>10} "
              f"{('EXPIRED' if exp else 'live') if exp is not None else 'n/a':>10}")
    print("")
    print(describe(verdict(results)))
    print("")
    print("  All four survive a naive screen. One survives the gate.")
    print("  Each rejection has a NAMED reason, which is the point: 'it did not")
    print("  work' is not actionable, 'the sign flips between halves' is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
