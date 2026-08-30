"""Five tests every candidate edge has to survive before anyone writes code.

WHY THIS EXISTS

I built research probes one after another, and each one added the test the
previous one had been missing. The result was that every probe was judged by a
*weaker instrument than the next one*:

    positioning data (COT)      2 of 5 tests
    term structure              3 of 5
    commodity inventories       4 of 5
    credit spreads              5 of 5

The verdicts stopped being comparable. "COT failed" meant something different
from "credit failed". On top of that, `bootstrap_p` had been copy-pasted into
five files, so a fix in one never reached the others.

This module is the single copy. Every probe imports from here and goes through
the complete set.

WHAT EACH TEST CATCHES — and the failure that produced it.

1. `block_bootstrap` — overlapping windows faking significance.
   Positioning data: seven "significant" legs collapsed to zero once the
   bootstrap block matched the forecast horizon.

2. `benjamini_hochberg` — multiple testing.
   Positioning data again: 33 hypotheses, one p = 0.03 means nothing.

3. `partial_correlation` — a signal that is price wearing a hat.
   Term structure: contango appears *after* the price has already collapsed,
   so the "predictor" was partly the thing it claimed to predict.

4. `halves` — the aggregation artefact (Simpson's paradox).
   Inventories: the whole sample gave -0.211 at p = 0.0000, while BOTH halves
   were +0.10. That is not an edge, that is a level shift between periods.

5. `decades` — an effect that existed and has been arbitraged away.
   The 10Y-3M yield curve passed the four tests above and still decayed
   monotonically decade over decade. It is a historical result, not a signal.

Two rules that are not functions, because code cannot enforce them:

  * INSTRUMENT FIDELITY. Validate on the instrument you will actually trade.
    A CFD on an index tracks cash; a CFD on a commodity tracks the front
    future. Confusing the two produced one clean false positive here: a
    crypto signal that was strong on a perpetual swap and dead on the CFD.

  * INCONSISTENT SIGNS ARE THE SIGNATURE OF AN ARTEFACT, not evidence that
    markets differ. If half your legs point one way and half the other, you
    have found noise with a story attached.

A NOTE ON WHAT THIS IS NOT. This does not find edges. It kills them. In 60
runs of pure noise the naive reading calls 41 significant and the
overlapping-windows artefact survives in 29 - run examples/noise_sweep.py and
watch it happen. That is the point: the expensive mistake in systematic
trading is not missing an edge, it is deploying capital on one that was never
there. Every claim in this file is one you can reproduce from this repository;
anything I cannot hand you the command for has been left out on purpose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# False-discovery rate for Benjamini-Hochberg. 0.10 rather than 0.05 because
# this gate sits in front of *further research*, not in front of capital: the
# cost of a false positive here is a wasted week, not a drawdown.
Q_BH = 0.10

# A candidate whose most recent decade is below this fraction of its strongest
# decade is treated as expired.
DECAY_THRESHOLD = 0.4


def block_bootstrap(x, block: int = 1, n: int = 10_000, seed: int = 11) -> float:
    """Does the mean differ from zero — corrected for OVERLAPPING windows.

    With a forecast horizon h, sampling more often than h means consecutive
    observations share (h-1)/h of their window. Resampling single points
    treats them as independent and understates p by an order of magnitude.
    Setting block = horizon preserves that dependence.

    `block` is capped at a quarter of the sample: a longer block would leave
    too few independent pieces to conclude anything.

    p is computed as (hits + 1) / (n + 1), not as a bare proportion. A bare
    proportion over 10,000 draws can return exactly 0.0000 — and that is a
    claim of IMPOSSIBILITY, not a measurement.

    Two-sided by construction, which matters when reading the result: a low p
    on a NEGATIVE mean is a confirmed loser, not an edge.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return 1.0
    block = max(1, min(int(block), len(x) // 4))
    rng = np.random.default_rng(seed)
    mean = x.mean()
    centred = x - mean
    pieces = int(np.ceil(len(x) / block))
    starts = rng.integers(0, len(centred) - block + 1, size=(n, pieces))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)
    hits = int((np.abs(centred[idx[:, :len(x)]].mean(axis=1)) >= abs(mean)).sum())
    return (hits + 1) / (n + 1)


def benjamini_hochberg(p, q: float = Q_BH) -> list:
    """Which p-values survive correction for multiple testing.

    Computed across ALL hypotheses at once — never per horizon or per leg.
    Otherwise you can simply keep adding dimensions until something "works".
    """
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    keep = [False] * m
    for i, idx in enumerate(order, start=1):
        if p[idx] <= q * i / m:
            for j in order[:i]:
                keep[j] = True
    return keep


def partial_correlation(x, y, z) -> float:
    """Correlation of x with y after removing the linear influence of z.

    Used with z = trailing return: it answers whether the signal carries
    anything of its own, or is only a label meaning "price is low". A signal
    that is a copy of price loses almost all of its correlation here.

    NOT APPLICABLE to trend-following signals: a trend signal is by definition
    a function of trailing price, so this test would return ~0 and look like a
    rejection while measuring nothing. Say so out loud rather than pretending
    to run the full set.
    """
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    design = np.column_stack([np.ones(len(z)), z])
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def halves(signal, ret) -> dict:
    """Does the sign hold in BOTH halves of the sample.

    Catches the aggregation paradox: a correlation that exists in the whole
    sample and in neither half, because it is produced by a level shift
    between periods rather than by a relationship within them.

    Returns an empty dict when it cannot rule, and the caller must treat that
    as a REJECTION. Absence of evidence is not evidence.
    """
    signal, ret = np.asarray(signal, float), np.asarray(ret, float)
    n = len(signal)
    if n < 200:
        return {}
    out = {}
    for label, sl in (("all", slice(None)), ("first", slice(0, n // 2)),
                      ("second", slice(n // 2, None))):
        a, b = signal[sl], ret[sl]
        if len(a) < 80 or np.std(a) == 0 or np.std(b) == 0:
            continue
        out[label] = {"corr": float(np.corrcoef(a, b)[0, 1])}
    if {"all", "first", "second"} <= set(out):
        s = np.sign(out["all"]["corr"])
        out["consistent"] = bool(np.isfinite(s)
                                 and s == np.sign(out["first"]["corr"])
                                 and s == np.sign(out["second"]["corr"]))
    return out


def decades(signal, ret, dates, threshold: float = DECAY_THRESHOLD) -> dict:
    """Does the effect still WORK, or did it merely once work.

    Stronger than `halves`, because halves average a strong beginning against
    a weak end and report "consistent" under monotonic decay. A candidate
    whose latest decade falls below `threshold` of its strongest decade is a
    historical result, not a tradeable signal.

    Below three decades of data no ruling is made: the `expired` key simply
    does not appear, and the caller must not read its absence as a pass.
    """
    signal, ret = np.asarray(signal, float), np.asarray(ret, float)
    dates = pd.DatetimeIndex(dates)
    out = {}
    for start in range(1980, 2040, 10):
        m = (dates >= f"{start}-01-01") & (dates < f"{start + 10}-01-01")
        if m.sum() < 150:
            continue
        a, b = signal[m], ret[m]
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        high = b[a >= np.quantile(a, 0.8)]
        low = b[a <= np.quantile(a, 0.2)]
        out[f"{start}s"] = {"n": int(m.sum()),
                            "spread": float(high.mean() - low.mean()),
                            "corr": float(np.corrcoef(a, b)[0, 1])}
    decs = [k for k in out if k.endswith("s")]
    if len(decs) >= 3:
        latest = out[sorted(decs)[-1]]
        strongest = max(abs(out[k]["corr"]) for k in decs)
        out["expired"] = bool(strongest > 0
                              and abs(latest["corr"]) < threshold * strongest)
    return out


def verdict(results: list, q: float = Q_BH) -> dict:
    """The single place a ruling is made — identically for every probe.

    A candidate passes ONLY on the complete set:
      * survives Benjamini-Hochberg across all hypotheses,
      * has a consistent sign in both halves of the sample,
      * is not expired (where the sample allows that to be ruled on).

    Each result is a dict with at least `p`; optionally `halves` and `decades`.
    Returns lists of names: `passed`, `inconsistent`, `expired`.
    """
    keep = benjamini_hochberg([r.get("p", 1.0) for r in results], q=q)
    passed, inconsistent, expired = [], [], []
    for r, k in zip(results, keep):
        name = r.get("name") or "?"
        if not k:
            continue
        if not (r.get("halves") or {}).get("consistent"):
            inconsistent.append(name)
        elif (r.get("decades") or {}).get("expired"):
            expired.append(name)
        else:
            passed.append(name)
    return {"passed": passed, "inconsistent": inconsistent, "expired": expired,
            "significant_raw": [r.get("name") or "?"
                                for r, k in zip(results, keep) if k]}


def describe(v: dict) -> str:
    """Render a verdict so the reasons for rejection are not lost."""
    lines = []
    if v["inconsistent"]:
        lines.append("  REJECTED - sign flips between halves (aggregation "
                     "artefact): " + ", ".join(v["inconsistent"]))
    if v["expired"]:
        lines.append("  EXPIRED - the effect existed but the latest decade is "
                     "near zero (historical, not tradeable): "
                     + ", ".join(v["expired"]))
    lines.append("  SIGNIFICANT: "
                 + (", ".join(v["passed"]) if v["passed"] else "NONE"))
    return "\n".join(lines)
