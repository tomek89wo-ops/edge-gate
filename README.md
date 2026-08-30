# edge-gate

**Five tests every candidate trading edge has to survive before anyone writes code.**

This does not find edges. It kills them. Over one research campaign it took
roughly **620 candidate hypotheses and passed one** — and that ratio is the
point. The expensive mistake in systematic trading is not missing an edge; it
is putting capital behind one that was never there.

```
python examples/demo.py
```

```
NAIVE READING — one p-value per candidate, no block, no correction
  overlapping_windows    p = 0.0001   SIGNIFICANT
  price_in_disguise      p = 0.0001   SIGNIFICANT
  arbitraged_away        p = 0.0001   SIGNIFICANT
  genuine                p = 0.0001   SIGNIFICANT

  A naive pipeline promotes: all four

THE GATE — block = horizon, BH across all candidates, halves, decades
  candidate               p (block)    BH     halves    decades
  overlapping_windows        0.2054  drop consistent       live
  price_in_disguise          0.0001  keep      FLIPS       live
  arbitraged_away            0.0001  keep consistent    EXPIRED
  genuine                    0.0001  keep consistent       live

  REJECTED - sign flips between halves (aggregation artefact): price_in_disguise
  EXPIRED  - the effect existed but the latest decade is near zero: arbitraged_away
  SIGNIFICANT: genuine
```

Four candidates. All four clear a naive screen. One clears the gate.

---

## Why it exists

I built research probes one after another, and each added the test the previous
one had been missing. So every probe was judged by a *weaker instrument than
the next one*:

| probe | tests it faced |
|---|---|
| positioning data (COT) | 2 of 5 |
| term structure | 3 of 5 |
| commodity inventories | 4 of 5 |
| credit spreads | 5 of 5 |

The verdicts stopped being comparable — "COT failed" meant something different
from "credit failed". Worse, `bootstrap_p` had been copy-pasted into five
files, so a fix in one never reached the others.

This module is the single copy. Every probe imports from here and faces the
complete set.

---

## The five tests, and the failure that produced each

### 1. `block_bootstrap` — overlapping windows faking significance

With a forecast horizon of *h* days sampled daily, consecutive observations
share (h−1)/h of their window. Resampling single points treats them as
independent and understates p by an order of magnitude.

**How bad is it?** Sweeping 60 seeds of *pure noise* through the demo's
construction — two independent white-noise series, no relationship whatsoever —
**41 of 60 runs come out significant** at p < 0.05 when you bootstrap single
points. Block correctly at the horizon and **29 of those 60 dissolve**. Two
thirds of pure noise looks like a discovery under the naive reading, and
roughly half of everything you would have promoted was the resampling scheme
talking, not the data.

Do not take that on my word — it is the one number this whole repository rests
on, so it ships as a script rather than a sentence:

```
python examples/noise_sweep.py
```

It prints every seed, both p-values, and the count.

Set `block = horizon`. Always.

Two details that matter:

- p is `(hits + 1) / (n + 1)`, never a bare proportion. A bare proportion over
  10,000 draws can return exactly `0.0000`, and that is a claim of
  **impossibility**, not a measurement.
- The test is **two-sided**. A low p on a *negative* mean is a confirmed loser,
  not an edge. Read the sign before you celebrate the p.

### 2. `benjamini_hochberg` — multiple testing

33 hypotheses, one `p = 0.03`, no discovery. Applied across **all** hypotheses
at once — never per horizon or per leg, because then you can add dimensions
until something "works".

### 3. `partial_correlation` — a signal that is price wearing a hat

Run with `z = trailing return`. It asks whether the signal carries anything of
its own, or is only a label meaning "price is low". A signal that is a copy of
price loses nearly all of its correlation here.

**Not applicable to trend-following.** A trend signal is by definition a
function of trailing price, so this test would return ~0 and look like a
rejection while measuring nothing. Say "four of five, the fifth does not apply"
out loud instead of quietly reporting a full set.

### 4. `halves` — the aggregation artefact

The one that hurts, because it looks like the strongest result you have.
A commodity-inventory series gave **−0.211 at p = 0.0000** over the full
sample, and **+0.10 in both halves**. That is Simpson's paradox: a level shift
between two periods, not a relationship inside either.

Returns `{}` when it cannot rule, and the caller **must treat that as a
rejection**. Absence of evidence is not evidence.

### 5. `decades` — an effect that has been arbitraged away

Stronger than `halves`, because halves average a strong beginning against a
weak end and report "consistent" under monotonic decay.

The 10Y-3M yield curve passed significance, multiple testing, orthogonality
*and* the split-half test — and had nothing left in its most recent decade. A
historical result, not a tradeable signal.

Below three decades of data, no ruling is made and the `expired` key simply
does not appear. Do not read its absence as a pass.

---

## Two rules code cannot enforce

**Instrument fidelity.** Validate on the instrument you will actually trade. A
CFD on an index tracks cash; a CFD on a commodity tracks the front future.
Confusing the two produced one clean false positive here: a crypto signal that
was strong on a perpetual swap and dead on the CFD that could actually be
bought. A term-structure signal survived all five tests and then turned out to
be unbuyable, because the instrument on offer *was* the future whose curve was
the signal.

**Inconsistent signs are the signature of an artefact**, not evidence that
markets differ. If half your legs point one way and half the other, you have
found noise with a story attached.

---

## Usage

```python
from edge_gate import block_bootstrap, halves, decades, verdict, describe

results = []
for name, (signal, forward_return, dates) in candidates.items():
    results.append({
        "name": name,
        "p": block_bootstrap(signal * forward_return, block=HORIZON_DAYS),
        "halves": halves(signal, forward_return),
        "decades": decades(signal, forward_return, dates),
    })

print(describe(verdict(results)))   # BH is applied inside verdict()
```

A candidate passes **only on the complete set**: survives Benjamini-Hochberg
across all hypotheses, keeps a consistent sign in both halves, and is not
expired.

---

## Install

```bash
pip install -r requirements.txt      # numpy, pandas
python -m pytest tests/ -q           # 28 tests
```

No packaging, no CLI, no configuration. It is one module you copy into a
research repo and import.

---

## A note on the test suite

One of the tests in here originally asserted that a single draw of pure noise
returns `p > 0.10`. It failed on the second seed I tried, at `p = 0.089` —
which is not a bug, it is the definition of a 10% level. The test now pins the
**false-positive rate across 40 draws** instead of the outcome of one.

That failure is the argument for this whole module in miniature: one sample
that looks significant is exactly what an untested pipeline reports as a
discovery.

---

## Licence

MIT. Take it, copy it into your own research code, change it.
