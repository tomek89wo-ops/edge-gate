"""Tests for the gate itself.

Probe-specific tests guard a probe's own mechanics — publication lag, signal
construction, instrument fidelity. This file guards the INSTRUMENT, and above
all `verdict()`: the function that folds five tests into one ruling and decides
whether anything gets built.

Every case below is a real failure from the research campaign that produced
this module, with names generalised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edge_gate import (DECAY_THRESHOLD, benjamini_hochberg, block_bootstrap,
                       decades, describe, halves, partial_correlation, verdict)


def _consistent():
    return {"all": {"corr": -0.2}, "first": {"corr": -0.18},
            "second": {"corr": -0.21}, "consistent": True}


# ── verdict: the complete set, not any single test ──────────────────────────

def test_candidate_must_pass_the_COMPLETE_set():
    """Significance alone is not enough. Each of the three conditions killed a
    candidate on its own during the campaign this came from."""
    v = verdict([{"name": "good", "p": 0.0001, "halves": _consistent(),
                  "decades": {"expired": False}}])
    assert v["passed"] == ["good"]


def test_inconsistent_sign_REJECTS_even_at_p_zero():
    """Commodity inventories: p = 0.0000 over the whole sample, and both
    halves carried the opposite sign. That is Simpson's paradox, not an edge."""
    v = verdict([{"name": "inventories", "p": 0.0000,
                  "halves": {"all": {"corr": -0.21}, "first": {"corr": 0.10},
                             "second": {"corr": 0.08}, "consistent": False},
                  "decades": {"expired": False}}])
    assert v["passed"] == []
    assert v["inconsistent"] == ["inventories"]


def test_expired_REJECTS_even_with_a_consistent_sign():
    """The 10Y-3M curve: p = 0.0047, sign consistent in both halves, and the
    effect decayed monotonically decade over decade."""
    v = verdict([{"name": "curve", "p": 0.0047, "halves": _consistent(),
                  "decades": {"expired": True}}])
    assert v["passed"] == []
    assert v["expired"] == ["curve"]


def test_missing_halves_is_a_REJECTION_not_a_pass():
    """`halves` returns {} when it cannot rule. Reading that as a pass is how
    a short sample gets promoted to an edge."""
    v = verdict([{"name": "short", "p": 0.0001, "halves": {}, "decades": {}}])
    assert v["passed"] == [] and v["inconsistent"] == ["short"]


def test_multiple_testing_is_applied_ACROSS_hypotheses():
    """One p = 0.03 out of 33 hypotheses means nothing. If BH were applied per
    leg you could add dimensions until something 'worked'."""
    results = [{"name": f"h{i}", "p": 0.03 if i == 0 else 0.6,
                "halves": _consistent(), "decades": {"expired": False}}
               for i in range(33)]
    assert verdict(results)["passed"] == []


def test_a_genuinely_strong_result_still_survives_BH():
    """The gate has to be able to say yes, or it is not a gate, it is a wall."""
    results = [{"name": f"h{i}", "p": 0.0001 if i < 5 else 0.6,
                "halves": _consistent(), "decades": {"expired": False}}
               for i in range(33)]
    assert len(verdict(results)["passed"]) == 5


# ── block bootstrap ─────────────────────────────────────────────────────────

def test_p_is_never_exactly_zero():
    """A bare proportion over 10,000 draws can return 0.0000, which is a claim
    of impossibility rather than a measurement. (hits+1)/(n+1) cannot."""
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 0.1, 500)          # overwhelmingly non-zero mean
    assert block_bootstrap(x, block=1) > 0.0


def test_false_positive_rate_is_CALIBRATED_not_zero():
    """The first version of this test asserted that one draw of pure noise
    gives p > 0.10. It failed on the second seed I tried, with p = 0.089 —
    which is not a bug, it is the definition of a 10% level. A single draw
    proves nothing about a p-value; the property worth pinning is the RATE.

    That failure is the whole argument for this module in miniature: one
    sample that looks significant is what an untested pipeline reports as a
    discovery."""
    ps = [block_bootstrap(np.random.default_rng(s).normal(0, 1, 800), block=1)
          for s in range(40)]
    rate = sum(p < 0.10 for p in ps) / len(ps)
    assert rate <= 0.25, f"false positives {rate:.0%} — far above the 10% level"
    assert float(np.median(ps)) > 0.30, "p-values are not roughly uniform"


def test_overlapping_windows_INFLATE_significance_at_block_one():
    """The failure this test exists for: with a 20-day horizon sampled daily,
    consecutive observations share 19/20 of their window. Block=1 treats them
    as independent and reports a p an order of magnitude too small."""
    rng = np.random.default_rng(7)
    daily = rng.normal(0.0, 1.0, 3000)
    overlapping = pd.Series(daily).rolling(20).mean().dropna().to_numpy()
    p_naive = block_bootstrap(overlapping, block=1)
    p_honest = block_bootstrap(overlapping, block=20)
    assert p_honest > p_naive, "block=horizon must not be more permissive"


def test_block_is_capped_at_a_quarter_of_the_sample():
    """A longer block leaves too few independent pieces to conclude anything."""
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 200)
    assert block_bootstrap(x, block=500) == block_bootstrap(x, block=50)


def test_too_few_observations_returns_one_not_a_guess():
    assert block_bootstrap(np.arange(10.0), block=1) == 1.0


def test_nans_are_dropped_not_propagated():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, 400)
    y = x.copy()
    y[::17] = np.nan
    assert np.isfinite(block_bootstrap(y, block=5))


def test_result_is_reproducible_for_a_given_seed():
    rng = np.random.default_rng(9)
    x = rng.normal(0.05, 1, 600)
    assert block_bootstrap(x, block=5) == block_bootstrap(x, block=5)


def test_bootstrap_is_TWO_SIDED():
    """A low p on a negative mean is a confirmed loser, not an edge. The test
    pins that the function does not quietly favour one direction."""
    rng = np.random.default_rng(11)
    base = rng.normal(0.0, 1.0, 800)
    up = block_bootstrap(base + 0.5, block=1)
    down = block_bootstrap(base - 0.5, block=1)
    assert up < 0.05 and down < 0.05


# ── Benjamini-Hochberg ──────────────────────────────────────────────────────

def test_bh_keeps_the_strong_and_drops_the_marginal():
    keep = benjamini_hochberg([0.001, 0.04, 0.5, 0.9], q=0.10)
    assert keep[0] is True and keep[2] is False and keep[3] is False


def test_bh_is_step_up_not_step_down():
    """A large p later in the ranking must not un-keep a small p before it."""
    assert benjamini_hochberg([0.001, 0.02, 0.03], q=0.10) == [True] * 3


def test_bh_on_an_empty_list_returns_empty():
    assert benjamini_hochberg([]) == []


# ── partial correlation ─────────────────────────────────────────────────────

def test_a_signal_that_is_price_in_disguise_loses_its_correlation():
    """Term structure: contango appears after the price has already collapsed,
    so part of the 'prediction' was the thing being predicted."""
    rng = np.random.default_rng(5)
    trailing = rng.normal(size=800)
    forward = 0.6 * trailing + rng.normal(0, 0.4, 800)
    disguised = trailing.copy()                      # signal IS the price move
    raw = float(np.corrcoef(disguised, forward)[0, 1])
    net = partial_correlation(disguised, forward, trailing)
    assert abs(raw) > 0.5 and abs(net) < 0.05


def test_an_independent_signal_KEEPS_its_correlation():
    rng = np.random.default_rng(6)
    trailing = rng.normal(size=800)
    own = rng.normal(size=800)
    forward = 0.5 * own + 0.3 * trailing + rng.normal(0, 0.3, 800)
    assert abs(partial_correlation(own, forward, trailing)) > 0.4


# ── halves ──────────────────────────────────────────────────────────────────

def test_halves_refuses_to_rule_on_a_short_sample():
    rng = np.random.default_rng(8)
    assert halves(rng.normal(size=100), rng.normal(size=100)) == {}


def test_a_sign_flip_between_halves_is_reported():
    n = 600
    sig = np.concatenate([np.linspace(-1, 1, n // 2), np.linspace(-1, 1, n // 2)])
    ret = np.concatenate([sig[:n // 2], -sig[n // 2:]])
    out = halves(sig, ret)
    assert out["consistent"] is False


def test_a_stable_relationship_is_reported_consistent():
    rng = np.random.default_rng(10)
    sig = rng.normal(size=600)
    ret = 0.4 * sig + rng.normal(0, 0.5, 600)
    assert halves(sig, ret)["consistent"] is True


# ── decades ─────────────────────────────────────────────────────────────────

def _series(per_decade_corr, start=1990):
    """Build a signal/return pair whose correlation per decade is prescribed."""
    rng = np.random.default_rng(12)
    sig, ret, dates = [], [], []
    for i, c in enumerate(per_decade_corr):
        idx = pd.bdate_range(f"{start + 10 * i}-01-01", periods=300)
        s = rng.normal(size=len(idx))
        r = c * s + np.sqrt(max(0.0, 1 - c * c)) * rng.normal(size=len(idx))
        sig.append(s); ret.append(r); dates.append(idx)
    return (np.concatenate(sig), np.concatenate(ret),
            pd.DatetimeIndex(np.concatenate(dates)))


def test_a_decaying_effect_is_marked_expired():
    """Strong, then weaker, then gone. Halves would average this and call it
    consistent; decades will not."""
    sig, ret, dates = _series([0.55, 0.30, 0.02])
    assert decades(sig, ret, dates)["expired"] is True


def test_a_stable_effect_is_not_marked_expired():
    sig, ret, dates = _series([0.40, 0.42, 0.38])
    assert decades(sig, ret, dates)["expired"] is False


def test_below_three_decades_NO_ruling_is_made():
    """And the caller must not read the missing key as a pass."""
    sig, ret, dates = _series([0.4, 0.4])
    assert "expired" not in decades(sig, ret, dates)


def test_decay_threshold_is_the_documented_fraction():
    assert DECAY_THRESHOLD == 0.4


# ── the rendered verdict has to carry the REASON ────────────────────────────

def test_describe_names_every_rejection_reason():
    v = verdict([
        {"name": "flips", "p": 0.0001,
         "halves": {"all": {"corr": 1}, "first": {"corr": 1},
                    "second": {"corr": -1}, "consistent": False},
         "decades": {"expired": False}},
        {"name": "gone", "p": 0.0001, "halves": _consistent(),
         "decades": {"expired": True}},
    ])
    text = describe(v)
    assert "flips" in text and "gone" in text
    assert "aggregation artefact" in text and "historical" in text


def test_describe_says_NONE_rather_than_going_quiet():
    """An empty list rendered as blank space reads like a pass."""
    assert "NONE" in describe(verdict([]))
