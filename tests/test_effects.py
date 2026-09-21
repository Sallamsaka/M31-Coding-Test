"""Behavioural tests for the paired main-effects estimator.

These are cheap and exact, which is the point: `src/effects.py` is the only part
of the experimental machinery whose correctness can be established without
spending CPU-hours, so it is the part that gets real tests. Everything asserted
here is a property of the estimator on data whose truth is known by construction.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src.effects import paired_effects, shrink

FACTORS = ["a", "b", "c"]


def _table(true: dict[str, float], noise: float = 0.0, seed: int = 0,
           folds=(0.010, -0.008, 0.003, -0.002, 0.005)) -> pd.DataFrame:
    """Full 2^3 factorial x 5 folds with known additive effects."""
    rng = np.random.default_rng(seed)
    rows = []
    for combo in itertools.product([False, True], repeat=len(FACTORS)):
        cfg = dict(zip(FACTORS, combo))
        base = 0.75 + sum(true[k] * cfg[k] for k in true)
        for f, fe in enumerate(folds):
            rows.append({**cfg, "fold": f,
                         "auroc": base + fe + (rng.normal(0, noise) if noise else 0.0)})
    return pd.DataFrame(rows)


def test_recovers_additive_effects_exactly_without_noise():
    true = {"a": 0.004, "b": -0.002, "c": 0.0}
    res = paired_effects(_table(true), FACTORS, "auroc").set_index("factor")
    for f, want in true.items():
        assert abs(res.loc[f, "effect"] - want) < 1e-12, f"{f}: {res.loc[f,'effect']}"


def test_shared_fold_effects_cancel_in_the_contrast():
    """A huge common fold offset must not move any effect estimate.

    This is the property that makes pairing worth having: fold difficulty is
    common to both levels of every factor, so it must subtract out entirely.
    """
    true = {"a": 0.004, "b": 0.0, "c": 0.0}
    mild = paired_effects(_table(true, folds=(0.0,) * 5), FACTORS, "auroc")
    wild = paired_effects(_table(true, folds=(0.5, -0.4, 0.3, -0.2, 0.1)),
                          FACTORS, "auroc")
    a = mild.set_index("factor").effect
    b = wild.set_index("factor").effect
    assert np.allclose(a.to_numpy(), b.reindex(a.index).to_numpy(), atol=1e-12)


def test_paired_se_is_tighter_than_the_unpaired_two_sample_se():
    """The defect this module was written to fix, asserted as a behaviour.

    Non-vacuity is built in: the unpaired SE must actually be larger, not merely
    different, and the other factors must genuinely contribute variance -- which
    is why `b` and `c` carry real effects here.
    """
    true = {"a": 0.004, "b": 0.006, "c": 0.005}
    df = _table(true, noise=0.003, seed=1)
    res = paired_effects(df, FACTORS, "auroc").set_index("factor")
    per = df.groupby(FACTORS)["auroc"].mean().reset_index()
    for f in FACTORS:
        hi, lo = per.loc[per[f], "auroc"], per.loc[~per[f], "auroc"]
        unpaired = np.sqrt(hi.var(ddof=1) / len(hi) + lo.var(ddof=1) / len(lo))
        assert unpaired > res.loc[f, "se"], f"{f}: pairing did not tighten the SE"


def test_shrinkage_collapses_to_zero_when_spread_is_pure_noise():
    eff = np.array([0.001, -0.0008, 0.0005, -0.0011, 0.0009])
    assert np.abs(shrink(eff, np.full(5, 0.02))).max() == 0.0


def test_shrinkage_is_a_contraction_toward_zero():
    """Shrunken estimates never exceed the raw ones, and never flip sign."""
    eff = np.array([0.02, -0.015, 0.008, -0.004, 0.001])
    se = np.full(5, 0.002)
    sh = shrink(eff, se)
    assert np.all(np.abs(sh) <= np.abs(eff) + 1e-15)
    assert np.all(np.sign(sh) * np.sign(eff) >= 0)


def test_constant_factor_is_skipped_not_crashed():
    df = _table({"a": 0.004, "b": 0.0, "c": 0.0})
    df["c"] = True                       # no longer varied
    res = paired_effects(df, FACTORS, "auroc")
    assert set(res.factor) == {"a", "b"}


def test_bh_q_values_are_monotone_and_bounded():
    df = _table({"a": 0.006, "b": 0.003, "c": 0.0}, noise=0.003, seed=2)
    res = paired_effects(df, FACTORS, "auroc")
    q = res.q_bh.to_numpy()
    assert np.all((q >= 0) & (q <= 1))
    assert np.all(np.diff(q) >= -1e-12), "BH q-values must be non-decreasing in p"


def test_effect_sign_follows_the_truth():
    res = paired_effects(_table({"a": 0.01, "b": -0.01, "c": 0.0}, noise=0.001,
                                seed=3), FACTORS, "auroc").set_index("factor")
    assert res.loc["a", "effect"] > 0 and res.loc["b", "effect"] < 0
