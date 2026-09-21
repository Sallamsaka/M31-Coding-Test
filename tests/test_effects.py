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

from src.effects import orthogonal_effects, paired_effects, shrink

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


# --------------------------------------------------------------------------
# Fractional designs need `orthogonal_effects`, not `paired_effects`.
# --------------------------------------------------------------------------

def _res_v():
    """2^(5-1), E = ABCD, as src.design builds it."""
    base = np.array(list(itertools.product([-1, 1], repeat=4)))
    d = np.hstack([base, base.prod(axis=1, keepdims=True)])
    cols = [f"c_{i}" for i in range(5)]
    return pd.DataFrame(d, columns=cols), cols


def test_pairing_returns_nothing_on_a_fractional_design():
    """The bug this split was written for, pinned so it cannot silently return.

    In a 2^(5-1) fraction the fifth factor is determined by the other four, so
    no run has a twin matching on everything else. `paired_effects` must come
    back empty rather than quietly contrasting mismatched runs.
    """
    df, cols = _res_v()
    df["y"] = np.arange(len(df), dtype=float)
    df["fold"] = 0
    assert paired_effects(df, cols, "y", fold_col="fold").empty


def test_orthogonal_effects_recovers_truth_exactly_without_noise():
    df, cols = _res_v()
    truth = {"c_0": 0.02, "c_3": -0.01}
    df["y"] = 0.5 + sum(v * df[k] for k, v in truth.items())
    res = orthogonal_effects(df, cols, "y").set_index("factor")
    for c in cols:
        want = 2 * truth.get(c, 0.0)      # effect spans -1 -> +1
        assert abs(res.loc[c, "effect"] - want) < 1e-12, c


def test_orthogonal_se_matches_two_sigma_over_root_n():
    """SE(effect) = 2*sigma/sqrt(N) -- the identity E14 rests on."""
    df, cols = _res_v()
    rng = np.random.default_rng(0)
    sigma, N = 0.01, len(df)
    est = []
    for _ in range(400):
        df["y"] = rng.normal(0, sigma, N)
        est.append(orthogonal_effects(df, cols, "y").set_index("factor")
                   .loc["c_0", "effect"])
    emp = np.std(est, ddof=1)
    assert abs(emp - 2 * sigma / np.sqrt(N)) < 0.0004, emp


def test_lenth_pse_is_robust_to_one_dominant_effect():
    """Lenth trims large effects before the second median, so a single huge
    factor must not inflate the noise estimate and mask the rest."""
    df, cols = _res_v()
    quiet = 0.5 + 0.001 * df["c_1"]
    res_small = orthogonal_effects(df.assign(y=quiet), cols, "y")
    res_big = orthogonal_effects(df.assign(y=quiet + 0.5 * df["c_0"]), cols, "y")
    pse_small = float(res_small.lenth_pse.iloc[0])
    pse_big = float(res_big.lenth_pse.iloc[0])
    assert abs(pse_big - pse_small) < 1e-9, (pse_small, pse_big)


def test_orthogonal_effects_ranks_by_magnitude():
    df, cols = _res_v()
    df["y"] = 0.5 + 0.02 * df["c_2"] + 0.005 * df["c_4"]
    res = orthogonal_effects(df, cols, "y")
    assert res.factor.iloc[0] == "c_2" and res.factor.iloc[1] == "c_4"
