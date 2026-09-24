"""Behavioural tests for per-label Platt scaling.

These assert what the transform DOES -- ranking preserved, floor preserved,
non-monotone labels refused -- rather than that some flag is set. Two of them
are positive-controlled: the test states the mutation that must turn it red.
"""
import numpy as np
import pytest

from src.calibrate import apply_platt, fit_platt, murphy, verify_noop


def _synthetic(n=600, n_lab=8, seed=0):
    """Scores that are informative but badly calibrated, plus an at-risk mask."""
    rng = np.random.default_rng(seed)
    y = np.zeros((n, n_lab), int)
    P = np.zeros((n, n_lab), float)
    at_risk = rng.random((n, n_lab)) < 0.85
    for j in range(n_lab):
        z = rng.normal(size=n)
        p_true = 1.0 / (1.0 + np.exp(-(z - 2.0)))       # ~12% positive
        y[:, j] = (rng.random(n) < p_true).astype(int)
        # A monotone but WRONG map: informative ranking, miscalibrated level.
        P[:, j] = np.clip(1.0 / (1.0 + np.exp(-(0.4 * z + 1.5))), 1e-6, 1 - 1e-6)
    return P, y, at_risk


def test_platt_leaves_macro_ap_and_auroc_exactly_unchanged():
    """The whole method rests on this being a rank no-op. Check it numerically.

    Holds for every label whose fitted slope is positive, which is every label
    in this synthetic set. Negative slopes are applied deliberately and tested
    separately (test_negative_slope_is_applied_and_reverses_the_ranking_by_default).
    """
    P, y, ar = _synthetic()
    pp = fit_platt(P, y, ar)
    assert pp.n_calibrated > 0, "nothing was calibrated -- test is vacuous"
    Q = apply_platt(P, pp, at_risk=ar)
    verify_noop(P, Q, y, ar)          # asserts internally to 1e-9


def test_platt_improves_a_proper_scoring_rule():
    """Monotone transform cannot help the rank metrics, so it must earn its
    place on a proper one. If Brier does not improve, there is nothing to buy."""
    P, y, ar = _synthetic()
    pp = fit_platt(P, y, ar)
    Q = apply_platt(P, pp, at_risk=ar)
    before, after = murphy(P, y, ar), murphy(Q, y, ar)
    assert after["brier"] < before["brier"], (
        f"Brier did not improve: {before['brier']:.5f} -> {after['brier']:.5f}")
    assert after["rel"] < before["rel"], "reliability did not improve"


def _overconfident(n=600, n_lab=6, seed=1):
    """Scores that are too SPREAD OUT in logit space, so Platt fits a < 1.

    This regime is what the floor bug needs and what the first version of this
    test lacked. With a > 1 the map pushes logit(1e-6) = -13.8 even further
    negative and the floor survives by accident; only a shrinking map (a < 1)
    lifts a floored pair back into the ranking.
    """
    rng = np.random.default_rng(seed)
    y = np.zeros((n, n_lab), int)
    P = np.zeros((n, n_lab), float)
    at_risk = rng.random((n, n_lab)) < 0.85
    for j in range(n_lab):
        z = rng.normal(size=n)
        p_true = 1.0 / (1.0 + np.exp(-(z - 1.5)))
        y[:, j] = (rng.random(n) < p_true).astype(int)
        # 4x too confident: the Platt slope should come back near 1/4.
        P[:, j] = np.clip(1.0 / (1.0 + np.exp(-(4.0 * z - 6.0))), 1e-6, 1 - 1e-6)
    return P, y, at_risk


def test_overconfident_scores_really_do_fit_a_shallow_slope():
    """Guards the guard: if this data stopped producing a < 1, the floor test
    below would silently become vacuous again."""
    P, y, ar = _overconfident()
    pp = fit_platt(P, y, ar)
    assert pp.n_calibrated > 0
    assert pp.a[pp.ok].max() < 1.0, (
        f"expected shrinking maps, got slopes up to {pp.a[pp.ok].max():.3f} -- "
        "the floor test is no longer exercising the failure mode")


def test_not_at_risk_pairs_stay_floored():
    """The bug this was written for: logit(1e-6) = -13.8, and sigmoid(a*-13.8+b)
    is NOT small for every fitted (a, b), so calibrating a masked matrix can
    lift not-at-risk pairs back into the ranking.

    verify_noop cannot see this -- macro AP/AUROC are computed with the at-risk
    mask applied, so the damaged pairs are excluded from the check.

    Positive control: drop the `at_risk=` argument below and this goes red.
    """
    P, y, ar = _overconfident()        # a < 1; _synthetic gives a > 1 and is VACUOUS here
    P = np.where(ar, P, 1e-6)          # what apply_at_risk_mask does
    pp = fit_platt(P, y, ar)
    Q = apply_platt(P, pp, at_risk=ar)
    lifted = Q[~ar]
    assert lifted.max() <= 1e-6 + 1e-12, (
        f"not-at-risk pair lifted to {lifted.max():.4g} -- floor not preserved")

    # Positive control, run inline rather than asserted in prose: without the
    # floor these pairs MUST rise. The first version of this test used data
    # that fit a > 1, where they did not, and the control stayed green.
    lifted_bad = apply_platt(P, pp)[~ar]
    assert lifted_bad.max() > 1e-3, (
        f"control did not fire: unfloored max is {lifted_bad.max():.4g}, so "
        "this test cannot detect the bug it was written for")


def _anti_correlated(seed=3, n=400):
    """One label whose scores are ANTI-correlated with it -> Platt wants a < 0."""
    rng = np.random.default_rng(seed)
    y = np.zeros((n, 1), int)
    y[:60, 0] = 1
    s = rng.random(n)
    s[:60] *= 0.2
    return np.clip(s, 1e-6, 1 - 1e-6).reshape(-1, 1), y, np.ones((n, 1), bool)


def test_negative_slope_is_refused_under_the_old_rule():
    """allow_negative=False keeps the strict rank-no-op rule: a decreasing map is
    refused and the label passes through untouched."""
    P, y, ar = _anti_correlated()
    pp = fit_platt(P, y, ar, allow_negative=False)
    assert not pp.ok[0], "a non-positive slope was accepted under the strict rule"
    assert np.allclose(apply_platt(P, pp, at_risk=ar), P), "refused label was modified"


def test_negative_slope_is_applied_and_reverses_the_ranking_by_default():
    """Default: the same fit for every label. An anti-correlated label gets a < 0,
    and applying it must REVERSE the ranking -- AUROC goes from below to above 0.5."""
    from sklearn.metrics import roc_auc_score
    P, y, ar = _anti_correlated()
    pp = fit_platt(P, y, ar)
    assert pp.ok[0] and pp.a[0] < 0, f"expected an applied negative slope, got a={pp.a[0]:.3f}"
    Q = apply_platt(P, pp, at_risk=ar)
    before, after = roc_auc_score(y[:, 0], P[:, 0]), roc_auc_score(y[:, 0], Q[:, 0])
    assert before < 0.5 < after and abs(after - (1 - before)) < 1e-9


def test_verify_noop_still_checks_every_increasing_map():
    """The ship-time gate excludes only negative-slope labels: a broken
    INCREASING-map label must still trip it."""
    P, y, ar = _synthetic()
    Q = P.copy()
    Q[:, 0] = 1.0 - Q[:, 0]                 # reverse label 0
    inc = np.ones(P.shape[1], bool)
    with pytest.raises(AssertionError):
        verify_noop(P, Q, y, ar, labels=inc)
    inc[0] = False                          # declared as a negative-slope label
    verify_noop(P, Q, y, ar, labels=inc)    # the rest are untouched -> passes


def test_labels_with_too_few_positives_pass_through():
    """At ~11 positives per label the thin ones must not be fit at all."""
    n = 300
    y = np.zeros((n, 2), int)
    y[:2, 0] = 1                      # 2 positives -> below min_pos
    y[:40, 1] = 1                     # plenty
    rng = np.random.default_rng(5)
    P = np.clip(rng.random((n, 2)), 1e-6, 1 - 1e-6)
    ar = np.ones((n, 2), bool)

    pp = fit_platt(P, y, ar, min_pos=5)
    assert not pp.ok[0], "a 2-positive label was calibrated"
    Q = apply_platt(P, pp, at_risk=ar)
    assert np.allclose(Q[:, 0], P[:, 0]), "thin label was modified"


def test_murphy_decomposition_is_self_consistent():
    """Brier = REL - RES + UNC, to binning error. A decomposition that does not
    reconstruct its own total is not measuring what it claims."""
    P, y, ar = _synthetic()
    d = murphy(P, y, ar, n_bins=20)
    recon = d["rel"] - d["res"] + d["unc"]
    assert abs(recon - d["brier"]) < 5e-3, (
        f"REL-RES+UNC = {recon:.5f} but Brier = {d['brier']:.5f}")


def test_verify_noop_actually_fires_on_a_non_monotone_map():
    """The guard must be capable of failing, or it is decoration."""
    P, y, ar = _synthetic()
    Q = P.copy()
    Q[:, 0] = 1.0 - Q[:, 0]           # reverse one label's ranking
    with pytest.raises(AssertionError):
        verify_noop(P, Q, y, ar)
