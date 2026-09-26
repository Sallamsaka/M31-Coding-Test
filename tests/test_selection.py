"""The pre-registered selection rule, tested on synthetic predictions.

The rule exists to stop the winner being chosen by reading the table. Its two
branches have opposite failure modes, so both are pinned:

* if a difference IS resolvable and the rule still falls back to simplicity,
  a genuinely better model never ships;
* if a difference is NOT resolvable and the rule takes the point estimate,
  every run adopts whichever model noise favoured -- the winner's curse the
  ledger exists to bound.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import macro_ap, paired_bootstrap_delta


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def test_logit_average_is_symmetric_and_bounded():
    rng = np.random.default_rng(0)
    a, b = rng.uniform(0.001, 0.999, 500), rng.uniform(0.001, 0.999, 500)
    ab = _sigmoid(0.5 * (_logit(a) + _logit(b)))
    ba = _sigmoid(0.5 * (_logit(b) + _logit(a)))
    assert np.allclose(ab, ba)
    assert ((ab > 0) & (ab < 1)).all(), "an averaged probability left (0,1)"
    # The geometric mean of odds sits between its parents.
    assert (ab >= np.minimum(a, b) - 1e-9).all()
    assert (ab <= np.maximum(a, b) + 1e-9).all()


@pytest.mark.parametrize("gap,expect_resolved", [(0.0, False), (0.35, True)])
def test_rule_only_adopts_a_winner_it_can_resolve(gap, expect_resolved):
    """Two models differing by `gap` in signal strength, same 365 patients."""
    rng = np.random.default_rng(1)
    n, k = 365, 40
    y = (rng.random((n, k)) < 0.05).astype(int)
    for j in range(k):                       # guarantee every label is scorable
        y[rng.integers(0, n), j] = 1
        y[rng.integers(0, n), j] = 0

    noise = rng.normal(size=(n, k))
    simple = _sigmoid(0.6 * y + noise)
    complex_ = _sigmoid((0.6 + gap) * y + noise)

    d = paired_bootstrap_delta(y, complex_, simple, n_boot=300, seed=0)
    pt, lo, hi, _ = d["d_macro_ap"]
    resolved = lo > 0 or hi < 0
    assert resolved == expect_resolved, (
        f"gap={gap}: delta {pt:+.4f} [{lo:+.4f},{hi:+.4f}]")

    if resolved:
        assert macro_ap(y, complex_)[0] > macro_ap(y, simple)[0]
