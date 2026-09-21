"""Main effects from a factorial, estimated as PAIRED contrasts, with BH.

Shared by `lr_programme` (2^5 blocks x dedup x C) and `design` (the transformer
2^(5-1)). Kept separate from both because it is pure arithmetic on a table and
is therefore the one part of the experimental machinery that can be tested
without spending an hour of CPU.

**Why paired.** An earlier version of `lr_programme.main_effects` computed

    effect = mean(scores | f=1) - mean(scores | f=0)
    SE     = sqrt(var_hi/n_hi + var_lo/n_lo)          # two-sample, unpaired

which is wrong in a way that costs real resolution. In a full factorial every
f=1 configuration has an *exact twin* at f=0 differing in nothing but f. The
unpaired SE carries the entire between-configuration variance contributed by the
other factors; the paired one cancels it, exactly as `paired_bootstrap_delta`
cancels patient difficulty by resampling the same patients for both models.

    unpaired:  Var(d) = Var(hi) + Var(lo)
    paired:    Var(d) = Var(hi) + Var(lo) - 2*Cov(hi, lo)

Twins share their other five factor settings, their folds and their training
data, so Cov is large and positive and the third term dominates. This is the
same mechanism that makes the project's paired bootstrap tighter than a
difference of two marginal intervals.

**Unit of replication.** Each pair is averaged over folds first, and the spread
is taken across the n/2 pairs. Folds are *not* used as the replication unit:
with K=5 that is 4 degrees of freedom, and fold scores are correlated because
the training sets overlap in (K-2)/(K-1) of their patients, so a naive
across-fold t-test has badly inflated Type I error (Nadeau & Bengio 2003). The
per-fold effect spread is reported alongside as a diagnostic, not as the
interval.

**Multiplicity.** One-sided Benjamini-Hochberg at q=0.15. One-sided because we
only care about improvement, and because it moves us out of the two-sided
correlated case BY does not cover and into the studentized case it does -- in a
balanced orthogonal factorial the contrasts are uncorrelated by construction,
which is BY's own worked example. BH 1995 names 2^k screening as a founding
application and says q should be set higher than a conventional alpha for
screening, hence 0.15 rather than 0.05.

**Reporting rule.** The q-value is for the screen, not for the decision. Act on
the shrunken estimate (`shrink`), never on the argmax and never on a
significance gate -- selection and inference are different problems, and at 0.6
sigma something still has to be chosen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def paired_effects(df: pd.DataFrame, factors: list[str], metric: str,
                   fold_col: str = "fold", q: float = 0.15) -> pd.DataFrame:
    """One row per factor: paired effect, SE, t, one-sided p, BH q, shrunken.

    `df` has one row per (configuration, fold). `factors` are the columns that
    define a configuration; every other configuration column is held fixed
    within a pair.
    """
    out = []
    cfg_cols = [c for c in factors]
    for f in factors:
        others = [c for c in cfg_cols if c != f]
        # Average over folds first, then pivot so each row is one twin pair.
        g = df.groupby(cfg_cols, dropna=False)[metric].mean().reset_index()
        wide = g.pivot_table(index=others, columns=f, values=metric)
        if wide.shape[1] != 2:
            continue                       # factor not varied, or not 2-level
        lo_lvl, hi_lvl = sorted(wide.columns)
        d = (wide[hi_lvl] - wide[lo_lvl]).dropna().to_numpy()
        n = len(d)
        if n < 2:
            continue
        eff = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(n))
        t = eff / se if se > 0 else np.nan
        # One-sided: H1 is "the high level is better".
        p = float(stats.t.sf(t, df=n - 1)) if se > 0 else np.nan
        # Diagnostic only -- see the module docstring on why this is not the SE.
        per_fold = df.groupby([fold_col] + cfg_cols, dropna=False)[metric].mean().reset_index()
        pf = []
        for _, sub in per_fold.groupby(fold_col):
            w = sub.pivot_table(index=others, columns=f, values=metric)
            if w.shape[1] == 2:
                pf.append(float((w[hi_lvl] - w[lo_lvl]).mean()))
        out.append({"factor": f, "effect": eff, "se": se, "t": t, "p_onesided": p,
                    "n_pairs": n, "sigma": abs(t) if se > 0 else np.nan,
                    "fold_spread": float(np.std(pf, ddof=1)) if len(pf) > 1 else np.nan})

    res = pd.DataFrame(out)
    if res.empty:
        return res
    res = res.sort_values("p_onesided").reset_index(drop=True)
    m = len(res)
    # Benjamini-Hochberg, with the standard monotonicity enforcement.
    raw = res["p_onesided"].to_numpy() * m / (np.arange(m) + 1)
    res["q_bh"] = np.minimum.accumulate(raw[::-1])[::-1].clip(0, 1)
    res["passes_bh"] = res["q_bh"] <= q
    res["ci_lo"] = res.effect - 1.96 * res.se
    res["ci_hi"] = res.effect + 1.96 * res.se
    res["shrunken"] = shrink(res.effect.to_numpy(), res.se.to_numpy())
    return res


def shrink(effect: np.ndarray, se: np.ndarray) -> np.ndarray:
    """James-Stein / empirical-Bayes shrinkage toward zero.

    The argmax of a set of noisy effect estimates is biased away from zero by
    the winner's curse -- the very bias this project measures at +0.038 to
    +0.051 on validation. Shrinking by the estimated signal-to-noise ratio

        tau^2 = max(0, var(effect) - mean(se^2)),   w = tau^2 / (tau^2 + se^2)

    removes most of it without a significance gate, which would throw away every
    real-but-small effect the literature says dominates here (Probst et al.
    measure individual hyperparameters at 0.001-0.002, i.e. below our noise).
    When the observed spread is entirely explained by noise, tau^2 = 0 and every
    effect shrinks to exactly zero -- the correct answer for "nothing here is
    resolvable".
    """
    tau2 = max(0.0, float(np.var(effect, ddof=1) - np.mean(se ** 2)))
    return effect * (tau2 / (tau2 + se ** 2))


def report(res: pd.DataFrame, metric: str, q: float = 0.15) -> None:
    if res.empty:
        print("  no two-level factors to contrast")
        return
    print(f"\n=== MAIN EFFECTS on {metric} (paired contrasts) ===")
    cols = ["factor", "effect", "ci_lo", "ci_hi", "se", "sigma",
            "p_onesided", "q_bh", "shrunken", "n_pairs"]
    print(res[cols].to_string(index=False, float_format=lambda v: f"{v:+.5f}"))
    k = int(res.passes_bh.sum())
    print(f"\n  {k}/{len(res)} pass one-sided BH at q={q}. "
          f"Act on `shrunken`, not on `effect`, and not on this column.")
    worst = res.fold_spread.max()
    if np.isfinite(worst):
        print(f"  largest across-fold spread of any effect: {worst:.5f} "
              f"(diagnostic; folds share training data so this is not an SE)")
