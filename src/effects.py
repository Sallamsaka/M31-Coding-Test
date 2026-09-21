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
        # TWO uncertainties, because they answer different questions and the
        # first one alone is badly misleading. Measured on the real factorial:
        # 0.000018 against 0.000164, a factor of NINE.
        #
        #   se_config -- across matched configuration pairs. Patients and folds
        #     are held fixed, so patient-sampling noise cancels ENTIRELY in the
        #     pairing. This measures how CONSISTENT the effect is across
        #     configurations, NOT whether it would replicate on new patients.
        #     Quoting it as the interval implies a generalisation claim it
        #     cannot support.
        #
        #   se_fold -- across folds, each scored on different held-out patients,
        #     so it does carry patient-sampling variation. Corrected by
        #     Nadeau & Bengio (2003): K-fold training sets overlap in
        #     (K-2)/(K-1) of their patients, so the naive 1/K variance is
        #     anticonservative and the corrected factor is 1/K + n_test/n_train.
        #     Still only K-1 degrees of freedom, so it is noisy in its own right.
        #
        # The reported interval uses the LARGER of the two. Neither is exactly
        # right -- that needs a patient-level bootstrap of the OOF predictions,
        # which requires storing them -- but se_config is a lower bound and
        # quoting the larger is the conservative choice.
        n_folds = len(pf)
        if n_folds > 1:
            nb = np.sqrt((1 / n_folds + 1 / (n_folds - 1)) / (1 / n_folds))
            se_fold = float(np.std(pf, ddof=1) / np.sqrt(n_folds) * nb)
        else:
            se_fold = np.nan
        se_rep = max(se, se_fold) if np.isfinite(se_fold) else se
        t_rep = eff / se_rep if se_rep > 0 else np.nan
        out.append({"factor": f, "effect": eff,
                    "se": se_rep, "se_config": se, "se_fold": se_fold,
                    "t": t_rep,
                    "p_onesided": float(stats.t.sf(t_rep, df=max(n_folds - 1, 1)))
                    if se_rep > 0 else np.nan,
                    "n_pairs": n, "sigma": abs(t_rep) if se_rep > 0 else np.nan,
                    "fold_spread": float(np.std(pf, ddof=1)) if n_folds > 1 else np.nan})

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
    cols = ["factor", "effect", "ci_lo", "ci_hi", "se", "se_config", "se_fold",
            "sigma", "p_onesided", "q_bh", "shrunken", "n_pairs"]
    cols = [c for c in cols if c in res.columns]
    print(res[cols].to_string(index=False, float_format=lambda v: f"{v:+.5f}"))
    k = int(res.passes_bh.sum())
    print(f"\n  {k}/{len(res)} pass one-sided BH at q={q}. "
          f"Act on `shrunken`, not on `effect`, and not on this column.")
    if "se_config" in res.columns and res.se_config.gt(0).any():
        ratio = (res.se_fold / res.se_config).replace([np.inf, -np.inf], np.nan)
        print(f"  se_fold / se_config ranges {ratio.min():.0f}x to "
              f"{ratio.max():.0f}x. se_config holds patients AND folds fixed,")
        print("  so it cannot speak to generalisation; the interval uses the larger.")


# ---------------------------------------------------------------------------
# Fractional factorials need a different estimator from full ones.
# ---------------------------------------------------------------------------

def orthogonal_effects(df: pd.DataFrame, coded: list[str], metric: str,
                       sigma_pure: float | None = None,
                       n_pure: int = 0) -> pd.DataFrame:
    """Main effects of an ORTHOGONAL design, by contrast rather than by pairing.

    **Why `paired_effects` cannot be used here, found by testing before running.**
    Pairing needs each f=+1 run to have a twin at f=-1 matching on every other
    factor. In a 2^(5-1) fraction the fifth factor is *determined* by the other
    four, so no such twin exists -- every pivot cell has one level missing and
    `paired_effects` correctly returns nothing at all.

    The contrast is nevertheless unbiased here, for a different reason:
    **orthogonality**. Each factor's +1 half is balanced in every other factor,
    so their contributions cancel in expectation without any matching. That is
    the property `resolution_v_design` verifies numerically.

        effect = mean(y | f=+1) - mean(y | f=-1),     SE = 2*sigma/sqrt(N)

    **Estimating sigma without residual degrees of freedom.** Fitting 5 main
    effects and 10 two-factor interactions to 16 runs leaves 0 residual df, so
    the design cannot estimate its own error from the corners. Two independent
    routes, both reported:

    * **Pure error** from replicated centre points (what they are for). Honest
      but thin -- 4 replicates is 3 df, so the t multiplier is large.
    * **Lenth's PSE** (Technometrics 31:469-473, 1989), the standard method for
      exactly this case. It assumes effect sparsity -- that most effects are
      null -- and estimates the noise from the median of the small ones:

          s0  = 1.5 * median(|effect|)
          PSE = 1.5 * median{|effect| : |effect| < 2.5*s0}

      Robust to a few large real effects, because they are trimmed out by the
      2.5*s0 cut before the second median.

    They are reported side by side deliberately: agreement is evidence, and
    disagreement means effect sparsity fails or the centre points are unlucky,
    which is itself worth knowing before anything is concluded.
    """
    out = []
    n = len(df)
    for c in coded:
        v = df[c].to_numpy()
        hi, lo = df.loc[v > 0, metric], df.loc[v < 0, metric]
        if len(hi) == 0 or len(lo) == 0:
            continue
        out.append({"factor": c, "effect": float(hi.mean() - lo.mean()),
                    "n_hi": len(hi), "n_lo": len(lo)})
    res = pd.DataFrame(out)
    if res.empty:
        return res

    e = res.effect.to_numpy()
    # Lenth's pseudo standard error.
    s0 = 1.5 * np.median(np.abs(e))
    small = np.abs(e)[np.abs(e) < 2.5 * s0] if s0 > 0 else np.abs(e)
    pse = 1.5 * np.median(small) if len(small) else float("nan")
    res["lenth_pse"] = pse
    d_lenth = max(1, len(e) // 3)
    res["t_lenth"] = res.effect / pse if pse > 0 else np.nan
    res["p_lenth"] = 2 * stats.t.sf(np.abs(res.t_lenth), df=d_lenth)
    res["margin_of_error"] = stats.t.ppf(0.975, d_lenth) * pse

    if sigma_pure is not None and np.isfinite(sigma_pure) and n_pure > 1:
        se = 2 * sigma_pure / np.sqrt(n)
        res["se_pure"] = se
        res["t_pure"] = res.effect / se if se > 0 else np.nan
        res["p_pure"] = 2 * stats.t.sf(np.abs(res.t_pure), df=n_pure - 1)
        res["ci_lo"] = res.effect - stats.t.ppf(0.975, n_pure - 1) * se
        res["ci_hi"] = res.effect + stats.t.ppf(0.975, n_pure - 1) * se
        res["shrunken"] = shrink(res.effect.to_numpy(),
                                 np.full(len(res), se))
    else:
        res["shrunken"] = shrink(e, np.full(len(e), pse if pse > 0 else 1.0))

    return res.reindex(res.effect.abs().sort_values(ascending=False).index
                       ).reset_index(drop=True)


def report_orthogonal(res: pd.DataFrame, metric: str) -> None:
    if res.empty:
        print("  no factors to contrast")
        return
    print(f"\n=== MAIN EFFECTS on {metric} (orthogonal contrasts) ===")
    cols = [c for c in ("factor", "effect", "ci_lo", "ci_hi", "se_pure",
                        "lenth_pse", "t_lenth", "p_lenth", "shrunken")
            if c in res.columns]
    print(res[cols].to_string(index=False, float_format=lambda v: f"{v:+.5f}"))
    moe = float(res.margin_of_error.iloc[0])
    big = res.loc[res.effect.abs() > moe, "factor"].tolist()
    print(f"\n  Lenth margin of error (95%): {moe:.5f}")
    print(f"  exceeding it: {big if big else 'none -- no effect is resolvable'}")
    print("  Act on `shrunken`. 'Not resolvable' is a result, not a failure.")
