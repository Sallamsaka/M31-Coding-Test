"""Main effects from a factorial, estimated as PAIRED contrasts, with BH.

Used by `design` (the transformer
2^(5-1)). Kept separate from it because it is pure arithmetic on a table and
is therefore the one part of the experimental machinery that can be tested
without spending an hour of CPU.

**Why paired.** An earlier version of this computed

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

import itertools

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
        # NOTE: no t or p from these n pairs. An earlier version computed one
        # with df = n_pairs - 1, which is the WRONG reference distribution --
        # the pairs are not the replication unit (E17). The reported p uses the
        # fold-based SE and df below.
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
    if q is None:
        q = float(res.q_level.iloc[0]) if "q_level" in res.columns else 0.15
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
                       n_pure: int = 0,
                       interactions: bool = True) -> pd.DataFrame:
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
    y = df[metric].to_numpy(float)

    # Main effects, then every two-factor interaction. Resolution V makes the
    # interactions estimable and unaliased, which was the design's whole reason
    # for existing -- reporting only main effects throws that away.
    #
    # They also fix Lenth's PSE, whose degrees of freedom are m/3: five main
    # effects give d = 1 and t(0.975, 1) = 12.7, a margin of error of 0.078 that
    # no effect on this scale could exceed. That is not "nothing is resolvable",
    # it is "the estimator had no degrees of freedom". With the ten interactions
    # included, m = 15 and d = 5.
    contrasts = [(c, df[c].to_numpy(float)) for c in coded]
    if interactions:
        base = list(contrasts)
        for (a_, va), (b_, vb) in itertools.combinations(base, 2):
            contrasts.append((f"{a_}:{b_}", va * vb))

    for name, v in contrasts:
        hi, lo = y[v > 0], y[v < 0]
        if len(hi) == 0 or len(lo) == 0:
            continue
        out.append({"factor": name, "effect": float(hi.mean() - lo.mean()),
                    "kind": "interaction" if ":" in name else "main",
                    "n_hi": len(hi), "n_lo": len(lo)})
    res = pd.DataFrame(out)
    if res.empty:
        return res

    # Lenth's PSE must be pooled over EVERY estimable contrast, not just the
    # main effects. Its degrees of freedom are m/3, so five main effects give
    # df = 1 and t(0.975, 1) = 12.71 -- a margin of error so wide the method
    # cannot reject anything, which is what it did here before this fix. A
    # 2^(5-1) resolution V design estimates 5 main effects AND 10 two-factor
    # interactions, so the honest pool is 15, giving df = 5 and t = 2.57.
    #
    # Pooling the interactions is also what makes the sparsity assumption
    # reasonable: it is the interactions that are mostly null, and they are the
    # reference distribution the main effects are being judged against.
    # BUG, caught by the printed df disagreeing with the margin of error
    # actually used: when `interactions=True` the loop below rebuilt the same
    # ten two-factor contrasts that `res` already holds, so every interaction
    # entered the pool TWICE -- 25 entries instead of 15, df 8 instead of 5,
    # and a PSE median double-weighted toward the interactions. The report
    # printed "df = 5" while `margin_of_error` had been computed at df = 8
    # (t = 2.31, not 2.57), which is how it was noticed.
    pool = list(res.effect.to_numpy())
    if not interactions:
        for a, b in itertools.combinations(coded, 2):
            v = (df[a] * df[b]).to_numpy()
            hi_i, lo_i = df.loc[v > 0, metric], df.loc[v < 0, metric]
            if len(hi_i) and len(lo_i):
                pool.append(float(hi_i.mean() - lo_i.mean()))
    e = np.asarray(pool)
    s0 = 1.5 * np.median(np.abs(e))
    small = np.abs(e)[np.abs(e) < 2.5 * s0] if s0 > 0 else np.abs(e)
    pse = 1.5 * np.median(small) if len(small) else float("nan")
    res["lenth_pse"] = pse
    res["lenth_n_contrasts"] = len(e)
    d_lenth = max(1, len(e) // 3)
    # Stored rather than recomputed downstream. The reporter used to derive it
    # from len(res), which is a DIFFERENT quantity whenever the pool and the
    # reported rows differ -- and that silent disagreement is what hid the
    # double-counting above.
    res["lenth_df"] = d_lenth
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
        res = _bh(res, "p_pure")
    else:
        # `e` is the 15-contrast Lenth POOL; shrinkage applies to the main
        # effects only, which is what `res` holds.
        main = res.effect.to_numpy()
        res["shrunken"] = shrink(main, np.full(len(main), pse if pse > 0 else 1.0))
        res = _bh(res, "p_lenth")

    return res.reindex(res.effect.abs().sort_values(ascending=False).index
                       ).reset_index(drop=True)


def _bh(res: pd.DataFrame, pcol: str, q: float = 0.15) -> pd.DataFrame:
    """Benjamini-Hochberg over every contrast in a fractional design.

    The fractional path had no multiplicity control at all while the full
    factorial path had BH at q=0.15 -- and it is the fractional path that
    reports FIFTEEN effects at once. Uncorrected, nine of fifteen AUROC
    contrasts "excluded zero", which is not a plausible state of the world;
    it is what 15 nominal 95% intervals do.

    Two-sided here, unlike `paired_effects`, and on firmer ground rather than
    weaker. `paired_effects` needs the one-sided restriction because its
    contrasts are correlated, which is outside BH 1995 and only inside
    Benjamini-Yekutieli's studentized case. A resolution-V design's contrasts
    are **exactly orthogonal by construction** and share one iid error term, so
    the p-values are independent and BH 1995's original independence proof
    applies directly, in either direction. A negative effect is as actionable
    as a positive one here: "dropout hurts" tells us to set it low.
    """
    if pcol not in res.columns:
        return res
    r = res.sort_values(pcol).reset_index(drop=True)
    m = len(r)
    raw = r[pcol].to_numpy() * m / (np.arange(m) + 1)
    r["q_bh"] = np.minimum.accumulate(raw[::-1])[::-1].clip(0, 1)
    r["passes_bh"] = r["q_bh"] <= q
    # STORED, so the reporter prints the q that was actually applied instead of
    # its own default. Two independent defaults for one threshold is precisely
    # the defect of D27 -- there it was Lenth's df derived twice and the two
    # copies drifting apart unnoticed. Changing q here would otherwise leave
    # `report_orthogonal` announcing 0.15 while `passes_bh` used something else.
    r["q_level"] = q
    return r


def report_orthogonal(res: pd.DataFrame, metric: str,
                      q: float | None = None) -> None:
    if res.empty:
        print("  no factors to contrast")
        return
    print(f"\n=== MAIN EFFECTS on {metric} (orthogonal contrasts) ===")
    cols = [c for c in ("factor", "kind", "effect", "ci_lo", "ci_hi", "se_pure",
                        "lenth_pse", "t_lenth", "p_lenth", "q_bh", "shrunken")
            if c in res.columns]
    print(res[cols].to_string(index=False, float_format=lambda v: f"{v:+.5f}"))
    moe = float(res.margin_of_error.iloc[0])
    big = res.loc[res.effect.abs() > moe, "factor"].tolist()
    d_lenth = int(res.lenth_df.iloc[0]) if "lenth_df" in res.columns else 1
    n_pool = int(res.lenth_n_contrasts.iloc[0])
    print(f"\n  Lenth margin of error (95%): {moe:.5f}  "
          f"(pooled over {n_pool} contrasts, df = {d_lenth})")
    if d_lenth <= 2:
        print(f"  WARNING: too few effects for Lenth. t(0.975, {d_lenth}) is "
              "enormous, so this margin")
        print("  rejects everything regardless of the data. Read the pure-error "
              "interval instead.")
    print(f"  exceeding it: {big if big else 'none'}")
    if "ci_lo" in res.columns:
        pe = res[(res.ci_lo > 0) | (res.ci_hi < 0)].factor.tolist()
        print(f"  excluding zero by PURE ERROR (uncorrected): "
              f"{pe if pe else 'none'}")
    if "passes_bh" in res.columns:
        bh = res.loc[res.passes_bh, "factor"].tolist()
        print(f"  surviving BH at q={q} across all {len(res)} contrasts: "
              f"{bh if bh else 'none'}")
        k = len(bh)
        print("  BH names MORE than the uncorrected line, and that is correct,"
              " not a bug:")
        print("  the two control different things. The line above is"
              " per-comparison alpha=0.05")
        print(f"  applied {len(res)} times, so ~{0.05 * len(res):.1f} of its"
              f" entries are expected to be")
        print(f"  false with no true effect at all. BH at q={q} instead bounds"
              f" the expected")
        print(f"  FALSE-DISCOVERY PROPORTION *within the named set*: of these"
              f" {k}, at most ~{q * k:.1f}")
        print("  are expected to be spurious. That is the screening trade BH"
              " 1995 recommends")
        print("  for 2^k designs -- a deliberately permissive q, bought with a"
              " bound on how")
        print("  much of the harvest is chaff. Quote BH; do not read it as the"
              " stricter test.")
    print("  Act on `shrunken`. 'Not resolvable' is a result, not a failure.")
