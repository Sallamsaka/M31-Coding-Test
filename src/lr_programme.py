"""The logistic-regression experiment programme, run as one designed sweep.

Five questions, all cheap (9 s a fit), all previously either unmeasured or
measured against a selection set:

1. **Feature blocks (2^5) x dedup x C.** §C2 declared all the added blocks
   "inside the noise floor" against a floor of ~0.01 at n=365. In a full
   factorial each main effect is a contrast of half the runs against the other
   half, so SE(effect) = 2*sigma/sqrt(N) -- roughly 4x sharper. A block worth
   0.004 was invisible then and is a third of the entire LR-vs-P4 gap.
2. **Nested CV for C.** §G3: 0.7666 was obtained with C tuned by repeatedly
   reading validation, so it carries the winner's curse.
3. **Feature-window boundaries.** (365, 1825, inf) is pure convention, tagged
   `[M]` in the config for the *direction* of the effect but never for the
   *boundaries themselves*.
4. **Missing-indicator vs median imputation.** For labs, missingness is
   informative -- which is why `lab_min_coverage` exists at all. PROBAST rates
   missing-indicator as inadequate for inference; for prediction it is a live
   question and TRIPOD+AI item 11 requires it reported either way.
5. **Augmentation, the two variants §C1 never tried.** That negative is a
   *domain shift*, not a failure of the idea: augmented rows score 0.7548 on
   their own domain and scale correctly there (-> 0.7655), but transfer at
   0.6772. The measured mechanism is that synthetic cutoffs are 13 years
   younger and **7x sparser** (26 vs 189 pre-cutoff events). So:
   **min-history floor** attacks the sparsity half, **broad-then-finetune**
   attacks the transfer directly.

Everything is scored per-fold and averaged (§W1: pooled OOF mixes scores from
K different models, and its precision advantage is ~2.6% against a ~0.002
systematic artifact -- an order of magnitude the wrong way).

Parallelism is sized to the machine, not to `os.cpu_count()`: 8 physical cores,
~2 GB usable, one feature build costs 357 MB RSS. Measured scaling gave 5.3x at
6 workers with memory stable, so 6 it is, leaving two cores for the OS.

Run: ``python -m src.lr_programme [--quick]``
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")   # must precede the BLAS import

from joblib import Parallel, delayed  # noqa: E402

N_JOBS = 6
C_GRID = (0.01, 0.03, 0.1)
BLOCKS = ("enable_reason", "enable_slope", "enable_time_since",
          "enable_cost", "enable_age_residual")


def _score_one(root, fold_i, fit_rows, oof_rows, block_flags, dedup, C, windows=None):
    """One (config, fold, C) cell. Returns a row dict."""
    import numpy as np
    from .data.examples import ExampleConfig
    from .data.features import FeatureConfig, build_features
    from .data.labels import at_risk_mask, load_labels
    from .evaluate import macro_ap, macro_auroc
    from .train_baseline import LRConfig, apply_at_risk_mask, fit_lr

    kw = dict(zip(BLOCKS, block_flags))
    fcfg = FeatureConfig(deduplicate=dedup, **kw)
    if windows is not None:
        fcfg = FeatureConfig(deduplicate=dedup, windows=windows, **kw)
    F = build_features(root, fcfg, ex_cfg=ExampleConfig(), fit_mask=fit_rows)
    lab = load_labels(root, ExampleConfig())
    y = lab["y"].astype(int)
    ar = at_risk_mask(lab)
    P = fit_lr(F, y, ar, LRConfig(C=C), verbose=False, fit_mask=fit_rows)
    P = apply_at_risk_mask(P, ar)
    au, n_au = macro_auroc(y[oof_rows], P[oof_rows])
    ap, _ = macro_ap(y[oof_rows], P[oof_rows])
    return {"fold": fold_i, "dedup": dedup, "C": C, "n_cols": F.X.shape[1],
            "auroc": au, "ap": ap, "n_scored": n_au,
            **{b: f for b, f in zip(BLOCKS, block_flags)}}


def run_factorial(root=".", quick=False) -> pd.DataFrame:
    from .cv import fold_masks
    folds = fold_masks(root, n_splits=5)
    combos = [(False,) * 5, (True,) * 5] if quick else \
        [tuple(bool(int(b)) for b in f"{i:05b}") for i in range(32)]
    dedups = (False,) if quick else (False, True)
    cs = (0.03,) if quick else C_GRID
    jobs = [(i, fit, oof, c, dd, C)
            for i, (fit, oof) in enumerate(folds)
            for c in combos for dd in dedups for C in cs]
    print(f"factorial: {len(jobs)} cells "
          f"({len(combos)} block combos x {len(dedups)} dedup x {len(cs)} C x {len(folds)} folds)",
          flush=True)
    t = time.time()
    rows = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
        delayed(_score_one)(root, i, fit, oof, c, dd, C) for i, fit, oof, c, dd, C in jobs)
    print(f"  {len(rows)} cells in {(time.time()-t)/60:.1f} min", flush=True)
    return pd.DataFrame(rows)


def main_effects(df: pd.DataFrame, metric: str = "auroc") -> pd.DataFrame:
    """Contrast of on-vs-off for each factor, averaged over everything else."""
    out = []
    per_cfg = df.groupby(list(BLOCKS) + ["dedup", "C"])[metric].mean().reset_index()
    for f in list(BLOCKS) + ["dedup"]:
        hi = per_cfg.loc[per_cfg[f], metric]
        lo = per_cfg.loc[~per_cfg[f], metric]
        eff = hi.mean() - lo.mean()
        se = np.sqrt(hi.var(ddof=1) / len(hi) + lo.var(ddof=1) / len(lo))
        out.append({"factor": f, "effect": eff, "se": se,
                    "sigma": abs(eff) / se if se > 0 else np.nan,
                    "n_hi": len(hi), "n_lo": len(lo)})
    return pd.DataFrame(out).sort_values("sigma", ascending=False)


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--quick", action="store_true",
                     help="2 block combos, 1 dedup, 1 C -- a smoke test of the harness")
    a = ap_.parse_args()
    df = run_factorial(".", quick=a.quick)
    df.to_csv("outputs/lr_factorial.csv", index=False)
    print(f"\nwrote outputs/lr_factorial.csv  ({len(df)} rows)")

    print("\n=== per-fold mean +- SD (the primary estimand, §W1) ===")
    g = df.groupby("fold")[["auroc", "ap"]].mean()
    print(f"  macro AUROC {g.auroc.mean():.4f} +- {g.auroc.std(ddof=1):.4f}   "
          f"macro AP {g.ap.mean():.4f} +- {g.ap.std(ddof=1):.4f}   (over {len(g)} folds)")

    if not a.quick:
        print("\n=== MAIN EFFECTS on macro AUROC ===")
        me = main_effects(df, "auroc")
        print(me.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        me.to_csv("outputs/lr_main_effects.csv", index=False)
        print("\n  Pre-registered rule: report effects with intervals; act on the "
              "shrunken\n  estimate, not the argmax. 'Not resolvable' is a result.")


if __name__ == "__main__":
    main()
