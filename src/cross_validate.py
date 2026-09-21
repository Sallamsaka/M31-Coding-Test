"""The headline instrument: K=5 grouped CV over the pool, and the locked read.

Everything reported before this was measured on the 365-patient validation set,
which has been read ~26 times and carries a winner's-curse bound of +0.038 to
+0.051 -- larger than every difference this project has ever measured between
models (§G1). This module is what replaces it.

**Per-fold mean +- SD is PRIMARY; pooled OOF is secondary (§W1).** The two are
different estimands and can order models differently. AUROC is a *ranking*
metric, and pooling ranks patient A's score from model 1 against patient B's
score from model 2 -- scores that were never on a common scale. The measured
symptom is the prevalence-model artifact: pooled OOF AUROC for a constant
predictor is **not 0.5**, because a patient's own outcome is excluded from the
fit that scores them, so a fold with more positives hands all its members a
lower predicted rate and score anticorrelates with outcome.

So pooling is reported two ways: raw, and after **within-fold rank
normalisation** (ranked *before* the 1e-6 at-risk floor is applied, or the ties
corrupt the ranks). The gap between them is the measured heterogeneity budget,
not a nuisance to hide.

**One partition, one K, for anything compared (§X).** Unequal K applies the
fit-size tax asymmetrically and hits the more data-hungry family harder -- we
would measure data-hunger and report it as inferiority. Every model here runs on
the *same* `fold_masks` output.

**What the CV number is, stated so the report cannot overclaim.** It estimates
the expected performance of the *procedure* trained on ~1,946 patients, not of
any single shipped model -- each patient is scored by a different one. It is a
lower bound on the shipped model (refit on the whole pool) by the fit-size tax.
The bootstrap interval covers evaluation noise only.

**The locked test set is read ONCE**, for one pre-registered question: does the
shipped model's macro AP exceed the at-risk-mask-only baseline, with an interval?
It is 358 patients with 3 positives on the rarest label, so it **cannot** rank
LR against GBDT against the transformer and must not be asked to (§W3). It also
de-biases the *fit*, not the *design*: 26 validation reads and all feature
engineering preceded the carve.

Run: ``python -m src.cross_validate``              (LR + GBDT + prevalence)
     ``python -m src.cross_validate --locked``     (the single locked read)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .cv import cv_pool_pids, fold_masks, locked_test_pids
from .data.examples import ExampleConfig
from .data.features import build_features
from .data.labels import at_risk_mask, load_labels
from .evaluate import macro_ap, macro_auroc, paired_bootstrap_delta
from .train_baseline import (GBDTConfig, LRConfig, apply_at_risk_mask, fit_gbdt,
                             fit_lr, fit_prevalence)

MODELS = ("prevalence", "lr", "gbdt")


def _rank_within(p: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Per-label rank-normalise a fold's scores to [0, 1].

    Applied BEFORE the at-risk floor, so the 1e-6 sentinels do not collapse into
    a tie block that destroys the ranking this is meant to preserve.
    """
    out = np.empty_like(p[rows], dtype=float)
    for j in range(p.shape[1]):
        v = p[rows, j]
        order = np.argsort(np.argsort(v))
        out[:, j] = order / max(len(v) - 1, 1)
    return out


def run_cv(root: str = ".", n_splits: int = 5, verbose: bool = True) -> dict:
    ex_cfg = ExampleConfig()
    lab = load_labels(root, ex_cfg)
    y = lab["y"].astype(int)
    ar = at_risk_mask(lab)
    folds = fold_masks(root, n_splits=n_splits)

    n = len(y)
    oof = {m: np.zeros((n, y.shape[1]), np.float32) for m in MODELS}
    oof_rank = {m: np.zeros((n, y.shape[1]), np.float32) for m in MODELS}
    scored = np.zeros(n, bool)
    per_fold: list[dict] = []

    for k, (fit_rows, oof_rows) in enumerate(folds):
        F = build_features(root, ex_cfg=ex_cfg, fit_mask=fit_rows)
        idx = np.flatnonzero(oof_rows)
        scored[idx] = True
        preds = {
            "prevalence": fit_prevalence(F, y, fit_mask=fit_rows),
            "lr": fit_lr(F, y, ar, LRConfig(), verbose=False, fit_mask=fit_rows),
            "gbdt": fit_gbdt(F, y, ar, GBDTConfig(), verbose=False,
                             fit_mask=fit_rows),
        }
        row = {"fold": k, "n_fit": int(fit_rows.sum()), "n_oof": len(idx)}
        for m, P in preds.items():
            # Rank BEFORE the floor (see _rank_within), then floor for scoring.
            oof_rank[m][idx] = _rank_within(P, idx)
            Pm = apply_at_risk_mask(P, ar)
            oof[m][idx] = Pm[idx]
            au, n_ok = macro_auroc(y[idx], Pm[idx], mask=ar[idx])
            ap, _ = macro_ap(y[idx], Pm[idx], mask=ar[idx])
            row[f"{m}_auroc"] = au
            row[f"{m}_ap"] = ap
            row[f"{m}_n_scored"] = n_ok
        per_fold.append(row)
        if verbose:
            print(f"  fold {k}: fit {row['n_fit']:,} oof {row['n_oof']:,}  "
                  + "  ".join(f"{m} AP {row[f'{m}_ap']:.4f}" for m in MODELS),
                  flush=True)

    df = pd.DataFrame(per_fold)
    df.to_csv("outputs/cv_per_fold.csv", index=False)

    # Each patient must be held out exactly once, or `bootstrap_macro`'s promise
    # to resample patients rather than rows is silently false.
    pid = lab["pid"] if "pid" in lab else None
    if pid is not None:
        p_oof = np.asarray(pid)[scored]
        assert len(np.unique(p_oof)) == len(p_oof), "a patient is scored twice"

    return {"per_fold": df, "oof": oof, "oof_rank": oof_rank, "scored": scored,
            "y": y, "ar": ar}


def report_cv(res: dict, verbose: bool = True) -> None:
    df, y, ar = res["per_fold"], res["y"], res["ar"]
    rows = np.flatnonzero(res["scored"])

    print("\n=== PRIMARY: per-fold mean +- SD across folds (§W1) ===")
    print(f"{'model':<12}{'macro AP':>20}{'macro AUROC':>22}")
    for m in MODELS:
        ap, au = df[f"{m}_ap"], df[f"{m}_auroc"]
        print(f"{m:<12}{ap.mean():>12.4f} +- {ap.std(ddof=1):.4f}"
              f"{au.mean():>14.4f} +- {au.std(ddof=1):.4f}")

    print("\n=== SECONDARY: pooled OOF, raw vs within-fold rank-normalised ===")
    print(f"{'model':<12}{'AP raw':>10}{'AP rank':>10}{'AUROC raw':>12}"
          f"{'AUROC rank':>12}")
    for m in MODELS:
        P, R = res["oof"][m], res["oof_rank"][m]
        Rm = apply_at_risk_mask(R, ar)
        a1 = macro_ap(y[rows], P[rows], mask=ar[rows])[0]
        a2 = macro_ap(y[rows], Rm[rows], mask=ar[rows])[0]
        u1 = macro_auroc(y[rows], P[rows], mask=ar[rows])[0]
        u2 = macro_auroc(y[rows], Rm[rows], mask=ar[rows])[0]
        print(f"{m:<12}{a1:>10.4f}{a2:>10.4f}{u1:>12.4f}{u2:>12.4f}")

    # The pooling artifact, measured rather than asserted away. A constant
    # predictor pooled across folds does NOT score 0.5, so run_baseline's
    # `assert abs(au - 0.5) < 1e-9` must be replaced here, not reused.
    pv = macro_auroc(y[rows], res["oof"]["prevalence"][rows],
                     mask=ar[rows])[0]
    print(f"\n  pooling artifact: prevalence-only pooled AUROC = {pv:.4f} "
          f"(0.5 only if pooling were benign)")
    print("  the raw-vs-rank gap above is the measured heterogeneity budget.")

    print("\n=== PAIRED COMPARISONS on identical folds ===")
    for a, b in (("lr", "gbdt"), ("lr", "prevalence"), ("gbdt", "prevalence")):
        d = paired_bootstrap_delta(y[rows], res["oof"][a][rows],
                                   res["oof"][b][rows])
        pt, lo, hi, _sd = d["d_macro_ap"]      # (point, lo, hi, sd)
        pg = d.get("p_a_gt_b_ap")
        tag = "resolvable" if (lo > 0 or hi < 0) else "not resolvable"
        extra = f"  P({a}>{b}) = {pg:.2f}" if pg is not None else ""
        print(f"  {a:>10} - {b:<12} dAP {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]"
              f"  {tag}{extra}")
    print("\n  P(A>B) is reported beside every interval on purpose: a"
          " significance gate\n  alone false-negatives ~90% of the time at this"
          " resolution (Bouthillier\n  et al. 2021), against ~30% for P(A>B).")


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--folds", type=int, default=5)
    ap_.add_argument("--locked", action="store_true",
                     help="THE SINGLE LOCKED-TEST READ -- pre-registered, once")
    a = ap_.parse_args()

    if a.locked:
        locked = locked_test_pids()
        pool = cv_pool_pids()
        print(f"locked test: {len(locked)} patients; pool {len(pool)}")
        print("This is the one pre-registered read. It answers exactly one")
        print("question -- does the shipped model beat the at-risk-mask-only")
        print("baseline on macro AP, with an interval -- and it CANNOT rank")
        print("LR against GBDT against the transformer (§W3). Not run")
        print("automatically: it is a deliberate, one-time action.")
        return

    print(f"K={a.folds} grouped CV over the pool, one partition for every model",
          flush=True)
    res = run_cv(".", n_splits=a.folds)
    report_cv(res)


if __name__ == "__main__":
    main()
