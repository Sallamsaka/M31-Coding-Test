"""LR optimization: the regularisation path, which is where the payoff is.

**Why regularisation and not features.** Two measurements, pointing the same way.
The 960-run feature factorial (§E17) put all six block factors at **+0.0006 AP
total**, 0/6 passing BH, every James-Stein shrunken estimate exactly 0. Tuning
**C** alone is worth **+0.0141 AP** (C=0.03 against sklearn's 1.0, on the CV
pool). Regularisation beats feature engineering by roughly **20x** here, and
§E13's Riley bound says why: the median label supports ~10.5 parameters and we
feed 3,296.

The same shape showed up twice more this session, in models that share nothing
with LR. The transformer's fusion projection wanted to be **bigger**
(`fdim128` +0.0054, `fdim16` -0.0091, monotone) while its regularisers helped;
GBDT's best screen cell had **10x** the default L2 and **5x** the minimum leaf
size while wanting *more* leaves and *more* iterations. Regularise harder, do not
shrink capacity -- and for LR the only capacity knob is the feature set, which is
already measured as null. So: the C path.

**What is actually open.** The existing factorial swept C over {0.01, 0.03, 0.1}
and then *averaged over it* -- `main_effects` was passed the blocks and dedup
only, so C never entered an effects table. Aggregating its 960 rows by hand:

    C       at-risk AUROC   at-risk AP
    0.01       0.7134         0.1907
    0.03       0.7182         0.2001     <- AUROC peak
    0.10       0.7165         0.2031     <- AP still RISING at the top

**On AP the grid ended while still climbing**, and AP is the pre-registered
primary (§E11: selecting on AUROC costs 5x the regret of selecting on AP, even
when the target is AUROC). That is the cheapest unexplored direction in the
project. This module extends the grid in both directions and adds `log1p_counts`,
a genuine knob for a linear model that has never once been swept.

⚠ **§D3, which governs how the result may be used.** sklearn minimises
``0.5*w'w + C*sum(loss)``: the penalty is fixed while the data term is a **sum**,
so effective regularisation moves with n *and* with feature width. A C tuned at
the CV fit size (1,946 rows) is **not** the C for the shipped refit (2,433). Tune
and ship at the same width, or re-tune -- this has already bitten once.

**Reached through `_fit_per_label`'s `make_model` hook**, so `fit_lr`,
`LRConfig` and `artifacts/model_lr.joblib` are untouched: §G4's rule that an
unmeasured change must not re-baseline every number in the report.

Run: ``python -u -m src.lr_optimize``          (resumes; safe to re-launch)
     ``python -u -m src.lr_optimize --report``
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

LEDGER = Path("outputs/lr_optimize.jsonl")
CSV = Path("outputs/lr_optimize.csv")

# Extended in BOTH directions. Upward because AP was still rising at 0.1;
# downward because the AUROC peak at 0.03 is interior and the two metrics
# disagreeing about the optimum is itself worth resolving rather than assuming.
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
LOG1P = (True, False)
N_FOLDS = 5


def _key(C: float, log1p: bool, fold: int) -> tuple:
    return (round(float(C), 10), bool(log1p), int(fold))


def _done() -> set:
    out = set()
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if {"C", "log1p", "fold"} <= set(r):
                    out.add(_key(r["C"], r["log1p"], r["fold"]))
    return out


def _score_cell(root, fold, fit_rows, oof_rows, C, log1p, test_mask):
    """One (C, log1p, fold) cell, scored on the fold's OOF rows AND our test set.

    The fit rows come from `fold_masks`, which excludes both the held-out fold
    and our 358-patient test set, so scoring there needs no separate model --
    which is the whole reason a test-set column is affordable at all.
    """
    from .data.examples import ExampleConfig
    from .data.features import build_features
    from .data.labels import at_risk_mask, load_labels
    from .evaluate import macro_ap, macro_auroc
    from .train_baseline import LRConfig, apply_at_risk_mask, fit_lr

    F = build_features(root, ex_cfg=ExampleConfig(), fit_mask=fit_rows)
    lab = load_labels(root, ExampleConfig())
    y = lab["y"].astype(int)
    ar = at_risk_mask(lab)
    P = apply_at_risk_mask(
        fit_lr(F, y, ar, LRConfig(C=C, log1p_counts=log1p), verbose=False,
               fit_mask=fit_rows), ar)

    out = {"C": C, "log1p": log1p, "fold": fold, "n_cols": F.X.shape[1]}
    for tag, m in (("oof", np.flatnonzero(oof_rows)),
                   ("test", np.flatnonzero(test_mask))):
        au, n_ok = macro_auroc(y[m], P[m], mask=ar[m])
        ap, _ = macro_ap(y[m], P[m], mask=ar[m])
        out[f"{tag}_auroc"], out[f"{tag}_ap"] = au, ap
        out[f"{tag}_n"], out[f"{tag}_scored"] = len(m), n_ok
    return out


def run(root: str = ".") -> pd.DataFrame:
    from .cv import fold_masks
    from .data.examples import ExampleConfig
    from .data.labels import load_labels

    lab = load_labels(root, ExampleConfig())
    # The carve is retired (cv.LOCKED_TEST_N = 0); this column now reports
    # fold 0's held-out rows, which every non-fold-0 model has genuinely not
    # seen. Levels before and after the re-carve are not comparable.
    _, _oof0 = fold_masks(root, n_splits=5)[0]
    test_mask = np.asarray(_oof0, bool)
    assert test_mask.sum() > 0, "fold 0 is empty"

    folds = fold_masks(root, n_splits=N_FOLDS)
    done = _done()
    rows = pd.read_csv(CSV).to_dict("records") if CSV.exists() else []
    cells = [(C, lg, fi) for C in C_GRID for lg in LOG1P
             for fi in range(len(folds))]
    todo = [c for c in cells if _key(*c) not in done]
    print(f"LR optimization: {len(C_GRID)} C x {len(LOG1P)} log1p x {N_FOLDS}"
          f" folds = {len(cells)} cells, {len(done)} done, {len(todo)} to run",
          flush=True)

    t0 = time.time()
    for i, (C, lg, fi) in enumerate(todo):
        fit_rows, oof_rows = folds[fi]
        r = _score_cell(root, fi, fit_rows, oof_rows, C, lg, test_mask)
        rows.append(r)
        LEDGER.parent.mkdir(exist_ok=True)
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(r) + "\n")
        pd.DataFrame(rows).to_csv(CSV, index=False)
        if (i + 1) % 5 == 0 or i == len(todo) - 1:
            print(f"  {i+1}/{len(todo)}  C={C} log1p={lg} fold={fi}"
                  f"  oof AP {r['oof_ap']:.4f}  test AP {r['test_ap']:.4f}"
                  f"  [{(time.time()-t0)/60:.0f} min]", flush=True)
    return pd.DataFrame(rows)


def report() -> None:
    if not CSV.exists():
        print("no outputs/lr_optimize.csv yet")
        return
    d = pd.read_csv(CSV)
    g = d.groupby(["C", "log1p"]).agg(
        n=("fold", "count"),
        oof_ap=("oof_ap", "mean"), oof_ap_sd=("oof_ap", "std"),
        oof_auroc=("oof_auroc", "mean"),
        test_ap=("test_ap", "mean"), test_auroc=("test_auroc", "mean"),
    ).reset_index().sort_values("oof_ap", ascending=False)
    print("\n=== LR regularisation path ===")
    print("  selection column is oof_ap (2,433 patients). test_* is our carved")
    print("  fold 0 -- reported for every row, never selected on.")
    print(g.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best = g.iloc[0]
    base = g[(g.C == 0.03) & (g.log1p)]
    if len(base):
        b = base.iloc[0]
        print(f"\n  shipped default (C=0.03, log1p=True): oof AP {b.oof_ap:.4f}")
        print(f"  best on oof: C={best.C} log1p={best.log1p} -> {best.oof_ap:.4f}"
              f"  ({best.oof_ap - b.oof_ap:+.4f})")
    # Is the optimum interior, or did the grid end while still climbing again?
    for lg in LOG1P:
        sub = g[g.log1p == lg].sort_values("C")
        if len(sub) >= 3:
            top = sub.loc[sub.oof_ap.idxmax(), "C"]
            edge = "AT AN EDGE -- extend the grid" if top in (min(C_GRID), max(C_GRID)) \
                else "interior"
            print(f"  log1p={lg}: AP optimum at C={top} ({edge})")
    g.to_csv("outputs/lr_optimize_summary.csv", index=False)
    print("\n  wrote outputs/lr_optimize_summary.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if not a.report:
        run(".")
    report()


if __name__ == "__main__":
    main()
