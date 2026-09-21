"""Does a minimum-history floor rescue augmentation? The C1 follow-up that never ran.

§C1 measured augmentation as a failure and the plan treated it as one. Re-reading
the 2x2 says it is not:

    macro AUROC          eval real      eval synthetic
    train real   (2,791)  **0.7679**        0.6345
    train synth  (2,791)    0.6772        **0.7548**
    train synth (12,864)    0.6942        **0.7655**

Synthetic rows are perfectly learnable *in their own domain* and they scale
correctly there (0.7548 -> 0.7655). They simply do not transfer. That is a
**domain shift**, not a broken idea, and §C1 measured the mechanism: synthetic
cutoffs are **7x sparser** (26 vs 189 pre-cutoff events) and 13 years younger
(median age 50 vs 64).

So the fix is to stop generating rows from a different population. The knob
already exists -- `ExampleConfig.min_events` -- and §C1 ran at the default 10,
which admits a cutoff with as few as 10 prior events against a real median of
189. Raising it trades **volume for similarity**: fewer synthetic rows, each
closer to the real population the model is scored on.

**What makes this a real experiment rather than a knob-twiddle.** The trade has
an interior optimum by construction:

  * at a low floor  -> many rows, but drawn from the wrong population (§C1)
  * at a high floor -> rows resemble real cutoffs, but there are almost none,
    and augmentation with zero extra rows is exactly the baseline

So the curve must rise and then fall, and a two-point comparison would miss it.
The sweep reports the row count at every floor, because "it did not help" and
"it produced 40 extra rows" are different findings and the second one is not
about augmentation at all.

**Scoring.** `cv.fold_masks` already scores `is_real` rows only, so synthetic
rows can enter a fit set and can never enter an evaluation set -- which is the
property that makes the comparison interpretable. Real-cutoff validation, always.

Run: ``python -m src.augment_probe``
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

LEDGER = Path("outputs/augment_probe.jsonl")

# The real-cutoff median is 189 pre-cutoff events (§C1). A floor of 10 is what
# C1 ran; the rest walk toward the real population.
FLOORS = (10, 25, 50, 100, 150, 200)


def _one(root: str, floor: int | None, n_splits: int = 5) -> dict:
    """One floor, scored by K=5 CV on REAL rows only. floor=None -> no augmentation."""
    from .cv import fold_masks
    from .data.examples import ExampleConfig, load_examples
    from .data.features import build_features
    from .data.labels import at_risk_mask, load_labels
    from .evaluate import macro_ap, macro_auroc
    from .train_baseline import LRConfig, apply_at_risk_mask, fit_lr

    ex_cfg = (ExampleConfig() if floor is None
              else ExampleConfig(augment=True, min_events=floor))
    ex = load_examples(root, ex_cfg)
    n_syn = int((~ex.is_real).sum())

    lab = load_labels(root, ex_cfg)
    y = lab["y"].astype(int)
    ar = at_risk_mask(lab)
    folds = fold_masks(root, n_splits=n_splits, ex_cfg=ex_cfg)

    au, ap = [], []
    for fit_rows, oof_rows in folds:
        F = build_features(root, ex_cfg=ex_cfg, fit_mask=fit_rows)
        P = apply_at_risk_mask(
            fit_lr(F, y, ar, LRConfig(), verbose=False, fit_mask=fit_rows), ar)
        idx = np.flatnonzero(oof_rows)
        au.append(macro_auroc(y[idx], P[idx], mask=ar[idx])[0])
        ap.append(macro_ap(y[idx], P[idx], mask=ar[idx])[0])
    return {"floor": floor, "n_rows": len(ex), "n_synthetic": n_syn,
            "auroc": float(np.mean(au)), "auroc_sd": float(np.std(au, ddof=1)),
            "ap": float(np.mean(ap)), "ap_sd": float(np.std(ap, ddof=1))}


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--floors", type=int, nargs="+", default=list(FLOORS))
    a = ap_.parse_args()

    print("min-history floor sweep -- augmentation, scored on REAL cutoffs only")
    print("  real-cutoff median pre-cutoff events = 189; C1 ran at floor 10\n")
    print(f"{'floor':>7}{'rows':>9}{'synthetic':>11}{'macro AUROC':>20}{'macro AP':>20}")

    rows = []
    for floor in [None] + list(a.floors):
        t = time.time()
        r = _one(".", floor)
        r["minutes"] = (time.time() - t) / 60
        rows.append(r)
        tag = "none" if floor is None else str(floor)
        print(f"{tag:>7}{r['n_rows']:>9,}{r['n_synthetic']:>11,}"
              f"{r['auroc']:>13.4f} +-{r['auroc_sd']:.4f}"
              f"{r['ap']:>13.4f} +-{r['ap_sd']:.4f}", flush=True)
        LEDGER.parent.mkdir(exist_ok=True)
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(r) + "\n")

    df = pd.DataFrame(rows)
    df.to_csv("outputs/augment_probe.csv", index=False)

    base = df[df.floor.isna()].iloc[0]
    best = df[df.floor.notna()].loc[df[df.floor.notna()].ap.idxmax()]
    print(f"\n  baseline (no augmentation): AP {base.ap:.4f}")
    print(f"  best floor {int(best.floor)}: AP {best.ap:.4f} "
          f"({best.n_synthetic:,} synthetic rows)")
    d = best.ap - base.ap
    # Across-fold SD of the difference is not available without the paired
    # per-fold values, so this is deliberately a scale check, not a test.
    print(f"  difference {d:+.4f}, against a per-fold SD of ~{base.ap_sd:.4f}")
    print("\n  A floor that helps AND produces few rows is not an augmentation")
    print("  result -- check n_synthetic before believing the AP column.")


if __name__ == "__main__":
    main()
