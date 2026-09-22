"""Tune the model that actually wins. GBDT has never had a single knob turned.

**The asymmetry this closes.** Roughly ten CPU-hours have gone into the
transformer -- seed sigma, a learning-rate basin, seven ablation arms, a 20-run
designed experiment -- and **zero** into GBDT, which beats it on every
instrument. On the cross-validated pool, the only de-biased comparison this
project has:

    gbdt   macro AP 0.2174 +- 0.0074      <- untuned defaults
    lr     macro AP 0.2042 +- 0.0122      <- C swept
    gbdt - lr  +0.0157 [+0.0059, +0.0249]  RESOLVABLE, P = 1.00

Spending the compute on the model that loses and none on the model that wins is
backwards, and the gap is entirely a historical accident: GBDT was added as a
baseline and never revisited.

**What prior findings say to expect.** §E13's Riley analysis puts the median
label at **10.5 supportable parameters** against 3,296 columns fed, and §E17
found every feature block null at 8x the old resolution. Both point the same way:
this task is **capacity-saturated**, so the levers that should matter are the
ones that REDUCE effective capacity -- fewer leaves, stronger L2, fewer
iterations -- not the ones that add it. If that is right, the tuned optimum sits
*below* the defaults, which is the same direction the transformer's capacity
factor was shifted for the same reason.

**Costing, measured rather than assumed.** A GBDT fit is 40 per-label models
over ~3,296 columns and is expensive -- `run_baseline` took 1,598 s for one.
A full 2^5 factorial at K=5 would be 160 fits and roughly a day. So the design is
sized from a timed fit: ``--budget-minutes`` picks the largest design that fits,
preferring a resolution-IV fraction over dropping factors, because §E14 showed
that at fixed N the number of factors does not change main-effect precision.

**Scored on the CV pool**, not validation. 2,433 rows resolve what 365 cannot,
and it is the instrument that produced the only resolvable model difference we
have. Paired contrasts via `effects.paired_effects`, which is correct here
because a full factorial has exact twins -- unlike the transformer design, where
a 2^(5-1) fraction required orthogonal contrasts instead (§E17).

Run: ``python -m src.gbdt_programme --time-one``   measure one fit, then size
     ``python -m src.gbdt_programme --budget-minutes 90``
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

LEDGER = Path("outputs/gbdt_programme.jsonl")

# Levels straddle the current defaults, with the LOW side pushed further because
# E13/E17 predict the optimum sits below them.
FACTORS = {
    "learning_rate":     (0.02, 0.10),     # default 0.05
    "max_leaf_nodes":    (7, 31),          # default 15
    "max_iter":          (150, 400),       # default 300
    "l2_regularization": (0.1, 10.0),      # default 1.0
    "min_samples_leaf":  (5, 25),          # default 5
}
DEFAULTS = dict(learning_rate=0.05, max_leaf_nodes=15, max_iter=300,
                l2_regularization=1.0, min_samples_leaf=5)


def _score_cell(root, fold_i, fit_rows, oof_rows, params) -> dict:
    from .data.examples import ExampleConfig
    from .data.features import build_features
    from .data.labels import at_risk_mask, load_labels
    from .evaluate import macro_ap, macro_auroc
    from .train_baseline import GBDTConfig, apply_at_risk_mask, fit_gbdt

    F = build_features(root, ex_cfg=ExampleConfig(), fit_mask=fit_rows)
    lab = load_labels(root, ExampleConfig())
    y = lab["y"].astype(int)
    ar = at_risk_mask(lab)
    P = apply_at_risk_mask(
        fit_gbdt(F, y, ar, GBDTConfig(**params), verbose=False, fit_mask=fit_rows), ar)
    idx = np.flatnonzero(oof_rows)
    au, n_ok = macro_auroc(y[idx], P[idx], mask=ar[idx])
    ap, _ = macro_ap(y[idx], P[idx], mask=ar[idx])
    return {"fold": fold_i, "auroc": au, "ap": ap, "n_scored": n_ok, **params}


def time_one(root=".") -> float:
    """Minutes for ONE (config, fold) cell, so the design can be sized honestly."""
    from .cv import fold_masks
    fit_rows, oof_rows = fold_masks(root, n_splits=5)[0]
    t = time.time()
    r = _score_cell(root, 0, fit_rows, oof_rows, DEFAULTS)
    m = (time.time() - t) / 60
    print(f"  one cell at the defaults: {m:.1f} min   (AUROC {r['auroc']:.4f}, "
          f"AP {r['ap']:.4f})")
    return m


def plan(minutes_per_cell: float, budget_minutes: float,
         n_folds: int = 5) -> tuple[list[dict], int]:
    """Largest affordable design. Returns (configs, folds_to_use).

    Order of concessions, and the order matters:

      1. use a FRACTION before dropping factors -- §E14 showed that at fixed N
         the number of factors does not change main-effect precision, so
         dropping one answers fewer questions for the same compute;
      2. use FEWER FOLDS before dropping to a two-corner comparison -- a design
         on 3 folds still estimates every main effect, while two corners
         estimates nothing and is just a head-to-head.

    Falling straight from "cannot afford 8 configs on 5 folds" to "run 2 corners"
    was the first version's behaviour and it threw away a usable design.
    """
    names = list(FACTORS)
    cells = int(budget_minutes / max(minutes_per_cell, 1e-9))
    for folds_try in (n_folds, 3):
        if cells // folds_try >= 8:
            n_folds = folds_try
            break
    else:
        n_folds = 3
    n_cfg = max(2, cells // n_folds)

    full = list(itertools.product([0, 1], repeat=len(names)))          # 32
    if n_cfg >= 32:
        combos, label = full, "2^5 full factorial"
    elif n_cfg >= 16:
        # 2^(5-1), generator E = ABCD: resolution V, all 2fi unaliased.
        combos = [c for c in full if c[4] == (c[0] ^ c[1] ^ c[2] ^ c[3])]
        label = "2^(5-1) resolution V"
    elif n_cfg >= 8:
        # 2^(5-2), D = AB, E = AC: resolution III, main effects only.
        combos = [c for c in full if c[3] == (c[0] ^ c[1]) and c[4] == (c[0] ^ c[2])]
        label = "2^(5-2) resolution III (main effects only)"
    else:
        combos = [(0,) * 5, (1,) * 5]
        label = "2 corners only -- budget too small for a design"
    print(f"  budget {budget_minutes:.0f} min / {minutes_per_cell:.1f} min per cell "
          f"= {cells} cells -> {len(combos)} configs x {n_folds} folds")
    print(f"  design: {label}")
    return ([{n: FACTORS[n][c[i]] for i, n in enumerate(names)} for c in combos],
            n_folds)


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--time-one", action="store_true")
    ap_.add_argument("--budget-minutes", type=float, default=90.0)
    ap_.add_argument("--minutes-per-cell", type=float, default=None,
                     help="skip the timing fit and use this instead")
    a = ap_.parse_args()

    from .cv import fold_masks
    print("GBDT tuning programme -- scored on the CV pool, not validation\n")

    if a.time_one:
        time_one(".")
        return
    mpc = a.minutes_per_cell if a.minutes_per_cell else time_one(".")
    configs, n_folds = plan(mpc, a.budget_minutes)

    folds = fold_masks(".", n_splits=n_folds)
    rows = []
    t0 = time.time()
    for ci, params in enumerate(configs):
        for fi, (fit_rows, oof_rows) in enumerate(folds):
            r = _score_cell(".", fi, fit_rows, oof_rows, params)
            rows.append(r)
            LEDGER.parent.mkdir(exist_ok=True)
            with LEDGER.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(r) + "\n")
        done = pd.DataFrame(rows)
        sub = done[done.fold >= 0].tail(len(folds))
        print(f"  config {ci+1}/{len(configs)}  AP {sub.ap.mean():.4f} "
              f"+-{sub.ap.std(ddof=1):.4f}  [{(time.time()-t0)/60:.0f} min]", flush=True)
        done.to_csv("outputs/gbdt_programme.csv", index=False)

    df = pd.DataFrame(rows)
    from .effects import paired_effects, report
    # A FULL factorial has exact twins, so pairing is correct here -- unlike the
    # transformer design, whose 2^(5-1) fraction needed orthogonal contrasts.
    me = paired_effects(df, list(FACTORS), "ap", fold_col="fold")
    report(me, "macro AP (GBDT, CV pool)")
    me.to_csv("outputs/gbdt_main_effects.csv", index=False)

    g = df.groupby(list(FACTORS))["ap"].mean()
    print(f"\n  best configuration by mean AP: {g.idxmax()}  -> {g.max():.4f}")
    print(f"  defaults for comparison: {tuple(DEFAULTS.values())}")
    print("\n  E13/E17 predict the optimum sits BELOW the defaults (fewer leaves,")
    print("  stronger L2, fewer iterations) because the task is capacity-saturated.")
    print("  Check the sign of each effect against that before believing it.")


if __name__ == "__main__":
    main()
