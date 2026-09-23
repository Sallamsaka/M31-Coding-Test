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
CSV = Path("outputs/gbdt_programme.csv")


def _key(params: dict) -> tuple:
    """A cell's identity: every factor value, in a fixed order.

    Deliberately not the config index -- `plan()` emits a different list at a
    different budget, so an index would match the wrong cell after any change
    to the budget, and do it silently.
    """
    return tuple(round(float(params[k]), 10) for k in FACTORS)


def _done_cells() -> set:
    """(fold, key) pairs already in the ledger, so a re-launch resumes."""
    out = set()
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if all(k in r for k in FACTORS) and "fold" in r:
                    out.add((int(r["fold"]), _key(r)))
    return out

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


class _DropConstant:
    """Drop columns with fewer than two distinct non-NaN values.

    Works around a genuine sklearn 1.9 regression. `_find_binning_thresholds`
    now computes bin edges as `sliding_window_view(distinct_values, 2).mean(1)`,
    which raises `ValueError: window shape cannot be larger than input array
    shape` when a column is constant. Earlier versions wrote the same thing as
    `distinct_values[:-1] + distinct_values[1:]`, which is simply empty for a
    constant column and therefore harmless.

    **Why this is safe and not a modelling change.** A constant feature can
    never produce a split with any gain, so removing it cannot change a single
    prediction of a tree ensemble. This is a crash fix, not a tuning decision.

    **Why it has to be per-label and inside the estimator.** Measured on fold 0:
    233 columns are constant across the fold's 1,946 fit rows, but each label
    fits on its own at-risk subset and has 258-295 constant columns -- **25 to
    250 of them constant only within that label**, 1,485 extra across the 40.
    So a fold-level filter would not have fixed it. Putting the drop inside the
    estimator keeps one model per label over the FULL column set, so
    `feature_names` stays valid and nothing downstream has to know.

    Deliberately NOT applied to the shipped `fit_gbdt`: §G4's rule is that an
    unmeasured change must not re-baseline every number in the report. This is
    reached only through `_fit_per_label`'s `make_model` hook.
    """

    def __init__(self, est):
        self.est = est

    def fit(self, X, y=None, **kw):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # all-NaN column -> nan, handled
            mn = np.nanmin(X, axis=0)
            mx = np.nanmax(X, axis=0)
        self.keep_ = np.isfinite(mn) & np.isfinite(mx) & (mx > mn)
        if not self.keep_.any():                 # degenerate: keep one column
            self.keep_[0] = True
        self.est.fit(X[:, self.keep_], y, **kw)
        return self

    def predict_proba(self, X):
        return self.est.predict_proba(X[:, self.keep_])

    def predict(self, X):
        return self.est.predict(X[:, self.keep_])

    def __getattr__(self, name):
        """Delegate everything else to the wrapped estimator.

        `_positive_column` reads `clf.classes_` to find which column of
        `predict_proba` is the positive class -- it must not assume index 1,
        because a label whose fit subset happens to be single-class would put
        the wrong column there. Delegating generically rather than forwarding
        `classes_` alone means the next attribute the caller needs does not
        become a third failure on this path.

        Only reached when normal lookup fails, and `est`/`keep_` are excluded so
        a half-constructed object raises AttributeError instead of recursing.
        """
        if name in ("est", "keep_"):
            raise AttributeError(name)
        return getattr(self.__dict__["est"], name)


def _score_cell(root, fold_i, fit_rows, oof_rows, params) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier

    from .data.examples import ExampleConfig
    from .data.features import build_features
    from .data.labels import at_risk_mask, load_labels
    from .evaluate import macro_ap, macro_auroc
    from .train_baseline import GBDTConfig, _fit_per_label, apply_at_risk_mask

    F = build_features(root, ex_cfg=ExampleConfig(), fit_mask=fit_rows)
    lab = load_labels(root, ExampleConfig())
    y = lab["y"].astype(int)
    ar = at_risk_mask(lab)
    cfg = GBDTConfig(**params)
    # `_fit_per_label` directly rather than `fit_gbdt`, purely so the estimator
    # factory can be wrapped. Every other argument is what `fit_gbdt` passes.
    P = apply_at_risk_mask(
        _fit_per_label(
            F, y, ar,
            lambda: _DropConstant(HistGradientBoostingClassifier(
                learning_rate=cfg.learning_rate, max_iter=cfg.max_iter,
                max_leaf_nodes=cfg.max_leaf_nodes,
                min_samples_leaf=cfg.min_samples_leaf,
                l2_regularization=cfg.l2_regularization,
                early_stopping=cfg.early_stopping, max_bins=cfg.max_bins,
                random_state=0)),
            verbose=False, fit_mask=fit_rows), ar)
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
    # 1 is now in the ladder. Measured cost is ~17.9 min per (config, fold), so
    # at any realistic overnight budget `cells` is around 8 and both 5 and 3
    # folds fail the `>= 8 configs` test -- which dropped straight through to
    # "2 corners only", a head-to-head that estimates nothing. One fold still
    # estimates every main effect, and because all configs are scored on the
    # SAME fold the fold is common-mode and cancels in every contrast. What it
    # cannot do is estimate between-fold variance, so the winner is confirmed on
    # the full partition afterwards rather than trusted from the screen.
    for folds_try in (n_folds, 3, 1):
        if cells // folds_try >= 8:
            n_folds = folds_try
            break
    else:
        n_folds = 1
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

    # n_folds == 1 means "the FIRST fold of the standard 5-fold partition", not
    # a one-fold cross-validation. fold_masks(n_splits=1) puts every patient in
    # fold 0, so `fold != 0` is empty and build_features asserts "empty
    # fit_mask" -- which is exactly what it did, and is the right place for that
    # to fail loudly rather than silently fit on nothing.
    #
    # Taking fold 0 of K=5 also keeps the fit size (1,946) identical to what the
    # 5-fold and 3-fold branches use, so a screen run at one fold stays
    # comparable to a later confirmation run over the full partition.
    folds = (fold_masks(".", n_splits=5)[:1] if n_folds == 1
             else fold_masks(".", n_splits=n_folds))
    # RESUME. A cell is ~18 minutes and this programme has now died twice on its
    # LAST config -- once to the sklearn constant-column crash (D29), once to an
    # ArrayMemoryError on the largest cell -- losing all eight both times,
    # because it always started from an empty list. `arms.py` and
    # `design.confirm_runs` have had resume since D28.1 and this did not; that
    # asymmetry is the whole bug.
    #
    # Keyed on (fold, every factor value), not on a config index: `plan()`
    # returns a different config list at a different budget, so an index would
    # silently match the wrong cell after a budget change.
    done, rows = _done_cells(), []
    if CSV.exists():
        rows = pd.read_csv(CSV).to_dict("records")
    n_skip = sum(1 for c in configs for fi in range(len(folds))
                 if (fi, _key(c)) in done)
    if n_skip:
        print(f"  resuming: {n_skip} of {len(configs)*len(folds)} cells already"
              f" done, {len(configs)*len(folds)-n_skip} to run", flush=True)

    t0 = time.time()
    for ci, params in enumerate(configs):
        for fi, (fit_rows, oof_rows) in enumerate(folds):
            if (fi, _key(params)) in done:
                continue
            r = _score_cell(".", fi, fit_rows, oof_rows, params)
            rows.append(r)
            LEDGER.parent.mkdir(exist_ok=True)
            with LEDGER.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(r) + "\n")
        cur = pd.DataFrame(rows)
        # Select THIS config's rows by its factor values rather than by taking
        # the tail: with resume the tail may be rows loaded from disk in a
        # different order, and a mean over the wrong cells would look perfectly
        # plausible.
        m = pd.Series(True, index=cur.index)
        for k, v in params.items():
            m &= (cur[k] == v)
        sub = cur[m]
        if len(sub):
            sd = sub.ap.std(ddof=1) if len(sub) > 1 else float("nan")
            print(f"  config {ci+1}/{len(configs)}  AP {sub.ap.mean():.4f} "
                  f"+-{sd:.4f}  [{(time.time()-t0)/60:.0f} min]", flush=True)
        cur.to_csv(CSV, index=False)

    df = pd.DataFrame(rows)
    _analyse(df, folds)


def _analyse(df: pd.DataFrame, folds) -> None:
    """Main effects for a FRACTIONAL design, which is what the budget affords.

    ⚠ The bug this replaces. The module was written for a full 2^5 and used
    `paired_effects`, whose docstring is explicit that it needs exact twins --
    configurations differing in exactly one factor. `plan()` now returns a
    2^(5-2) resolution III fraction at any realistic budget, and a fraction has
    no twins, so `paired_effects` returned an EMPTY frame and the report printed
    "no two-level factors to contrast" while exiting 0. Eight cells and 2.4 CPU
    hours produced no effects table at all, and nothing failed.

    This is the same failure `orthogonal_effects` was written for on the
    transformer side (§E17), reached by a different route: there the design was
    fractional from the start, here it became fractional when the budget
    planner downgraded it.

    **Why a residual SE rather than Lenth.** Lenth's PSE needs m/3 degrees of
    freedom and the pool here is 5 main effects, giving df = 1 and
    t(0.975, 1) = 12.7 -- a margin that rejects nothing, which is §D27 exactly.
    With 8 runs and 6 parameters (intercept + 5 main effects) there are **2
    residual degrees of freedom**, and that is a weak but honest error estimate.
    Resolution III also aliases every main effect with two-factor interactions,
    so these are screening estimates and are labelled as such.
    """
    from scipy import stats

    names = list(FACTORS)
    X1 = np.column_stack([
        np.where(df[n].to_numpy(float) > np.mean(FACTORS[n]), 1.0, -1.0)
        for n in names])
    y = df["ap"].to_numpy(float)
    X = np.column_stack([np.ones(len(y)), X1])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    if dof < 1:
        print("\n  saturated: %d runs, %d parameters -- no residual df, so no"
              " intervals" % (len(y), X.shape[1]))
        se = float("nan")
    else:
        sigma = float(np.sqrt((resid ** 2).sum() / dof))
        se = sigma / np.sqrt(len(y))          # orthogonal +-1 coding
    eff = 2 * beta[1:]                        # contrast = 2 * coefficient

    print("\n=== MAIN EFFECTS on macro AP (GBDT, %d runs, resolution III) ==="
          % len(y))
    print("  %-20s %10s %10s %22s" % ("factor", "effect", "t", "95% CI"))
    order = np.argsort(-np.abs(eff))
    for i in order:
        if np.isfinite(se) and se > 0:
            t = eff[i] / (2 * se)
            lo = eff[i] - stats.t.ppf(0.975, dof) * 2 * se
            hi = eff[i] + stats.t.ppf(0.975, dof) * 2 * se
            print("  %-20s %+10.5f %10.2f   [%+.5f, %+.5f]"
                  % (names[i], eff[i], t, lo, hi))
        else:
            print("  %-20s %+10.5f %10s %22s" % (names[i], eff[i], "-", "-"))
    if np.isfinite(se) and se > 0:
        print("\n  residual sigma %.5f on %d df -> SE(effect) %.5f"
              % (sigma, dof, 2 * se))
        print("  ⚠ 2 df is a very weak error estimate: t(0.975, 2) = %.2f, so"
              " the intervals" % stats.t.ppf(0.975, 2))
        print("  are wide by construction. And resolution III aliases every main")
        print("  effect with two-factor interactions -- these are SCREENING")
        print("  estimates, to be confirmed at the promising corner, not quoted.")
    pd.DataFrame({"factor": names, "effect": eff,
                  "se": 2 * se if np.isfinite(se) else np.nan}
                 ).to_csv("outputs/gbdt_main_effects.csv", index=False)

    g = df.groupby(list(FACTORS))["ap"].mean()
    print(f"\n  best configuration by mean AP: {g.idxmax()}  -> {g.max():.4f}")
    print(f"  defaults for comparison: {tuple(DEFAULTS.values())}")
    print("\n  E13/E17 predict the optimum sits BELOW the defaults (fewer leaves,")
    print("  stronger L2, fewer iterations) because the task is capacity-saturated.")
    print("  Check the sign of each effect against that before believing it.")


if __name__ == "__main__":
    main()
