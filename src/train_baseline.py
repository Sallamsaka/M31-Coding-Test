"""Prevalence, L2 logistic regression and gradient boosting, one model per label.

Three corrections to the obvious defaults, each from a measurement.

**`early_stopping` must be set explicitly.** sklearn's ``'auto'`` enables it
only when ``n > 10000``; at 2,791 it is silently off, so a config asking for
``max_iter=500`` with patience would in fact run a fixed 100 iterations with no
monitoring. And turning it on is worse: ``validation_fraction=0.1`` carves out
279 rows, about **2.9 expected positives** for the rarest label.

**Leaf size must be tiny.** ``min_samples_leaf=20`` is 69% of the rarest
label's entire positive class. EHRSHOT forces ``min_child_samples=1`` in code
(not in the paper) because otherwise the model "will refuse to learn anything".
Capacity is limited through ``max_leaf_nodes`` and L2 instead.

**Logistic regression is a contender, not a reference.** On EHRSHOT's
new-diagnosis tasks -- the same task family, at 793-1,392 training patients --
L2 logistic on counts scored **0.749 macro AUROC against the GBM's 0.719**,
while the GBM won on AP (0.212 vs 0.179). Expect the two metrics to disagree.

The mechanism is worth stating: a tree needs a *region* containing enough
positives to justify a split, and with 29 positives the leaves run out of data
after two or three. A linear model pools weak evidence additively across all
2,821 columns and never faces that constraint.

Two further choices, both deliberate:

* **Not-at-risk rows are dropped per label.** A patient already diagnosed with
  condition c cannot be an incident case for c, so training on them spends
  capacity learning a rule we apply exactly at inference anyway.
* **No class weighting, no resampling.** Measured at N=2,500 with 1% events:
  median AUROC after correction was *never higher*, while the calibration
  intercept went from +0.06 to below -4.5. Both our metrics are rank metrics,
  so there is nothing to win and calibration to lose.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler

from .data.features import FeatureMatrix

__all__ = ["GBDTConfig", "LRConfig", "fit_prevalence", "fit_lr", "fit_gbdt", "apply_at_risk_mask"]


@dataclass(frozen=True)
class GBDTConfig:
    learning_rate: float = 0.05
    max_iter: int = 300
    max_leaf_nodes: int = 15      # capacity lever, since leaf size must stay small
    min_samples_leaf: int = 5     # NOT 20 -- see module docstring
    l2_regularization: float = 1.0   # sklearn's default is 0.0
    early_stopping: bool = False     # explicit: 'auto' is off at n < 10000
    max_bins: int = 255


@dataclass(frozen=True)
class LRConfig:
    C: float = 0.03
    max_iter: int = 2000
    log1p_counts: bool = True     # monotone, so a no-op for trees; real for a linear model


# sklearn 1.8 deprecates `penalty="l2"` in favour of `l1_ratio=0`. Shown once
# rather than 40 times -- one per label -- because at 40 repetitions it buries
# the actual output. NOT silenced and NOT yet migrated: every number in the
# report was measured with the current parameterisation, and swapping it
# mid-project would quietly re-baseline the results for a cosmetic gain.
warnings.filterwarnings("once", category=FutureWarning,
                        module="sklearn.linear_model._logistic")


def _positive_column(clf, X: np.ndarray) -> np.ndarray:
    """Read P(y=1) by locating the class, never by hard-coding column 1.

    A constant fold or a degenerate label can order ``classes_`` differently,
    and indexing ``[:, 1]`` then silently inverts the predictions.
    """
    proba = clf.predict_proba(X)
    j = list(clf.classes_).index(1)
    return proba[:, j]


def _fit_rows(F: FeatureMatrix, fit_mask: np.ndarray | None) -> np.ndarray:
    """Rows a model may be FITTED on.

    ``None`` returns exactly ``F.split == "train"`` -- the historical value,
    reproduced as the same expression rather than an equivalent one, so the
    default path is unchanged by inspection.

    Anything else must select only training rows. The brief forbids validation
    or test entering a fit set, and a cross-validation harness that gets this
    wrong looks identical to one that gets it right, so the check is an assert
    here rather than a convention every caller has to remember.
    """
    if fit_mask is None:
        # Default excludes the locked test set (cv.trainable_pids). It used to
        # return `F.split == "train"`, which is 2,791 rows INCLUDING the locked
        # 358 -- so run_baseline, the path that ships predictions.csv, trained
        # on the held-out set.
        from .cv import trainable_pids
        pool = trainable_pids()
        return (F.split == "train") & np.isin(F.pid, list(pool))
    m = np.asarray(fit_mask, bool)
    assert m.shape == (len(F.split),), (
        f"fit_mask {m.shape} does not match {len(F.split)} rows")
    assert m.any(), "empty fit_mask"
    bad = int((m & (F.split != "train")).sum())
    assert not bad, f"fit_mask selects {bad} non-train rows"
    return m


def fit_prevalence(F: FeatureMatrix, y: np.ndarray,
                   fit_mask: np.ndarray | None = None) -> np.ndarray:
    """Predict the train base rate for everyone. Macro AUROC must be exactly 0.5.

    This is not a model; it is a test of the metric code. If it does not return
    0.500, the evaluation is broken and no other number means anything.
    """
    tr = _fit_rows(F, fit_mask)
    rate = y[tr].mean(0)
    return np.tile(rate, (len(y), 1)).astype(np.float32)


def _fit_per_label(F: FeatureMatrix, y: np.ndarray, at_risk: np.ndarray,
                   make_model, transform=None, verbose: bool = True,
                   models_out: list | None = None,
                   fit_mask: np.ndarray | None = None) -> np.ndarray:
    """Fit one model per label.

    ``models_out``, if given, collects the fitted estimators (or the constant
    used for a degenerate label) so the trained model can actually be shipped.
    Without it the models are fitted, used once for prediction and discarded,
    which makes `predictions.csv` unreproducible without a full retrain.
    """
    tr = _fit_rows(F, fit_mask)
    X = F.X if transform is None else transform(F.X)
    P = np.zeros_like(y, dtype=np.float32)
    t0 = time.time()
    for j in range(y.shape[1]):
        rows = tr & at_risk[:, j]          # drop not-at-risk rows from TRAINING only
        yj = y[rows, j]
        if yj.sum() < 2 or yj.sum() == len(yj):
            const = float(y[tr, j].mean())      # degenerate -> constant predictor
            P[:, j] = const
            if models_out is not None:
                models_out.append({"kind": "constant", "value": const})
            continue
        m = make_model()
        m.fit(X[rows], yj)
        P[:, j] = _positive_column(m, X)
        if models_out is not None:
            models_out.append({"kind": "model", "estimator": m})
        if verbose and (j + 1) % 10 == 0:
            print(f"    {j+1}/{y.shape[1]} labels  {time.time()-t0:.0f}s", flush=True)
    return P


def fit_lr(F: FeatureMatrix, y: np.ndarray, at_risk: np.ndarray,
           cfg: LRConfig | None = None, verbose: bool = True,
           models_out: list | None = None,
           preprocess_out: dict | None = None,
           fit_mask: np.ndarray | None = None) -> np.ndarray:
    """L2 logistic regression, one model per label.

    ``preprocess_out`` captures everything the fitted coefficients need in
    order to be applied to new data: the NaN fill values, whether counts were
    log1p'd, and the fitted scale. Those live inside the transform closure, so
    without this the saved estimators are unusable -- the coefficients would
    be applied to differently-scaled inputs and silently produce nonsense.
    """
    cfg = cfg or LRConfig()
    scaler = MaxAbsScaler()
    state: dict = {}
    # Hoisted out of the closure below so the imputation medians, the scaler
    # and _fit_per_label all use one array. MaxAbsScaler is the worst of the
    # ten fit sites to get wrong: it is an *extremum*, so a single held-out
    # outlier sets the column scale outright and no O(1/n) averaging dilutes it.
    tr0 = _fit_rows(F, fit_mask)

    def transform(X: np.ndarray) -> np.ndarray:
        # `lab_slope` is deliberately NaN below two readings, which the tree
        # reads natively and a linear model cannot read at all. Impute to the
        # TRAIN median per column rather than to zero: zero is a meaningful
        # slope ("this lab is flat"), so imputing it would assert flatness for
        # every patient who never had the test.
        if np.isnan(X).any():
            med = np.nanmedian(np.where(tr0[:, None], X, np.nan), axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            state["impute"] = med
            X = np.where(np.isnan(X), med, X)
        Z = np.log1p(np.clip(X, 0, None)) if cfg.log1p_counts else X
        scaler.fit(Z[tr0])                 # train-only scaling
        state["scale_"] = scaler.scale_
        state["log1p"] = cfg.log1p_counts
        return scaler.transform(Z).astype(np.float32)

    P = _fit_per_label(
        F, y, at_risk,
        lambda: LogisticRegression(penalty="l2", C=cfg.C, solver="lbfgs",
                                   max_iter=cfg.max_iter),
        transform=transform, verbose=verbose, models_out=models_out,
        fit_mask=fit_mask)
    if preprocess_out is not None:
        preprocess_out.update(state)
    return P


def fit_gbdt(F: FeatureMatrix, y: np.ndarray, at_risk: np.ndarray,
             cfg: GBDTConfig | None = None, verbose: bool = True,
             models_out: list | None = None,
             fit_mask: np.ndarray | None = None) -> np.ndarray:
    cfg = cfg or GBDTConfig()
    return _fit_per_label(
        F, y, at_risk,
        lambda: HistGradientBoostingClassifier(
            learning_rate=cfg.learning_rate, max_iter=cfg.max_iter,
            max_leaf_nodes=cfg.max_leaf_nodes, min_samples_leaf=cfg.min_samples_leaf,
            l2_regularization=cfg.l2_regularization,
            early_stopping=cfg.early_stopping, max_bins=cfg.max_bins,
            random_state=0),
        verbose=verbose, models_out=models_out, fit_mask=fit_mask)


def apply_at_risk_mask(P: np.ndarray, at_risk: np.ndarray,
                       floor: float = 1e-6) -> np.ndarray:
    """Push not-at-risk pairs to the bottom of the ranking.

    Guaranteed non-negative for AUROC (the rule is deterministic, so this makes
    ``A_prevalent`` exactly 1 rather than merely high) and non-negative for AP
    (moving true negatives below every positive cannot reduce precision at any
    recall level).

    A ``1e-6`` floor rather than a hard zero: if our notion of prevalence ever
    disagreed with the label builder's on even one patient, an exact zero would
    be unrecoverable, whereas a floor costs nothing measurable in either metric.
    """
    out = P.copy()
    out[~at_risk] = floor
    return out
