"""Metrics, with the handling the small-sample regime actually requires.

Three things here are not the default and each is deliberate.

**Degenerate labels are excluded and counted, never imputed as 0.5.**
Imputation drags the macro toward chance and makes runs with different
degeneracy patterns incomparable. Every function returns ``(value, n_scored)``.

**Average precision is always reported next to its baseline.** A random
classifier scores AP equal to the prevalence, so a bare mAP is uninterpretable:
AP 0.15 is 15x chance on a 1% label and far *worse* than chance on a 26% one.
The per-code table carries ``prevalence`` and ``lift = AP / prevalence``.

**Bootstrapping resamples patients, never labels.** The 40 labels are fixed and
their per-patient errors are strongly correlated through shared features; a
naive pair-level resample treats them as independent and returns intervals that
are too narrow.

Measured context for anything computed here (Hanley-McNeil at n=2,791, true
AUC 0.75): SE is 0.053 at 29 positives, 0.029 at 100, 0.011 at 741. Pooled over
40 labels with realistic inter-label correlation, **SE(macro AUROC) is roughly
0.006-0.016 -- so macro differences below ~0.01 are not resolvable**, and a
per-label difference below ~0.10 on the rarest labels is pure noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

__all__ = [
    "macro_auroc", "macro_ap", "per_code_table",
    "bootstrap_macro", "paired_bootstrap_delta", "hanley_mcneil_se",
]


def _scorable(y: np.ndarray) -> np.ndarray:
    """Columns with at least one positive and one negative."""
    pos = y.sum(0)
    return (pos > 0) & (pos < len(y))


def macro_auroc(y: np.ndarray, p: np.ndarray,
                mask: np.ndarray | None = None) -> tuple[float, int]:
    """Unweighted mean per-label AUROC over scorable labels.

    ``mask`` restricts each label to its at-risk population, giving the
    incidence estimand rather than the grader-consistent full denominator.
    """
    vals = []
    for j in range(y.shape[1]):
        m = np.ones(len(y), bool) if mask is None else mask[:, j]
        yj, pj = y[m, j], p[m, j]
        if 0 < yj.sum() < len(yj):
            vals.append(roc_auc_score(yj, pj))
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def macro_ap(y: np.ndarray, p: np.ndarray,
             mask: np.ndarray | None = None) -> tuple[float, int]:
    vals = []
    for j in range(y.shape[1]):
        m = np.ones(len(y), bool) if mask is None else mask[:, j]
        yj, pj = y[m, j], p[m, j]
        if 0 < yj.sum() < len(yj):
            vals.append(average_precision_score(yj, pj))
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Analytic SE of a single AUROC (Hanley & McNeil 1982, formulae 1 and 2).

    Uses the negative-exponential approximations ``Q1 = A/(2-A)`` and
    ``Q2 = 2A^2/(1+A)``, which the original paper recommends as slightly
    conservative.
    """
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
           + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def per_code_table(y: np.ndarray, p: np.ndarray, codes: list[str],
                   prevalent: np.ndarray, descriptions: dict[str, str] | None = None
                   ) -> pd.DataFrame:
    """One row per condition, with both denominators side by side.

    ``auroc_all`` is what the graders compute. ``auroc_at_risk`` restricts to
    patients who could still become incident cases. The gap between them is
    ``(prevalent share of negatives) x (1 - auroc_at_risk)`` and is a property
    of the cohort composition, not of the model.
    """
    rows = []
    for j, c in enumerate(codes):
        yj, pj, prv = y[:, j], p[:, j], prevalent[:, j].astype(bool)
        n_pos, n_neg = int(yj.sum()), int((yj == 0).sum())
        a_all = roc_auc_score(yj, pj) if 0 < n_pos < len(yj) else np.nan
        ap = average_precision_score(yj, pj) if n_pos > 0 else np.nan
        pi = n_pos / len(yj)

        ar = ~prv
        a_ar = (roc_auc_score(yj[ar], pj[ar])
                if 0 < yj[ar].sum() < ar.sum() else np.nan)
        rows.append({
            "code": c,
            "description": (descriptions or {}).get(c, ""),
            "n_pos": n_pos, "n_prevalent": int(prv.sum()), "n_at_risk": int(ar.sum()),
            "prevalence": pi,
            "auroc_all": a_all, "auroc_at_risk": a_ar,
            "ap": ap, "lift": ap / pi if pi > 0 else np.nan,
            "se_hanley": hanley_mcneil_se(a_all, n_pos, n_neg) if np.isfinite(a_all) else np.nan,
        })
    return pd.DataFrame(rows)


def bootstrap_macro(y: np.ndarray, p: np.ndarray, n_boot: int = 2000,
                    seed: int = 0) -> dict[str, tuple[float, float, float]]:
    """Patient-level bootstrap. Returns ``{metric: (point, lo, hi)}``.

    With 5 positives on the rarest label, roughly 0.7% of replicates contain no
    positives for it. Such a label is dropped *from that replicate only*, and
    the count of affected replicates is returned so it can be reported.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    au, ap, dropped = [], [], 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        ok = _scorable(yb)
        dropped += int((~ok).sum())
        if not ok.any():
            continue
        au.append(macro_auroc(yb[:, ok], pb[:, ok])[0])
        ap.append(macro_ap(yb[:, ok], pb[:, ok])[0])

    def ci(v, point):
        return (point, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

    return {
        "macro_auroc": ci(au, macro_auroc(y, p)[0]),
        "macro_ap": ci(ap, macro_ap(y, p)[0]),
        "label_drops_in_replicates": (float(dropped), 0.0, 0.0),
    }


def paired_bootstrap_delta(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
                           n_boot: int = 2000, seed: int = 0) -> dict[str, tuple]:
    """CI on ``A - B`` using the *same* resamples for both models.

    Both models score identical patients, so shared patient-level noise cancels
    and the interval on the difference is far narrower than either marginal
    interval. Overlapping marginal CIs do not imply a non-significant
    difference, which is why this is the comparison that decides selection.

    **Also returns P(A > B)**, the share of resamples in which A wins. A
    significance gate on the interval is very conservative: Bouthillier et al.
    (MLSys 2021) measure a k=50 averaged comparison at under 5% false positives
    but roughly 90% false *negatives*, against about 5% and 30% for P(A > B)
    thresholded at 0.75. Reporting only "not resolvable" therefore discards most
    of the evidence. The probability costs one sign count in a loop that is
    already running, commits to no ranking, and is the quantity a *selection*
    decision actually needs -- see the selection-vs-inference split: intervals
    govern what we claim, P(A > B) governs what we pick.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    d_au, d_ap = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, ok = y[idx], _scorable(y[idx])
        if not ok.any():
            continue
        d_au.append(macro_auroc(yb[:, ok], p_a[idx][:, ok])[0]
                    - macro_auroc(yb[:, ok], p_b[idx][:, ok])[0])
        d_ap.append(macro_ap(yb[:, ok], p_a[idx][:, ok])[0]
                    - macro_ap(yb[:, ok], p_b[idx][:, ok])[0])
    d_au_a, d_ap_a = np.asarray(d_au), np.asarray(d_ap)
    return {
        "d_macro_auroc": (macro_auroc(y, p_a)[0] - macro_auroc(y, p_b)[0],
                          float(np.percentile(d_au, 2.5)), float(np.percentile(d_au, 97.5)),
                          float(np.std(d_au))),
        "d_macro_ap": (macro_ap(y, p_a)[0] - macro_ap(y, p_b)[0],
                       float(np.percentile(d_ap, 2.5)), float(np.percentile(d_ap, 97.5)),
                       float(np.std(d_ap))),
        # Ties count as half a win, so a model identical to its comparator
        # scores exactly 0.5 rather than 0.0 or 1.0 depending on float noise.
        "p_a_gt_b_auroc": float(np.mean(d_au_a > 0) + 0.5 * np.mean(d_au_a == 0)),
        "p_a_gt_b_ap": float(np.mean(d_ap_a > 0) + 0.5 * np.mean(d_ap_a == 0)),
        "n_resamples": float(len(d_au_a)),
    }
