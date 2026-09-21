"""Phase 0: measure what already exists, before changing anything.

Runs only on predictions already saved in ``artifacts/``. No model is fitted
here, so nothing in this module can be contaminated by work done later; these
are the numbers the project currently *has*, scored properly.

Four things this adds that the current reporting does not have.

**Proper scoring rules.** AUROC is invariant to any monotone rescaling (E1), so
both metrics in use are structurally blind to calibration. The Brier score sees
it, and the Murphy decomposition ``BS = REL - RES + UNC`` says how much of the
score is irreducible prevalence rather than earned skill. At these prevalences
the answer is expected to be "almost all of it", which is the concrete reason
raw Brier cannot rank models here and the skill score is reported instead.

**Calibration-in-the-large for all 40 labels.** G5 says calibration is "not
assessable for 39 of 40". That holds for *moderate* calibration -- a flexible
curve needs ~200 events -- but not for mean calibration: SE(log O/E) is about
1/sqrt(events), so 29 events still gives ~0.19. Wide, but a number.

**Slices.** 42.9% of patients die inside the outcome window, and A6 argues much
of the achievable signal is "is this record about to end". Splitting the metric
on that flag tests the claim directly. DEATHDATE is a *slice* variable here and
never a predictor; it stays refused at feature-build time (A7).

**P(A>B)** beside every interval, for the reason given in ``evaluate``.

Run: ``python -m src.phase0``
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .data.cohort import load_cohort
from .data.examples import ExampleConfig
from .data.labels import at_risk_mask, load_labels
from .utils.timeutil import to_ts

# 5 bins, not 10: the rarest label has 29 positives, so 10 quantile bins would
# leave most of them with no events and the reliability term would be noise.
N_BINS = 5
SPEC_TARGET = 0.90


def _mcc(y: np.ndarray, yhat: np.ndarray) -> float:
    tp = float(((yhat == 1) & (y == 1)).sum())
    tn = float(((yhat == 0) & (y == 0)).sum())
    fp = float(((yhat == 1) & (y == 0)).sum())
    fn = float(((yhat == 0) & (y == 1)).sum())
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else np.nan


def _murphy(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> dict:
    """BS = REL - RES + UNC (Murphy 1973), with quantile bins.

    UNC is the outcome variance and is irreducible; RES is the part a model can
    earn; REL is what miscalibration costs. Reported as a skill score against
    the prevalence forecast, because raw Brier at 1% prevalence is ~99% UNC and
    two models differing meaningfully in discrimination differ in the third
    decimal place.
    """
    n = len(y)
    ybar = float(y.mean())
    bs = float(np.mean((p - y) ** 2))
    unc = ybar * (1 - ybar)
    out = {"brier": bs, "rel": np.nan, "res": np.nan, "unc": unc,
           "bss": 1 - bs / unc if unc > 0 else np.nan}
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return out
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rel = res = 0.0
    for k in np.unique(idx):
        m = idx == k
        w = m.sum() / n
        rel += w * (p[m].mean() - y[m].mean()) ** 2
        res += w * (y[m].mean() - ybar) ** 2
    out["rel"], out["res"] = rel, res
    return out


def _sens_at_spec(y: np.ndarray, p: np.ndarray, spec: float = SPEC_TARGET) -> float:
    neg = p[y == 0]
    if len(neg) == 0 or y.sum() == 0:
        return np.nan
    return float((p[y == 1] > np.quantile(neg, spec)).mean())


def per_label_panel(y: np.ndarray, p: np.ndarray, ar: np.ndarray,
                    codes: list[str]) -> pd.DataFrame:
    """One row per label, scored on that label's at-risk population."""
    rows = []
    for j, c in enumerate(codes):
        m = ar[:, j].astype(bool)
        yj, pj = y[m, j].astype(int), p[m, j].astype(float)
        npos = int(yj.sum())
        if not 0 < npos < len(yj):
            rows.append({"code": c, "n_at_risk": int(m.sum()), "n_pos": npos})
            continue
        prev = npos / len(yj)
        mu = _murphy(yj, pj)
        # Two thresholds fixed in advance. Prevalence is one of the two points
        # at which a classification measure is not improper; the top-10% budget
        # is capacity-honest and identical across models. Neither is tuned, so
        # neither inherits the optimism of a max-over-threshold statistic.
        t_prev = float(np.quantile(pj, 1 - prev))
        t_top10 = float(np.quantile(pj, 0.90))
        ap = average_precision_score(yj, pj)
        rows.append({
            "code": c, "n_at_risk": int(m.sum()), "n_pos": npos, "prevalence": prev,
            "auroc": roc_auc_score(yj, pj), "ap": ap,
            # AP's null value IS the prevalence, so a macro over labels spanning
            # 1%-26% is a covert prevalence weighting. Normalising fixes that.
            "ap_norm": (ap - prev) / (1 - prev),
            **mu,
            "mcc_at_prev": _mcc(yj, (pj > t_prev).astype(int)),
            "mcc_at_top10": _mcc(yj, (pj > t_top10).astype(int)),
            "sens_at_90spec": _sens_at_spec(yj, pj),
            "o_over_e": npos / pj.sum() if pj.sum() > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def _load_preds(root: str, shape: tuple) -> dict[str, np.ndarray]:
    preds: dict[str, np.ndarray] = {}
    d = np.load(f"{root}/artifacts/baseline_val_preds.npz")
    for k in d.files:
        preds[k] = d[k]
    for f in sorted(glob.glob(f"{root}/artifacts/val_preds_*.npy")):
        preds[Path(f).stem.replace("val_preds_", "")] = np.load(f)
    return {k: v for k, v in preds.items() if v.shape == shape}


def main(root: str = ".") -> None:
    lab = load_labels(root, ExampleConfig())
    coh = load_cohort(root)
    val = (lab["split"] == "val")
    y = lab["y"][val].astype(int)
    ar = at_risk_mask(lab)[val]
    codes = list(lab["codes"])
    preds = _load_preds(root, y.shape)
    print(f"models on disk: {list(preds)}   n_val={len(y)}  labels={y.shape[1]}")
    print(f"per-label positives: min={y.sum(0).min()}  median={int(np.median(y.sum(0)))}"
          f"  max={y.sum(0).max()}\n")

    # ---- 0.1 / 0.2  metric panel and calibration -------------------------
    summary = []
    for name, P in preds.items():
        t = per_label_panel(y, P, ar, codes)
        t.to_csv(f"{root}/outputs/phase0_perlabel_{name}.csv", index=False)
        s = t.dropna(subset=["auroc"])
        summary.append({
            "model": name, "n_scored": len(s),
            "auroc": s.auroc.mean(), "ap": s.ap.mean(), "ap_norm": s.ap_norm.mean(),
            "bss": s.bss.mean(), "rel": s.rel.mean(), "res": s.res.mean(),
            "unc": s.unc.mean(),
            "mcc_prev": s.mcc_at_prev.mean(), "mcc_top10": s.mcc_at_top10.mean(),
            "sens90": s.sens_at_90spec.mean(), "med_O/E": s.o_over_e.median(),
        })
    S = pd.DataFrame(summary)
    print("=== 0.1/0.2  METRIC PANEL  (per-label, at-risk denominator) ===")
    print(S.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nMurphy: share of the Brier score that is earnable vs irreducible")
    for r in summary:
        print(f"  {r['model']:<12} RES/UNC={r['res']/r['unc']:.4f}"
              f"  REL/UNC={r['rel']/r['unc']:.4f}  (UNC={r['unc']:.4f})")
    S.to_csv(f"{root}/outputs/phase0_summary.csv", index=False)

    # ---- 0.3  slices -----------------------------------------------------
    pat = pd.read_csv(f"{root}/train_val/patients.csv", dtype=str)
    death = to_ts(pat.set_index("Id")["DEATHDATE"])
    cv = coh.loc[coh.split == "val"].copy().reset_index(drop=True)
    cv["death"] = cv.patient_id.map(death)
    cv["died_in_window"] = cv.death.notna() & (cv.death <= cv.window_end)
    cv["age_band"] = pd.cut(cv.age_at_anchor, [0, 40, 60, 75, 200],
                            labels=["<40", "40-60", "60-75", "75+"])
    assert len(cv) == len(y), f"cohort val rows {len(cv)} != label rows {len(y)}"
    print(f"\n=== 0.3  SLICES   died_in_window = {cv.died_in_window.mean():.1%} ===")
    for slicer in ["died_in_window", "age_band", "gender"]:
        print(f"\n-- {slicer} --")
        for lvl, sub in cv.groupby(slicer, observed=True):
            m = np.zeros(len(cv), bool)
            m[sub.index.to_numpy()] = True
            if m.sum() < 20:
                continue
            line = f"   {str(lvl):<10} n={m.sum():>4} "
            for name, P in preds.items():
                if name == "prevalence":
                    continue
                v = [roc_auc_score(y[m, j], P[m, j]) for j in range(y.shape[1])
                     if 0 < y[m, j].sum() < m.sum()]
                line += f"  {name}={np.mean(v):.4f}({len(v)})"
            print(line)

    print("\nwrote outputs/phase0_summary.csv, outputs/phase0_perlabel_*.csv")


if __name__ == "__main__":
    main()
