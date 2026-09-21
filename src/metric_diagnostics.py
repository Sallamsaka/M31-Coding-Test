"""Is macro AP a defensible metric to SELECT on? Measured, not assumed.

Run: ``python -m src.metric_diagnostics``

The selection rule ranks configurations by macro average precision. That was
inherited from the plan and never tested, and it deserves testing, because a
metric being *reported* and a metric being *reliable enough to choose with*
are different properties.

The specific worry: macro AP averages 40 per-label APs, and the median label
has 11 validation positives. Average precision is dominated by the head of
the ranking, so at 11 positives a single patient moving a few places can
swing a label's AP substantially. AUROC integrates over all
positive-negative pairs and should be far better behaved. If that is true,
selecting on AP means selecting on noise while believing otherwise.

Five diagnostics, each answering a different question:

1. **Sampling noise** -- bootstrap SE and coefficient of variation. How much
   does the metric move when nothing about the model changes?
2. **Resolving power** -- for two real models, how often does a bootstrap
   resample preserve their ordering? A metric that flips the winner half the
   time cannot select anything.
3. **Concentration** -- what share of the macro average comes from a handful
   of labels? A "macro over 40" that is really a macro over 4 is fragile in
   a way the name hides.
4. **Degeneracy** -- how often a label becomes unscorable under resampling.
5. **Rank agreement between metrics** -- do AUROC and AP even order the same
   models the same way? If they disagree, the choice of metric IS the choice
   of model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .data.examples import ExampleConfig
from .data.labels import at_risk_mask, load_labels

__all__ = ["main"]


def _load(root: Path, yv_shape) -> dict[str, np.ndarray]:
    out = {}
    npz = root / "artifacts" / "baseline_val_preds.npz"
    if npz.exists():
        d = np.load(npz)
        for k in d.files:
            if d[k].shape == yv_shape:
                out[k] = d[k]
    for f in sorted((root / "artifacts").glob("val_preds_*.npy")):
        a = np.load(f)
        if a.shape == yv_shape:
            out[f.stem.replace("val_preds_", "")] = a
    return out


def _macro(y, p, fn):
    vals = [fn(y[:, j], p[:, j]) for j in range(y.shape[1])
            if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(vals)) if vals else np.nan


def macro_auroc(y, p):
    return _macro(y, p, roc_auc_score)


def macro_ap(y, p):
    return _macro(y, p, average_precision_score)


def main(root: str | Path = ".") -> None:
    root = Path(root)
    lab = load_labels(root, ExampleConfig())
    y = lab["y"].astype(int)
    ar = at_risk_mask(lab)
    val = lab["split"] == "val"
    yv, arv = y[val], ar[val]

    preds = {k: np.where(arv.astype(bool), v, 1e-6)
             for k, v in _load(root, yv.shape).items()}
    if len(preds) < 2:
        print("need at least two models on disk; run the baseline first")
        return
    names = sorted(preds)
    print(f"models: {', '.join(names)}")
    npos = yv.sum(0)
    print(f"validation positives per label: median {int(np.median(npos))}  "
          f"min {int(npos.min())}  max {int(npos.max())}\n")

    rng = np.random.default_rng(0)
    B, n = 800, len(yv)
    idx = [rng.integers(0, n, n) for _ in range(B)]

    # ---- 1. sampling noise -------------------------------------------------
    print("1. SAMPLING NOISE  (patient bootstrap, model fixed)")
    print(f"   {'model':<16} {'AUROC':>8} {'SE':>7} {'CV':>7}   "
          f"{'AP':>8} {'SE':>7} {'CV':>7}")
    boots = {}
    for k in names:
        au = np.array([macro_auroc(yv[i], preds[k][i]) for i in idx])
        ap = np.array([macro_ap(yv[i], preds[k][i]) for i in idx])
        boots[k] = (au, ap)
        print(f"   {k:<16} {au.mean():>8.4f} {au.std():>7.4f} "
              f"{au.std()/au.mean():>6.1%}   {ap.mean():>8.4f} {ap.std():>7.4f} "
              f"{ap.std()/ap.mean():>6.1%}")
    print("   CV is the fair comparison: the two metrics live on different scales,")
    print("   so absolute SE cannot be compared directly.\n")

    # ---- 2. resolving power ------------------------------------------------
    print("2. RESOLVING POWER  (does a resample preserve the ordering?)")
    print(f"   {'pair':<26} {'AUROC agree':>12} {'AP agree':>10}")
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            ka, kb = names[a], names[b]
            au_t = macro_auroc(yv, preds[ka]) > macro_auroc(yv, preds[kb])
            ap_t = macro_ap(yv, preds[ka]) > macro_ap(yv, preds[kb])
            au_ag = np.mean([(macro_auroc(yv[i], preds[ka][i])
                              > macro_auroc(yv[i], preds[kb][i])) == au_t
                             for i in idx[:300]])
            ap_ag = np.mean([(macro_ap(yv[i], preds[ka][i])
                              > macro_ap(yv[i], preds[kb][i])) == ap_t
                             for i in idx[:300]])
            print(f"   {ka[:12]+' vs '+kb[:12]:<26} {au_ag:>11.1%} {ap_ag:>10.1%}")
    print("   50% means the metric is a coin flip on that pair.\n")

    # ---- 3. concentration --------------------------------------------------
    print("3. CONCENTRATION  (is 'macro over 40' really a macro over 40?)")
    for k in names[:3]:
        aps = np.array([average_precision_score(yv[:, j], preds[k][:, j])
                        for j in range(40) if 0 < yv[:, j].sum() < len(yv)])
        aus = np.array([roc_auc_score(yv[:, j], preds[k][:, j]) - 0.5
                        for j in range(40) if 0 < yv[:, j].sum() < len(yv)])
        s_ap = np.sort(aps)[::-1]
        s_au = np.sort(np.abs(aus))[::-1]
        print(f"   {k:<16} top-5 labels supply {s_ap[:5].sum()/aps.sum():>5.1%} of macro AP, "
              f"{s_au[:5].sum()/np.abs(aus).sum():>5.1%} of macro (AUROC-0.5)")
    print()

    # ---- 4. degeneracy -----------------------------------------------------
    drops = np.mean([(yv[i].sum(0) == 0).sum() for i in idx])
    print(f"4. DEGENERACY  mean labels unscorable per resample: {drops:.2f} of 40")
    print("   Each dropped label changes which labels the macro averages over,")
    print("   so replicates are not strictly comparable.\n")

    # ---- 5. do the metrics agree on the ranking? ---------------------------
    print("5. RANK AGREEMENT BETWEEN METRICS")
    order_au = sorted(names, key=lambda k: -macro_auroc(yv, preds[k]))
    order_ap = sorted(names, key=lambda k: -macro_ap(yv, preds[k]))
    print(f"   by AUROC: {' > '.join(order_au)}")
    print(f"   by AP   : {' > '.join(order_ap)}")
    print(f"   same winner: {'yes' if order_au[0] == order_ap[0] else 'NO'}")
    if order_au != order_ap:
        print("   The orderings differ, so the choice of selection metric is")
        print("   itself a choice of model, and must be justified rather than")
        print("   inherited.")


if __name__ == "__main__":
    main()
