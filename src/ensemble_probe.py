"""Which models are worth combining, and is a weak-but-different model useful?

Measured on the predictions already on disk -- no training. The whole point of
saving predictions from every expensive run is that this costs seconds.

**The finding that motivated it.** On the 365-patient validation set, with an
UNTUNED transformer:

    lr                        AP 0.2473      alone
    gbdt                      AP 0.2623      alone
    P3 (transformer)          AP 0.1832      alone -- far the worst
    ---
    lr + gbdt                 AP 0.2702
    gbdt + P3                 AP 0.2714      BEATS lr + gbdt
    lr + gbdt + P3            AP 0.2743      best

**A model that is much worse alone is a better partner than one that is
better alone.** The agreement matrix says why: lr and gbdt agree at Spearman
0.824 -- two feature models reading the same matrix -- while gbdt and the
transformer agree at 0.751, the lowest of any strong pair. Ensembles are paid in
disagreement, not in accuracy.

That reframes the transformer's role here. It loses every head-to-head and still
earns its place, so the question "does the transformer beat GBDT" was the wrong
one; "does it disagree usefully with GBDT" is the right one.

**Why intervals, not point estimates.** These differences are 0.001 to 0.007 AP
on 365 patients, well inside the range this project has repeatedly failed to
resolve. Every comparison here is reported with a paired bootstrap and P(A>B),
because "0.2714 beats 0.2702" is not a result on its own.

**Caveat carried into every number.** The validation set is a SELECTION set with
~26 reads (§G1). Agreement between models is not distorted by that -- two models
either rank patients alike or they do not -- but the ensemble scores are. The CV
out-of-fold predictions are the de-biased version and this reads them when they
exist.

Run: ``python -m src.ensemble_probe``            (validation predictions)
     ``python -m src.ensemble_probe --oof``      (cross-validated, de-biased)
"""

from __future__ import annotations

import argparse
import glob
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data.examples import ExampleConfig
from .data.labels import at_risk_mask, load_labels
from .evaluate import macro_ap, macro_auroc, paired_bootstrap_delta

MAX_COMBO = 3        # beyond three, the count explodes and every extra member
                     # is a further multiplicity test on a selection set


def _logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def blend(preds: dict, keys) -> np.ndarray:
    """Equal-weight logit average -- the same rule run_baseline ships.

    Equal weights on purpose: fitting blend weights on the same data used to
    score them is another selection step, and at 365 patients it would find
    weights that do not transfer.
    """
    return 1.0 / (1.0 + np.exp(-sum(_logit(preds[k]) for k in keys) / len(keys)))


def load_val_preds(root: str = ".") -> tuple[dict, np.ndarray, np.ndarray]:
    lab = load_labels(root, ExampleConfig())
    val = (lab["split"] == "val")
    y = lab["y"][val].astype(int)
    ar = at_risk_mask(lab)[val]
    P: dict[str, np.ndarray] = {}
    npz = Path(root) / "artifacts" / "baseline_val_preds.npz"
    if npz.exists():
        d = np.load(npz)
        for k in d.files:
            # `prevalence` is a constant predictor and `ensemble` is already a
            # blend -- including either would flatter every combination.
            if d[k].shape == y.shape and k not in ("prevalence", "ensemble"):
                P[k] = d[k]
    for f in sorted(glob.glob(f"{root}/artifacts/val_preds_*.npy")):
        a = np.load(f)
        if a.shape == y.shape:
            P[Path(f).stem.replace("val_preds_", "")] = a
    return P, y, ar


def load_oof_preds(root: str = ".") -> tuple[dict, np.ndarray, np.ndarray]:
    """Cross-validated out-of-fold predictions -- the de-biased instrument."""
    f = Path(root) / "artifacts" / "cv_oof_preds.npz"
    if not f.exists():
        raise SystemExit("no artifacts/cv_oof_preds.npz -- run cross_validate first")
    d = np.load(f)
    scored = d["scored"].astype(bool)
    lab = load_labels(root, ExampleConfig())
    y = lab["y"][scored].astype(int)
    ar = at_risk_mask(lab)[scored]
    P = {k: d[k][scored] for k in d.files
         if k not in ("scored", "prevalence") and not k.endswith("__rank")
         and d[k].shape[0] == len(scored)}
    return P, y, ar


def agreement(P: dict, ar: np.ndarray) -> pd.DataFrame:
    m = ar.astype(bool)
    ks = list(P)
    return pd.DataFrame(
        [[spearmanr(P[a][m], P[b][m]).statistic for b in ks] for a in ks],
        index=ks, columns=ks)


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--oof", action="store_true",
                     help="use cross-validated OOF predictions instead of val")
    a = ap_.parse_args()

    P, y, ar = (load_oof_preds() if a.oof else load_val_preds())
    if len(P) < 2:
        raise SystemExit(f"need at least two models, found {list(P)}")
    src = "CROSS-VALIDATED OOF (de-biased)" if a.oof else "validation (SELECTION set)"
    print(f"{len(P)} models on {len(y):,} rows -- {src}\n")

    def sc(q):
        return macro_auroc(y, q, mask=ar)[0], macro_ap(y, q, mask=ar)[0]

    print("SINGLE MODELS")
    singles = {}
    for k in P:
        au, apv = sc(P[k])
        singles[k] = apv
        print(f"  {k:<18} AUROC {au:.4f}   AP {apv:.4f}")

    print("\nAGREEMENT (Spearman, at-risk entries). Lower = more to gain.")
    A = agreement(P, ar)
    print("  " + A.to_string(float_format=lambda v: f"{v:.3f}"))

    rows = []
    for r in range(2, min(MAX_COMBO, len(P)) + 1):
        for combo in itertools.combinations(P, r):
            au, apv = sc(blend(P, combo))
            rows.append({"members": "+".join(combo), "n": r,
                         "auroc": au, "ap": apv,
                         "best_single": max(singles[k] for k in combo),
                         "mean_agreement": float(np.mean(
                             [A.loc[x, z] for x, z in itertools.combinations(combo, 2)]))})
    E = pd.DataFrame(rows).sort_values("ap", ascending=False).reset_index(drop=True)
    E["lift_over_best_member"] = E.ap - E.best_single

    print("\nENSEMBLES (equal-weight logit average), ranked by macro AP")
    print(E.head(8).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best_single_name = max(singles, key=singles.get)
    top = E.iloc[0]
    print(f"\n  best single model : {best_single_name} (AP {singles[best_single_name]:.4f})")
    print(f"  best ensemble     : {top.members} (AP {top.ap:.4f})")

    d = paired_bootstrap_delta(y, blend(P, top.members.split("+")), P[best_single_name])
    pt, lo, hi, _ = d["d_macro_ap"]
    tag = "RESOLVABLE" if (lo > 0 or hi < 0) else "not resolvable"
    print(f"  ensemble - best single: dAP {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]  {tag}"
          f"   P(ens>single) = {d['p_a_gt_b_ap']:.2f}")

    print("\n  Does a WORSE model make a BETTER partner?")
    for _, r in E.head(4).iterrows():
        print(f"    {r.members:<30} agreement {r.mean_agreement:.3f}"
              f"   lift over its best member {r.lift_over_best_member:+.4f}")
    if not a.oof:
        print("\n  Validation is a selection set (~26 reads). Agreement is unaffected")
        print("  by that; the ensemble SCORES are not. Re-run with --oof once the")
        print("  cross-validated predictions exist.")

    E.to_csv("outputs/ensemble_probe.csv", index=False)
    print("\nwrote outputs/ensemble_probe.csv")


if __name__ == "__main__":
    main()
