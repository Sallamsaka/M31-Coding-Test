"""End-to-end baseline: features -> four models -> validation report -> submission.

Run this and the assignment is answerable. Everything after it -- the
transformer, the ablations -- is an attempt to beat the number this prints,
which is why it exists before any of that and why it writes a real
`predictions.csv` rather than a placeholder.

The four models are deliberately unequal in ambition:

* **prevalence** is not a model. It predicts the training base rate for
  everyone, so its macro AUROC must come out at exactly 0.500. If it does
  not, the metric code is wrong and no other number on the page means
  anything. It is a test that happens to be shaped like a model.
* **logistic regression** is a contender, not a reference. On EHRSHOT's
  new-diagnosis tasks at 793-1,392 training patients it beat a GBM on AUROC
  (0.749 vs 0.719) while losing on AP (0.179 vs 0.212). Expect the two
  metrics to disagree here too.
* **gradient boosting** is the evidence-backed favourite at this scale.
* **the logit-average ensemble** is the one bet the evidence favours, and it
  is kept only because the paired bootstrap says it genuinely beats both
  parents rather than because averaging feels prudent.

Usage::

    python -m src.run_baseline                 # full, ~20 min
    python -m src.run_baseline --smoke         # LR only, ~30 s
    python -m src.run_baseline --no-gbdt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np

from .data.examples import ExampleConfig
from .data.cohort import load_target_descriptions
from .data.features import build_features
from .data.labels import at_risk_mask, load_labels
from .evaluate import (bootstrap_macro, macro_ap, macro_auroc,
                       paired_bootstrap_delta, per_code_table)
from .predict import write_predictions
from .train_baseline import (apply_at_risk_mask, fit_gbdt, fit_lr,
                             fit_prevalence)
from .utils import wandb_shim

__all__ = ["main"]


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--smoke", action="store_true", help="LR only, no bootstrap")
    ap.add_argument("--no-gbdt", action="store_true")
    ap.add_argument("--stride", type=int, default=0,
                    help="cutoff augmentation stride in years; 0 disables it")
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args(argv)

    root = Path(args.root)
    ex_cfg = (ExampleConfig() if args.stride == 0
              else ExampleConfig(augment=True, stride_years=args.stride))
    run = wandb_shim.init("baseline", {"stride": args.stride, "smoke": args.smoke})

    t0 = time.time()
    F = build_features(root, ex_cfg=ex_cfg)
    lab = load_labels(root, ex_cfg)
    y = lab["y"].astype(int)
    at_risk = at_risk_mask(lab)
    codes = [str(c) for c in lab["codes"]]
    n_real = int(lab["is_real"].sum())
    val = F.split == "val"
    assert F.is_real[val].all(), "validation must never be augmented"
    print(f"features {F.X.shape}  train {int((F.split=='train').sum()):,}  "
          f"val {int(val.sum())}  [{time.time()-t0:.0f}s]", flush=True)

    preds: dict[str, np.ndarray] = {}
    fitted: dict[str, list] = {}

    # --- the metric test, shaped like a model ------------------------------
    P = fit_prevalence(F, y)
    au, _ = macro_auroc(y[val], P[val])
    assert abs(au - 0.5) < 1e-9, f"prevalence-only scored {au}, must be exactly 0.5"
    print(f"SANITY  prevalence-only macro AUROC = {au:.6f}  (exactly 0.5 required)")
    preds["prevalence"] = P

    # --- logistic regression ----------------------------------------------
    t = time.time()
    lr_models: list = []
    lr_pre: dict = {}
    P = apply_at_risk_mask(
        fit_lr(F, y, at_risk, verbose=False, models_out=lr_models,
               preprocess_out=lr_pre), at_risk)
    preds["lr"] = P
    fitted["lr"] = lr_models
    print(f"LR    fit {time.time()-t:.0f}s", flush=True)

    # --- gradient boosting -------------------------------------------------
    if not (args.smoke or args.no_gbdt):
        t = time.time()
        gbdt_models: list = []
        P = apply_at_risk_mask(
            fit_gbdt(F, y, at_risk, verbose=False, models_out=gbdt_models), at_risk)
        preds["gbdt"] = P
        fitted["gbdt"] = gbdt_models
        print(f"GBDT  fit {time.time()-t:.0f}s", flush=True)
        preds["ensemble"] = _sigmoid(
            0.5 * (_logit(preds["lr"]) + _logit(preds["gbdt"])))

    # --- report -------------------------------------------------------------
    print(f"\n{'model':<12} {'macroAUROC':>11} {'macroAP':>9} {'AUROC@at-risk':>14}")
    scores = {}
    for name, P in preds.items():
        a, n_ok = macro_auroc(y[val], P[val])
        p, _ = macro_ap(y[val], P[val])
        ar, _ = macro_auroc(y[val], P[val], mask=at_risk[val])
        scores[name] = (a, p)
        print(f"{name:<12} {a:>11.4f} {p:>9.4f} {ar:>14.4f}   (n={n_ok})")
        run.log({f"{name}/macro_auroc": a, f"{name}/macro_ap": p,
                 f"{name}/auroc_at_risk": ar})

    best = max(scores, key=lambda k: scores[k][1])       # rank by val mAP
    print(f"\nselected by val macro AP: {best}")

    if not args.smoke:
        ci = bootstrap_macro(y[val], preds[best][val], n_boot=args.n_boot)
        for k, (pt, lo, hi) in ci.items():
            if k != "label_drops_in_replicates":
                print(f"  {k:<12} {pt:.4f}  [{lo:.4f}, {hi:.4f}]")
        if "ensemble" in preds:
            for other in ("lr", "gbdt"):
                d = paired_bootstrap_delta(y[val], preds["ensemble"][val],
                                           preds[other][val], n_boot=args.n_boot)
                pt, lo, hi, _ = d["d_macro_ap"]
                verdict = "REAL" if (lo > 0 or hi < 0) else "not resolvable"
                print(f"  ensemble - {other:<5} dAP {pt:+.4f} [{lo:+.4f},{hi:+.4f}] {verdict}")

        tbl = per_code_table(y[val], preds[best][val], codes,
                             lab["prevalent"][val],
                             load_target_descriptions(root))
        Path("outputs").mkdir(exist_ok=True)
        tbl.to_csv("outputs/per_code_val.csv", index=False)
        print(f"\nper-code table -> outputs/per_code_val.csv  "
              f"(lift median {tbl.lift.median():.1f}x)")

    # Validation predictions for every model, so `compare.py` can run the
    # paired bootstrap without refitting anything.
    Path("artifacts").mkdir(exist_ok=True)
    np.savez_compressed("artifacts/baseline_val_preds.npz",
                        **{k: v[val] for k, v in preds.items()})

    # --- persist the fitted model ------------------------------------------
    # Without this, `predictions.csv` cannot be regenerated without a full
    # retrain, and there is nothing to publish to the Hub. Saved alongside the
    # feature names and the at-risk rule, because a bag of estimators with no
    # record of its input columns is not a usable artifact.
    Path("artifacts").mkdir(exist_ok=True)
    for name, models in fitted.items():
        if name in (best, "lr"):
            joblib.dump({"models": models, "feature_names": F.names,
                         "preprocess": lr_pre if name == "lr" else {},
                         "codes": codes, "blocks": F.blocks,
                         "at_risk_rule": "prevalent == 0",
                         "val_macro_auroc": scores[name][0],
                         "val_macro_ap": scores[name][1]},
                        f"artifacts/model_{name}.joblib", compress=3)
            print(f"saved artifacts/model_{name}.joblib")

    # --- submission ----------------------------------------------------------
    # Real examples occupy the first len(cohort) rows with eid == pid, so the
    # real block is already in the cohort order `write_predictions` expects.
    sub = write_predictions(preds[best][:n_real], model_name=best, root=root,
                            extra_meta={"val_macro_auroc": scores[best][0],
                                        "val_macro_ap": scores[best][1],
                                        "stride": args.stride})
    print(f"wrote outputs/predictions.csv  {sub.shape}")
    run.summary(selected=best, val_macro_ap=scores[best][1])
    run.finish()


if __name__ == "__main__":
    main()
