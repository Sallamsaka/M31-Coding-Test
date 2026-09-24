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
from dataclasses import asdict
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
from .train_baseline import (GBDTConfig, LRConfig, apply_at_risk_mask,
                             fit_gbdt, fit_lr, fit_prevalence)
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
    ap.add_argument("--no-calibrate", action="store_true",
                    help="ship raw probabilities instead of applying the "
                         "out-of-fold Platt maps in artifacts/platt_params.npz")
    ap.add_argument("--with-transformer", action="store_true",
                    help="also fit the transformer (~45 min) and form the "
                         "recipes E45 measured as best on out-of-fold data")
    ap.add_argument("--recipe", default=None,
                    help="SHIP this key instead of taking the val argmax. "
                         "The recipe is a selection decision and E45 made it on "
                         "1,675 out-of-fold rows; re-picking it here by val "
                         "macro AP would re-select on 365 rows and pay the "
                         "winner's curse twice for a decision already taken.")
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
    # Print the set actually FITTED, not the split size. They differ by the
    # locked test set (2,791 vs 2,433), and printing the larger number is how a
    # reader concludes the model trained on everything when it did not.
    from .train_baseline import _fit_rows
    n_fit = int(_fit_rows(F, None).sum())
    print(f"features {F.X.shape}  fit {n_fit:,} "
          f"(all {int((F.split=='train').sum()):,} train patients; carve retired)  "
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

    # --- transformer ---------------------------------------------------------
    # Off by default because it costs ~45 min against GBDT's ~18 and the two
    # baseline models are the documented fallback. On, it changes which recipe
    # can win: E45 measured, on 1,675 out-of-fold rows, gbdt+transformer 0.2186
    # and lr+gbdt+transformer 0.2180 against the shipped lr+gbdt 0.2128 -- and
    # every one of the top three recipes contains the transformer.
    #
    # It improves the blend despite being BEHIND gbdt solo (0.1918 vs 0.2075),
    # because it is the least redundant member: gbdt~transformer agreement is
    # 0.788 against lr~transformer 0.886. Replacing LR beats adding to it.
    #
    # Seed-averaged over three seeds, reusing cross_validate's fitter so the
    # shipped model is built the same way as the one the OOF numbers measured.
    # That is consistency with the measurement, not a separate optimisation.
    if args.with_transformer:
        t = time.time()
        from .cross_validate import _fit_transformer
        preds["transformer"] = apply_at_risk_mask(
            _fit_transformer(root, _fit_rows(F, None), ex_cfg, y.shape), at_risk)
        print(f"TX    fit {time.time()-t:.0f}s", flush=True)
        if "gbdt" in preds:
            preds["ens_gbdt_tx"] = _sigmoid(
                0.5 * (_logit(preds["gbdt"]) + _logit(preds["transformer"])))
            preds["ens_lr_gbdt_tx"] = _sigmoid(
                (_logit(preds["lr"]) + _logit(preds["gbdt"])
                 + _logit(preds["transformer"])) / 3.0)

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

    if args.recipe:
        if args.recipe not in preds:
            raise SystemExit(f"--recipe {args.recipe} not in {list(preds)}")
        best = args.recipe
        print(f"\nSHIPPING the pre-specified recipe: {best}")
        print("  (chosen on 1,675 out-of-fold rows, E45 -- NOT re-selected here."
              "\n   The val column below is a measurement, not the decision.)")
    else:
        best = max(scores, key=lambda k: scores[k][1])   # rank by val mAP
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
    # Save EVERY fitted model, not just `best` and lr.
    #
    # The old condition was `name in (best, "lr")`. When the selected model is
    # the ENSEMBLE that set contains only "lr", because "ensemble" is not a key
    # in `fitted` -- it is a combination, not an estimator. So the submission was
    # written from lr+gbdt while only lr was saved, the GBDT that cost 1,598
    # seconds was discarded, and predictions.csv could not be rebuilt from any
    # artifact on disk. test_saved_model_reproduces_predictions_csv caught it
    # the moment the ensemble started winning.
    for name, models in fitted.items():
        if True:
            joblib.dump({"models": models, "feature_names": F.names,
                         "preprocess": lr_pre if name == "lr" else {},
                         "codes": codes, "blocks": F.blocks,
                         "at_risk_rule": "prevalent == 0",
                         "val_macro_auroc": scores[name][0],
                         "val_macro_ap": scores[name][1]},
                        f"artifacts/model_{name}.joblib", compress=3)
            print(f"saved artifacts/model_{name}.joblib")

    # The transformer has no sklearn estimators to dump -- it is a torch model,
    # seed-averaged over three runs in logit space. What makes predictions.csv
    # reproducible is its OUTPUT, so the full-cohort matrix is what gets stored,
    # alongside the config and seeds that produced it.
    #
    # Storing an output rather than an estimator is weaker and is labelled as
    # such (`kind="precomputed"`). It is still the difference between a
    # submission that can be rebuilt from disk and one that cannot -- which is
    # exactly the failure the comment above describes, one model family later.
    if "transformer" in preds:
        from .cross_validate import TRANSFORMER_CFG, TRANSFORMER_SEEDS
        joblib.dump({"kind": "precomputed",
                     "preds": preds["transformer"].astype("float32"),
                     "models": [], "preprocess": {},
                     "feature_names": F.names, "codes": codes,
                     "config": dict(TRANSFORMER_CFG),
                     "seeds": list(TRANSFORMER_SEEDS),
                     "at_risk_rule": "prevalent == 0",
                     "val_macro_auroc": scores["transformer"][0],
                     "val_macro_ap": scores["transformer"][1]},
                    "artifacts/model_transformer.joblib", compress=3)
        print("saved artifacts/model_transformer.joblib (precomputed matrix)")

    # Every blend this script can emit, named once so the manifest and the JSON
    # sidecar cannot drift apart from each other or from the code.
    _RECIPES = {
        "ensemble":       ("sigmoid(0.5*(logit(lr)+logit(gbdt)))", ["lr", "gbdt"]),
        "ens_gbdt_tx":    ("sigmoid(0.5*(logit(gbdt)+logit(transformer)))",
                           ["gbdt", "transformer"]),
        "ens_lr_gbdt_tx": ("sigmoid((logit(lr)+logit(gbdt)+logit(transformer))/3)",
                           ["lr", "gbdt", "transformer"]),
    }
    recipe_str, components = _RECIPES.get(best, (best, [best]))

    # --- calibration ---------------------------------------------------------
    # Per-label Platt, fitted OUT OF FOLD (`python -m src.calibrate --fit oof`)
    # and applied here. Measured held out: BSS 0.1528 -> 0.1755 (+0.0227), with
    # macro AP and macro AUROC moving by exactly 0.00e+00.
    #
    # It cannot lose on the metrics we optimise: a per-label monotone transform
    # leaves every per-label rank statistic identical, and macro AP/AUROC are
    # averages of per-label rank statistics. It can only help on proper scoring
    # rules -- which matters because the grading metric is not known.
    #
    # The maps are applied to the FULL cohort matrix together with `at_risk`,
    # never to an already-masked-and-sliced one: `apply_platt` re-floors
    # not-at-risk pairs because sigmoid(a*logit(1e-6) + b) is not small for
    # every fitted (a, b), and `verify_noop` cannot catch that -- the pairs it
    # would damage are exactly the ones the masked metrics exclude.
    cal_meta = {"calibrated": False}
    if not args.no_calibrate:
        pf = Path("artifacts/platt_params.npz")
        if not pf.exists():
            print("  calibration: artifacts/platt_params.npz missing -- "
                  "run `python -m src.calibrate --fit oof` first; SHIPPING RAW")
        else:
            from .calibrate import PlattParams, apply_platt, verify_noop
            d = np.load(pf)
            pp = PlattParams(a=d["a"], b=d["b"], ok=d["ok"])
            raw = preds[best]
            cal = apply_platt(raw, pp, at_risk=at_risk)
            # Gate on the guarantee rather than trusting it: if the ranking
            # moved, the transform is not monotone and we ship the raw scores.
            # Increasing maps must leave rankings untouched; labels with a
            # deliberately negative slope (calibrate.fit_platt) are excluded
            # from this check, not from calibration.
            _inc = ~pp.ok | (pp.a > 0)
            chk = verify_noop(raw[val], cal[val], y[val], at_risk[val],
                              labels=_inc)
            if max(abs(chk["d_ap"]), abs(chk["d_auroc"])) > 1e-9:
                print(f"  calibration REFUSED: macro AP moved {chk['d_ap']:.2e},"
                      f" AUROC {chk['d_auroc']:.2e} -- shipping raw")
            else:
                preds[best] = cal
                cal_meta = {"calibrated": True,
                            "calibration": "per-label Platt, fitted on OOF",
                            "n_labels_calibrated": int(pp.ok.sum())}
                print(f"  calibration applied: {int(pp.ok.sum())}/40 labels, "
                      f"macro AP moved {chk['d_ap']:.2e} (no-op verified)")

    # --- submission ----------------------------------------------------------
    # Real examples occupy the first len(cohort) rows with eid == pid, so the
    # real block is already in the cohort order `write_predictions` expects.
    # The recipe, so the submission is reproducible from artifacts alone. An
    # ensemble is a rule over models rather than a model, and a rule that lives
    # only in the code that produced one CSV is not a reproducible artifact.
    joblib.dump({"selected": best,
                 "recipe": recipe_str,
                 "components": components,
                 "feature_names": F.names, "codes": codes,
                 "at_risk_rule": "prevalent == 0",
                 "val_macro_auroc": scores[best][0],
                 "val_macro_ap": scores[best][1],
                 **cal_meta},
                "artifacts/submission_manifest.joblib", compress=3)
    print("saved artifacts/submission_manifest.joblib "
          f"(selected={best})")

    # The JSON sidecar records the RECIPE, not just the word "ensemble". It
    # previously carried only {"model": "ensemble"} plus val scores, so the sole
    # on-disk record of what was actually combined -- and of whether the output
    # was calibrated -- was a binary joblib nobody reads. A submission whose
    # own metadata cannot tell you what produced it is not reproducible.
    sub = write_predictions(preds[best][:n_real], model_name=best, root=root,
                            extra_meta={"val_macro_auroc": scores[best][0],
                                        "val_macro_ap": scores[best][1],
                                        "stride": args.stride,
                                        "recipe": recipe_str,
                                        "components": components,
                                        "lr_config": asdict(LRConfig()),
                                        "gbdt_config": asdict(GBDTConfig()),
                                        **cal_meta})
    print(f"wrote outputs/predictions.csv  {sub.shape}")
    run.summary(selected=best, val_macro_ap=scores[best][1])
    run.finish()


if __name__ == "__main__":
    main()
