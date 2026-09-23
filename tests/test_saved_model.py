"""The saved model must reproduce the submission, cold.

A serialised bag of estimators is only a model if someone else can apply it.
The failure this guards against is specific and easy to ship: the LR
coefficients are fitted on log1p'd, median-imputed, max-abs-scaled features,
and all three of those transforms were originally built inside a closure and
never saved. Loading the estimators alone and calling `predict_proba` would
have run them against differently-scaled inputs and produced plausible,
wrong numbers -- no exception, no warning.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts" / "model_lr.joblib"
PRED = ROOT / "outputs" / "predictions.csv"


@pytest.mark.slow
def test_saved_model_reproduces_predictions_csv():
    joblib = pytest.importorskip("joblib")
    if not (MODEL.exists() and PRED.exists()):
        pytest.skip("run `python -m src.run_baseline` first")

    from src.data.cohort import load_cohort, load_target_codes
    from src.data.features import build_features
    from src.data.labels import at_risk_mask, load_labels

    bundle = joblib.load(MODEL)
    pre = bundle["preprocess"]
    assert pre, "no preprocessing saved -- the coefficients are unusable"
    assert "scale_" in pre and "log1p" in pre

    F = build_features(ROOT)
    assert F.names == bundle["feature_names"], "feature layout changed"

    def _score(bundle_, preprocess):
        """Re-run one saved bundle over the current feature matrix."""
        X = F.X.astype(np.float64)
        if "impute" in preprocess:
            X = np.where(np.isnan(X), preprocess["impute"], X)
        if preprocess.get("log1p"):
            X = np.log1p(np.clip(X, 0, None))
        if "scale_" in preprocess:
            X = (X / preprocess["scale_"]).astype(np.float32)
        out = np.zeros((len(X), len(bundle_["codes"])), np.float32)
        for j, entry in enumerate(bundle_["models"]):
            if entry["kind"] == "constant":
                out[:, j] = entry["value"]
            else:
                m = entry["estimator"]
                out[:, j] = m.predict_proba(X)[:, list(m.classes_).index(1)]
        return out

    # Reproduce whatever was SHIPPED, which is not necessarily lr.
    #
    # This assumed the submission came from model_lr.joblib. It does not when
    # the ensemble wins selection: predictions.csv was then written from
    # lr+gbdt while only lr was saved, so the submission could not be rebuilt
    # from anything on disk and the GBDT that cost 1,598 s was discarded. The
    # test caught that correctly -- it was the pipeline that was wrong.
    man_path = ROOT / "artifacts" / "submission_manifest.joblib"
    manifest = joblib.load(man_path) if man_path.exists() else {"selected": "lr",
                                                               "components": ["lr"]}

    def _logit(q):
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return np.log(q / (1 - q))

    parts = []
    for comp in manifest.get("components", ["lr"]):
        bp = ROOT / "artifacts" / f"model_{comp}.joblib"
        assert bp.exists(), (
            f"the submission needs component {comp!r} but {bp.name} was never "
            "saved -- the shipped predictions cannot be reproduced")
        b = joblib.load(bp)
        if b.get("kind") == "precomputed":
            # The transformer is a torch model, seed-averaged over three runs;
            # there is no sklearn estimator to re-score. Its saved artifact is
            # the full-cohort output matrix. Weaker than an estimator and
            # labelled as such -- but it is what makes the submission
            # rebuildable, which is what this test is actually about.
            parts.append(np.asarray(b["preds"], dtype=np.float64))
        else:
            parts.append(_score(b, b.get("preprocess") or {}))

    if len(parts) == 1:
        P = parts[0]
    else:
        # The recipe recorded alongside the components.
        P = 1.0 / (1.0 + np.exp(-sum(_logit(q) for q in parts) / len(parts)))

    lab = load_labels(ROOT)
    ar = at_risk_mask(lab)

    # Calibration is part of the shipped pipeline, so it is part of what has to
    # be reproduced. Without this the reconstruction rebuilds the RAW blend and
    # disagrees with a calibrated predictions.csv -- which would look like a
    # broken artifact rather than a missing step.
    if manifest.get("calibrated"):
        from src.calibrate import PlattParams, apply_platt
        pf = ROOT / "artifacts" / "platt_params.npz"
        assert pf.exists(), (
            "the submission is marked calibrated but artifacts/platt_params.npz "
            "is missing -- the shipped predictions cannot be reproduced")
        d = np.load(pf)
        P = apply_platt(P, PlattParams(a=d["a"], b=d["b"], ok=d["ok"]), at_risk=ar)

    P = np.where(ar, P, 1e-6)
    P = np.clip(P, 1e-6, 1 - 1e-6)

    cohort = load_cohort(ROOT)
    is_test = (cohort.split == "test").to_numpy()
    codes = load_target_codes(ROOT)
    got = pd.DataFrame(P[: len(cohort)][is_test], columns=codes)
    got.insert(0, "patient_id", cohort.loc[is_test, "patient_id"].to_numpy())
    anchors = pd.read_csv(ROOT / "test_anchors.csv", dtype=str)
    got = (got.set_index("patient_id").reindex(anchors["Id"].to_numpy())
              .reset_index(drop=True))

    want = pd.read_csv(PRED, dtype={"patient_id": str}).drop(columns="patient_id")
    delta = np.abs(got.to_numpy(float) - want.to_numpy(float)).max()
    assert delta < 1e-6, (
        f"reloaded model disagrees with predictions.csv by {delta:.2e} -- the "
        "saved artifact does not reproduce the submission")
