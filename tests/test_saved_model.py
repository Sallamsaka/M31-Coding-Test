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

    X = F.X.astype(np.float64)
    if "impute" in pre:
        X = np.where(np.isnan(X), pre["impute"], X)
    if pre["log1p"]:
        X = np.log1p(np.clip(X, 0, None))
    X = (X / pre["scale_"]).astype(np.float32)      # MaxAbsScaler.transform

    P = np.zeros((len(X), len(bundle["codes"])), np.float32)
    for j, entry in enumerate(bundle["models"]):
        if entry["kind"] == "constant":
            P[:, j] = entry["value"]
        else:
            m = entry["estimator"]
            P[:, j] = m.predict_proba(X)[:, list(m.classes_).index(1)]

    lab = load_labels(ROOT)
    ar = at_risk_mask(lab)
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
