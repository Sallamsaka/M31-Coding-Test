"""Rebuild predictions.csv from the published model files, without training.

    python -m src.reproduce --from-hub sallamsaka/M31-Coding-Test
    python -m src.reproduce                      # the same, from local artifacts/

Needs the provided data in the repo root (as for training). Steps:

1. build the 3,320 tabular features and the event sequences from the data;
2. score the 40 logistic regressions and the 40 boosted-tree models from their
   saved bundles (with the preprocessing stored alongside them);
3. run the three transformer seeds from their saved WEIGHTS and average their
   logits;
4. blend the three models (mean of logits), apply the 40 per-condition
   calibration maps, and set already-diagnosed pairs to ~0;
5. write the test patients' probabilities and compare them with
   outputs/predictions.csv if it exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .calibrate import PlattParams, apply_platt
from .data.cohort import load_cohort, load_target_codes
from .data.features import build_features
from .data.labels import at_risk_mask, load_labels
from .data.sequences import SeqConfig, Vocab, build_sequences
from .models.gpt import GPTConfig, PatientTransformer
from .train_finetune import predict

HUB_FILES = ["model_lr.joblib", "model_gbdt.joblib", "platt_params.npz",
             "submission_manifest.joblib", "vocab.json",
             "transformer_seed300.pt", "transformer_seed301.pt", "transformer_seed302.pt"]


def _logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def _files(from_hub: str | None, root: Path) -> dict[str, Path]:
    if from_hub:
        from huggingface_hub import hf_hub_download   # noqa: PLC0415
        return {f: Path(hf_hub_download(repo_id=from_hub, filename=f)) for f in HUB_FILES}
    from .hf_upload import _shipped_transformer_ckpts  # noqa: PLC0415
    art = root / "artifacts"
    out = {f: art / f for f in HUB_FILES[:5]}
    for p in _shipped_transformer_ckpts(root):
        out[f"transformer_seed{torch.load(p, map_location='cpu', weights_only=False)['seed']}.pt"] = p
    return out


def _score_bundle(bundle: dict, X: np.ndarray) -> np.ndarray:
    """40 per-condition sklearn models, with their stored preprocessing."""
    pre = bundle.get("preprocess") or {}
    X = X.astype(np.float64)
    if "impute" in pre:
        X = np.where(np.isnan(X), pre["impute"], X)
    if pre.get("log1p"):
        X = np.log1p(np.clip(X, 0, None))
    if "scale_" in pre:
        X = (X / pre["scale_"]).astype(np.float32)
    out = np.zeros((len(X), len(bundle["codes"])), np.float32)
    for j, entry in enumerate(bundle["models"]):
        if entry["kind"] == "constant":
            out[:, j] = entry["value"]
        else:
            m = entry["estimator"]
            out[:, j] = m.predict_proba(X)[:, list(m.classes_).index(1)]
    return out


def _load_vocab(path: Path) -> Vocab:
    v = json.loads(path.read_text(encoding="utf-8"))
    return Vocab(stoi={t: i for i, t in enumerate(v["itos"])}, itos=v["itos"],
                 edges=v["edges"], fused=set(v["fused"]), meta=v.get("meta", {}))


def reproduce(root: str | Path = ".", from_hub: str | None = None) -> pd.DataFrame:
    root = Path(root)
    files = _files(from_hub, root)
    manifest = joblib.load(files["submission_manifest.joblib"])

    F = build_features(root)
    parts = [_score_bundle(joblib.load(files[f"model_{c}.joblib"]), F.X)
             for c in ("lr", "gbdt")]
    print("  scored LR and GBDT", flush=True)

    # Transformer: same vocabulary, sequences and feature scaling as training.
    vocab = _load_vocab(files["vocab.json"])
    pack = build_sequences(root, vocab, SeqConfig())
    assert (F.eid == pack.eid).all(), "features and sequences disagree on order"
    Xt = np.log1p(np.clip(np.nan_to_num(F.X, nan=0.0), 0, None))
    scale = np.abs(Xt[F.split == "train"]).max(0)          # train-only, as in training
    feats = (Xt / np.where(scale > 0, scale, 1.0)).astype(np.float32)
    rows = np.arange(len(pack.tokens))
    acc = 0.0
    for s in (300, 301, 302):
        ck = torch.load(files[f"transformer_seed{s}.pt"], map_location="cpu",
                        weights_only=False)
        assert ck["vocab_size"] == len(vocab), "vocabulary does not match the weights"
        model = PatientTransformer(GPTConfig(**ck["gpt_config"]))
        model.load_state_dict(ck["state_dict"], strict=True)
        acc = acc + _logit(predict(model, pack.tokens, pack.dt, pack.lengths, rows,
                                   features=feats, age=pack.age_days))
        print(f"  ran transformer seed {s}", flush=True)
    parts.append(1 / (1 + np.exp(-acc / 3)))

    P = 1 / (1 + np.exp(-sum(_logit(q) for q in parts) / len(parts)))
    ar = at_risk_mask(load_labels(root))
    if manifest.get("calibrated"):
        d = np.load(files["platt_params.npz"])
        P = apply_platt(P, PlattParams(a=d["a"], b=d["b"], ok=d["ok"]), at_risk=ar)
    P = np.clip(np.where(ar, P, 1e-6), 1e-6, 1 - 1e-6)

    cohort = load_cohort(root)
    is_test = (cohort.split == "test").to_numpy()
    got = pd.DataFrame(P[: len(cohort)][is_test], columns=load_target_codes(root))
    got.insert(0, "patient_id", cohort.loc[is_test, "patient_id"].to_numpy())
    order = pd.read_csv(root / "test_anchors.csv", dtype=str)["Id"].to_numpy()
    return got.set_index("patient_id").reindex(order).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-hub", default=None, metavar="REPO_ID")
    ap.add_argument("--out", default="outputs/predictions_reproduced.csv")
    a = ap.parse_args()
    got = reproduce(".", a.from_hub)
    got.to_csv(a.out, index=False)
    print(f"wrote {a.out}  {got.shape}")
    ref = Path("outputs/predictions.csv")
    if ref.exists():
        want = pd.read_csv(ref, dtype={"patient_id": str})
        assert list(want.patient_id) == list(got.patient_id), "patient order differs"
        d = np.abs(want.drop(columns="patient_id").to_numpy(float)
                   - got.drop(columns="patient_id").to_numpy(float)).max()
        print(f"max |reproduced - outputs/predictions.csv| = {d:.2e}  "
              f"({'MATCH' if d < 1e-5 else 'MISMATCH'})")


if __name__ == "__main__":
    main()
