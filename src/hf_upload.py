"""Push the selected model and its card to the Hugging Face Hub.

Run: ``python -m src.hf_upload --repo <user>/<name>``

Needs ``HF_TOKEN`` in the environment, or a prior ``huggingface-cli login``.
Without one this prints what it *would* upload and exits 0, so the pipeline
script never fails on a missing credential -- the upload is a delivery step,
not a result, and a missing token should not invalidate a completed run.

Nothing derived from validation or test labels is uploaded. The card reports
5-fold cross-validation results on the training patients (the test outcomes
are withheld).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

__all__ = ["build_card", "upload"]

CARD = """---
license: mit
tags:
  - tabular-classification
  - healthcare
  - synthetic-data
  - ehr
library_name: pytorch
---

# Patient Timeline Forecasting — {model}

Predicts which of 40 target conditions are **newly** diagnosed in the five
years after a patient's anchor date, from structured Synthea EHR events
recorded strictly before it.

Trained for the M31 research-intern take-home, on synthetic data only.

## Model

`sigmoid((logit(LR) + logit(GBDT) + logit(transformer)) / 3)`, per-label Platt
calibrated (fitted out-of-fold). The transformer is 1 layer / width 64 / 2 heads,
reads one token per event with a Time2Vec encoding of time before the anchor **and
of the patient's age at that event**, and fuses the 3,320 tabular features at the
readout; 3 seeds averaged in logit space. Files: `model_lr.joblib`,
`model_gbdt.joblib`, `model_transformer.joblib` (the transformer's output matrix),
`transformer_seed300.pt` / `transformer_seed301.pt` / `transformer_seed302.pt`
(its weights, one per seed), `platt_params.npz` (the 40 calibration maps),
`submission_manifest.joblib` (how the three models are combined), `vocab.json`
(the transformer's vocabulary).

## Task

- 3,514 Synthea patients, split 2,791 train / 365 validation / 358 test.
- Anchor = last recorded encounter minus five calendar years, floored to midnight.
- Label = **first-ever** diagnosis of the condition falls in `[anchor, anchor+5y]`.
  A patient already diagnosed before the anchor is *prevalent*: structurally
  negative, and reported separately rather than counted as an ordinary negative.

## Results

The test set's outcomes are withheld, so performance is estimated by 5-fold
cross-validation over the 2,791 training patients: each fold is predicted by
models trained on the other four (the transformer uses the provided validation
set only to choose its stopping epoch). Mean ± SD over the 5 test folds:

| model | macro AUROC | macro AP |
|---|---|---|
| logistic regression | 0.7595 ± 0.0056 | 0.2077 ± 0.0192 |
| gradient-boosted trees | 0.7640 ± 0.0081 | 0.2260 ± 0.0185 |
| transformer (3 seeds) | 0.7661 ± 0.0065 | 0.2203 ± 0.0157 |
| **submitted: blend of the three, calibrated** | **0.7755 ± 0.0028** | **0.2435 ± 0.0220** |

(Calibration in the last row is cross-fitted: fitted on four folds, scored on the fifth.)

## What the model is actually learning

The anchor is defined from each patient's last encounter, so for the **42.9%**
of training patients with a death date, the outcome window is exactly their
last five years of life — 100% of those deaths fall within 30 days of the
window end. A substantial part of the achievable signal is therefore
"is this record about to end", which is a property of how the task was
constructed rather than of clinical prediction. The same rule generated the
test anchors, so this is not leakage, but it does bound how the results should
be interpreted.

## Leakage controls

`DEATHDATE`, `HEALTHCARE_EXPENSES` and `HEALTHCARE_COVERAGE` are refused at
load time. `STOP`-derived durations are excluded: the organisers blanked
post-anchor stops in the test split, so such a feature would both leak and
shift. Every fitted statistic — vocabulary, quantile edges, scalers — is fitted
on training patients only. 145 automated checks cover this, including a grep
test that no module outside the time utility parses a timestamp.

## Reproducing the predictions

The files on this page are enough to rebuild the submitted `predictions.csv`
exactly, without any training. The patient data is not included here: you need
the dataset provided with the M31 take-home.

**You need** Python 3.13, git, and that dataset.

1. Get the code and install its dependencies:

   ```
   git clone https://github.com/Sallamsaka/M31-Coding-Test
   cd M31-Coding-Test
   pip install -r requirements.txt
   ```

2. Copy the provided data into that folder, so that it looks like this:

   ```
   M31-Coding-Test/
     train_val/              provided CSV tables, train and validation patients
     test/                   provided CSV tables, test patients
     patient_splits.csv
     target_conditions.csv
     test_anchors.csv
     src/, outputs/, ...     already there from the clone
   ```

3. Run:

   ```
   python -m src.reproduce --from-hub sallamsaka/M31-Coding-Test
   ```

   This downloads the model files from this page, builds each test patient's
   features and event sequence, runs logistic regression, the boosted trees and
   the three transformer seeds, averages and calibrates them, and writes
   `outputs/predictions_reproduced.csv` (358 patients, 40 conditions).

4. Check the last line it prints. The clone already contains the submitted
   `outputs/predictions.csv`, and the script compares against it:

   ```
   max |reproduced - outputs/predictions.csv| = 7.40e-08  (MATCH)
   ```

   Anything below 1e-5 prints `MATCH`; the remaining difference is floating-point
   rounding.

Retraining everything from scratch is described in the GitHub README.
"""


def build_card(model: str, metrics: dict) -> str:
    if metrics:
        rows = "\n".join(f"| {k} | {v:.4f}" + " |" if isinstance(v, float)
                         else f"| {k} | {v} |" for k, v in metrics.items())
        table = "| metric | value |\n|---|---|\n" + rows
    else:
        table = "_not yet computed_"
    return CARD.format(model=model, metrics=table)


def _cached_token() -> str | None:
    """The token from a prior `huggingface-cli login`, which the docstring
    always promised to honour but the code never read."""
    try:
        from huggingface_hub import get_token        # noqa: PLC0415
        return get_token()
    except Exception:
        return None


def _shipped_transformer_ckpts(root: Path) -> list[Path]:
    """The three seed checkpoints behind model_transformer.joblib.

    model_transformer.joblib stores the transformer's OUTPUT matrix, not its
    weights, so the weights must be shipped separately. They are identified by
    recomputing the config fingerprint the shipped fit used (TRANSFORMER_CFG,
    full train split, default ExampleConfig) -- not by modification time and
    not by a hardcoded hash, either of which silently goes stale on a refit.
    """
    import hashlib                                   # noqa: PLC0415

    import torch                                     # noqa: PLC0415

    from .cross_validate import TRANSFORMER_CFG, TRANSFORMER_SEEDS
    from .data.cohort import load_cohort
    from .data.sequences import SeqConfig
    from .train_finetune import TrainConfig, _config_fingerprint

    c = load_cohort(root)
    tr = sorted(set(c.loc[c.split == "train", "pid"].tolist()))
    fk = hashlib.sha256(",".join(map(str, tr)).encode()).hexdigest()[:12]
    out = []
    for sd in TRANSFORMER_SEEDS:
        cfg = TrainConfig(seed=sd, dev_frac=0.0, holdout_frac=0.0, epochs=30,
                          patience=4, min_delta=0.002, predict_all=True,
                          **TRANSFORMER_CFG)
        hits = []
        for p in sorted((root / "artifacts").glob(f"model_P4_seed{sd}_*.pt")):
            v = torch.load(p, map_location="cpu", weights_only=False)["vocab_size"]
            fp = _config_fingerprint(cfg, "P4", v, SeqConfig().block_size, fk, "")
            if p.stem.endswith(fp[:8]):
                hits.append(p)
        if len(hits) != 1:
            raise FileNotFoundError(
                f"expected one shipped checkpoint for seed {sd}, found {hits}")
        out += hits
    return out


def upload(repo: str, root: Path | str = ".", dry_run: bool = False) -> None:
    root = Path(root)
    meta_path = root / "outputs" / "predictions_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    model = meta.get("model", "baseline")
    metrics = {k: v for k, v in meta.items()
               if k.startswith("val_") and isinstance(v, (int, float))}

    card = root / "outputs" / "README_hf.md"
    card.parent.mkdir(parents=True, exist_ok=True)
    card.write_text(build_card(model, metrics), encoding="utf-8")

    files = [card]
    for rel in ("artifacts/vocab.json", "outputs/predictions.csv",
                "outputs/per_code_val.csv", "outputs/predictions_meta.json",
                "artifacts/submission_manifest.joblib",
                "artifacts/platt_params.npz"):
        p = root / rel
        if p.exists():
            files.append(p)
    # The model itself: exactly what the submission manifest names, and
    # nothing else. This used to glob `model_*.pt`, which swept up ~90
    # experiment checkpoints from every CV fold and ablation, while
    # platt_params.npz -- without which the calibrated submission cannot be
    # rebuilt -- was not uploaded at all.
    components = meta.get("components", ["lr"])
    files += [root / "artifacts" / f"model_{c}.joblib" for c in components
              if (root / "artifacts" / f"model_{c}.joblib").exists()]
    if "transformer" in components:
        files += _shipped_transformer_ckpts(root)

    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
             or _cached_token())
    if dry_run or not token:
        why = "dry run" if dry_run else "no HF_TOKEN in environment"
        print(f"[hf_upload] {why}; would upload {len(files)} file(s) to {repo}:")
        for p in files:
            print(f"  {p.relative_to(root)}  ({p.stat().st_size:,} bytes)")
        print(f"[hf_upload] model card written to {card.relative_to(root)}")
        return

    from huggingface_hub import HfApi           # noqa: PLC0415
    api = HfApi(token=token)
    api.create_repo(repo, exist_ok=True, repo_type="model")
    uploaded = set()
    for p in files:
        name = "README.md" if p == card else p.name
        if name.startswith("model_P4_seed") and name.endswith(".pt"):
            # Readable names on the Hub; the fingerprint is listed in the card.
            name = f"transformer_seed{name.split('_seed')[1].split('_')[0]}.pt"
        api.upload_file(path_or_fileobj=str(p), path_in_repo=name,
                        repo_id=repo, repo_type="model")
        uploaded.add(name)
        print(f"  uploaded {name}")
    # upload_file never deletes. A retrained transformer has new fingerprints,
    # so the previous seed checkpoints would sit beside the new ones and a
    # reviewer could not tell which is the submitted model. Remove ONLY stale
    # transformer checkpoints -- never anything else in the repository.
    for name in api.list_repo_files(repo, repo_type="model"):
        stale_ckpt = name.endswith(".pt") and name.startswith(("model_P4_seed", "transformer_seed"))
        if stale_ckpt and name not in uploaded:
            api.delete_file(name, repo_id=repo, repo_type="model",
                            commit_message=f"remove superseded checkpoint {name}")
            print(f"  deleted stale {name}")
    print(f"[hf_upload] https://huggingface.co/{repo}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. yourname/m31-timeline")
    ap.add_argument("--root", default=".")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    upload(args.repo, args.root, args.dry_run)


if __name__ == "__main__":
    main()
