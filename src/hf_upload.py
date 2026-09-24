"""Push the selected model and its card to the Hugging Face Hub.

Run: ``python -m src.hf_upload --repo <user>/<name>``

Needs ``HF_TOKEN`` in the environment, or a prior ``huggingface-cli login``.
Without one this prints what it *would* upload and exits 0, so the pipeline
script never fails on a missing credential -- the upload is a delivery step,
not a result, and a missing token should not invalidate a completed run.

Nothing derived from validation or test labels is uploaded. The card reports
validation metrics because they are model-selection output, which the brief
permits; it does not ship the validation predictions themselves.
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

Trained for the M31 research-intern take-home. **Synthetic data only. Not a
clinical device, and not usable for any decision about a real person.**

## Model

`sigmoid((logit(LR) + logit(GBDT) + logit(transformer)) / 3)`, per-label Platt
calibrated (fitted out-of-fold). The transformer is 1 layer / width 64 / 2 heads,
reads one token per event with a Time2Vec encoding of time before the anchor **and
of the patient's age at that event**, and fuses the 3,320 tabular features at the
readout; 3 seeds averaged in logit space. Files: `model_lr.joblib`,
`model_gbdt.joblib`, `model_transformer.joblib` (the transformer's output matrix),
`model_P4_seed300_*.pt` / `seed301` / `seed302` (its weights), `platt_params.npz`,
`submission_manifest.joblib`.

## Task

- 3,514 Synthea patients, split 2,791 train / 365 validation / 358 test.
- Anchor = last recorded encounter minus five calendar years, floored to midnight.
- Label = **first-ever** diagnosis of the condition falls in `[anchor, anchor+5y]`.
  A patient already diagnosed before the anchor is *prevalent*: structurally
  negative, and reported separately rather than counted as an ordinary negative.

## Results (validation, n=365)

{metrics}

The validation set resolves differences of roughly **±0.01 macro AUROC** and no
better. Per-condition AUROC on the rarest label (5 positives) carries a
Hanley–McNeil 95% interval of about ±0.24. Numbers below that separation are
reported but should not be read as rankings.

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
on training patients only. 150 automated checks cover this, including a grep
test that no module outside the time utility parses a timestamp.

## Limitations

Synthea is not real EHR data. A discriminator separates it from MIMIC at
AUC 0.999, and across 19 health datasets the winning classifier agreed between
real- and synthetic-trained models in only 21–26% of cases. **Whichever model
wins here, that ranking should not be assumed to transfer.**

## Reproducing

```bash
git clone https://github.com/Sallamsaka/M31-Coding-Test && pip install -r requirements.txt
./run_all.ps1
```
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
        api.upload_file(path_or_fileobj=str(p), path_in_repo=name,
                        repo_id=repo, repo_type="model")
        uploaded.add(name)
        print(f"  uploaded {name}")
    # upload_file never deletes. A retrained transformer has new fingerprints,
    # so the previous seed checkpoints would sit beside the new ones and a
    # reviewer could not tell which is the submitted model. Remove ONLY stale
    # transformer checkpoints -- never anything else in the repository.
    for name in api.list_repo_files(repo, repo_type="model"):
        if name.startswith("model_P4_seed") and name.endswith(".pt") and name not in uploaded:
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
