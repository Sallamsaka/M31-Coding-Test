# Forecasting new diagnoses from patient timelines (Synthea)

For each patient, predict which of 40 conditions are first diagnosed in the five
years after an anchor date (last encounter minus five years), using only events
before the anchor. 3,514 synthetic patients: 2,791 train / 365 validation / 358 test.

- Report: `report/report.md` (render with `python -m src.make_report`)
- Model: [huggingface.co/sallamsaka/M31-Coding-Test](https://huggingface.co/sallamsaka/M31-Coding-Test)
- Training curves: [wandb project m31-patient-timeline](https://wandb.ai/sallamsaka-university-of-toronto/m31-patient-timeline)
- Submission: `outputs/predictions.csv`

## Setup

Python 3.13, CPU only. Put the provided data folders (`train_val/`, `test/`) and
CSVs (`patient_splits.csv`, `target_conditions.csv`, `test_anchors.csv`) in the
repo root.

```powershell
pip install -r requirements.txt
python -m pytest -q                                   # tests (~5 min)
python -m src.cross_validate --max-folds 3            # out-of-fold predictions
python -m src.calibrate --fit oof --model ens_lr_gbdt_tx --oof-model lr+gbdt+transformer
python -m src.run_baseline --with-transformer --recipe ens_lr_gbdt_tx   # -> predictions.csv
```

`.\run_all.ps1 -Smoke` runs every code path in a few minutes.

## Model

An average of three models' logits, with per-condition calibration:

- logistic regression and gradient-boosted trees on 3,320 count/lab/demographic features;
- a one-layer transformer over the event sequence (one token per event, with the
  time before the anchor and the patient's age at each event encoded), fused with
  the same features.

Model choices were made with 5-fold cross-validation on the training patients;
the validation set was used for early stopping and monitoring.

## Code

| path | what it does |
|---|---|
| `src/data/cohort.py` | load the CSVs, compute anchors, build the event table |
| `src/data/labels.py` | labels: first-ever diagnosis inside the window |
| `src/data/features.py` | tabular features |
| `src/data/sequences.py` | vocabulary and token sequences |
| `src/models/gpt.py` | the transformer |
| `src/train_baseline.py`, `src/train_finetune.py` | training |
| `src/cross_validate.py`, `src/evaluate.py` | cross-validation and paired bootstrap comparisons |
| `src/run_baseline.py`, `src/predict.py` | final fit and `predictions.csv` |
| `src/tx_sweep.py` | the transformer experiments reported in section 2 |
| `src/hf_upload.py`, `src/log_shipped_run.py` | Hugging Face upload; wandb-logged refit |

Experiment tracking is opt-in (`$env:WANDB=1`); every run also logs to
`outputs/metrics.jsonl`.
