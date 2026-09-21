# Patient Timeline Forecasting — M31 Research Intern Take-Home

Predict which of 40 target conditions are **newly** diagnosed in the five years
after each patient's anchor date, using only events recorded before it.
3,514 Synthea patients, split 2,791 train / 365 validation / 358 test.

## Quick start

```powershell
pip install -r requirements.txt
.\run_all.ps1 -Smoke        # every code path, ~3 min
.\run_all.ps1 -Stages 1,2,3 # tests, diagnostics, baseline -> predictions.csv (~20 min)
.\run_all.ps1               # everything including the transformer arms (~7 h, CPU)
```

`outputs/predictions.csv` is written by stage 3, so a valid submission exists
before anything slow runs.


## What is where

| path | role |
|---|---|
| **entry points** | |
| `run_all.ps1` | reproduce everything, in dependency order |
| `src/run_baseline.py` | features → prevalence / LR / GBDT / ensemble → `predictions.csv` |
| `src/train_finetune.py` | transformer training, length-bucketed, four arms |
| `src/compare.py` | final head-to-head + the pre-registered selection rule |
| `src/diagnostics.py` | censoring, anchor selection, the selection ledger |
| `src/figures.py`, `src/make_report.py` | report figures and print-ready HTML |
| `src/hf_upload.py` | model card + weights to the Hub |
| **data** | |
| `src/utils/timeutil.py` | the **only** module allowed to parse a timestamp |
| `src/data/cohort.py` | the only module that knows file paths; builds the cohort and the unified event frame |
| `src/data/examples.py` | one row per `(patient, cutoff)`; cutoff augmentation, off by default |
| `src/data/labels.py` | incident vs prevalent, with frozen golden counts |
| `src/data/features.py` | 3,320-column matrix, every statistic fitted on train only |
| `src/data/sequences.py` | tokenizer and vocabulary for the transformer |
| **model and evaluation** | |
| `src/models/gpt.py` | ~2M-parameter transformer, continuous time, Δt attention bias |
| `src/train_baseline.py` | prevalence / logistic regression / gradient boosting |
| `src/evaluate.py` | macro AUROC/AP, bootstrap and paired-bootstrap CIs |
| `src/predict.py` | writes `predictions.csv`, asserts the submission contract |
| `src/utils/io.py` | caching keyed on the full config plus file fingerprints |
| `src/utils/wandb_shim.py` | tracking that cannot take a run down with it |

Experiment tracking is **opt-in**: set `$env:WANDB=1`. Everything is written to
`outputs/metrics.jsonl` either way, and that file is what the report is built
from. The reason is not preference — wandb's background service died mid-run
and took a 40-minute fit with it, from an atexit handler no wrapper can catch.

## The three conventions everything depends on

```python
FEATURE_WINDOW     = "ts < anchor"                      # right-OPEN
LABEL_WINDOW       = "anchor <= first_dx <= anchor+5y"  # both edges CLOSED
first_dx           = min over the patient's WHOLE record, never the window
```

Each was settled by measurement, not taste:

- **Feature edge exclusive at midnight.** Zero of 358 test patients have an
  event at or after their given anchor, while 11 have events in the preceding
  24 hours. The organisers cut at midnight.
- **Label right edge closed.** 119 first-diagnoses land exactly on
  `anchor + 5y` and none beyond. An exclusive edge silently discards 2% of all
  positives.
- **First diagnosis over the whole record.** Synthea re-records recurring
  conditions. Taking the minimum *inside* the window turns every recurrence
  into a false positive and inflates training positives by 23.4%
  (1,220 pairs across 1,065 patients).

## Leakage

Refused at load time, not merely asserted about in a test:

- `DEATHDATE` — populated for 1,354 of 3,156 train/val patients and **zero** of
  358 test patients. A model using it looks excellent in training and
  contributes nothing at test.
- `HEALTHCARE_EXPENSES` / `HEALTHCARE_COVERAGE` — *lifetime* totals present in
  both splits, so they encode post-anchor spend and would silently *raise* the
  test score. The per-encounter cost columns are a different object and are
  used.
- `STOP`-derived durations — 6,078 pre-anchor training medications have a
  `STOP` after the anchor, and the organisers blanked those in test (4.41% null
  vs 1.17% in train), so any duration feature both leaks and shifts.

`pytest tests/` covers 63 checks including a grep test that nothing outside
`timeutil` calls `pd.to_datetime`, and one that nothing outside `cohort` and
`sequences` reads the ordering timestamp.

## Results

See `outputs/per_code_val.csv` and the report. Headline numbers are on the
365-patient validation set, with the caveat that its resolution limit is
roughly ±0.01 macro AUROC — differences below that are not measurable here.

## Reproducibility

`requirements.txt` is pinned. Every cache key is a hash of the full config
dataclass plus the input files' size and mtime, so changing a setting
invalidates exactly what it should. Experiment tracking runs through
`src/utils/wandb_shim.py`, which falls back to offline mode with no API key and
to a local `outputs/metrics.jsonl` if the library is missing — the JSONL is the
source of truth and wandb mirrors it, not the other way round.
