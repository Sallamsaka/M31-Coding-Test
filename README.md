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

## What the label actually estimates, and what it does not

**The estimand.** For `Y = 1{condition j first diagnosed within 5 years of the
anchor}`, the quantity being modelled is

```
E[Y | x] = P(T <= 5y, cause = j | x) = CIF_j(5y | x)
```

the **cause-specific cumulative incidence function** — the *subdistribution*
(Fine–Gray) estimand, not a cause-specific hazard and not a survivor-conditional
risk.

This matters because **155 of the 365 validation patients (42.5%) die inside the
outcome window**, and a patient who dies at year 2 cannot be diagnosed in years
3–5. Contributing `Y=0` for that patient is the **correct** label for this
estimand, not an unhandled competing risk: the subdistribution keeps decedents
in the risk set by construction. With complete follow-up the IPCW weights are 1,
which is why no weighting machinery appears in the code. Stated explicitly
because the binary label looks naive and is not. (Separately, 42.9% of all
train/val patients carry a death date at *any* horizon — a different quantity,
and the two are easy to conflate.)

The practical consequence is narrow. Competing risks are a **calibration**
concern rather than a discrimination one — published cause-specific vs
subdistribution comparisons differ by ~0.001–0.004 in C-statistic — so the
ranking metrics reported here would be substantially unchanged either way.

Results are reported split by died-in-window vs survived. **That split is a
subgroup-performance check, not a test of whether the model is "really
predicting death"** — asking whether AUROC is equal *within* each death stratum
is a different question from whether the anchor rule selects on the future, and
only the first is answered here. What it showed unprompted is worth more than
what it was asked: logistic regression is flat across the two strata while both
transformers score relatively *better* on decedents (0.6260 → 0.6607), which is
a plausible mechanism for the LR-vs-transformer rank decorrelation (Spearman
0.436) reported below.

## Scope — what this is not evidence about

This measures **pipeline quality on Synthea-generated patients, not clinical
performance on humans.** Synthea's conditions are emitted by rule-based
generative modules, which is the mechanism behind the φ = 1.000 label pair in
this dataset: two conditions are perfectly collinear because a module emits them
together, not because of biology. A model that scores well here has learned a
generator, and the generator is the thing it has learned.

Two further limits, stated rather than implied:

- **The anchor rule selects on the future.** Anchors are derived for train and
  validation and *provided* for test, and the derivation uses the record, so the
  population being predicted is partly defined by what happens after the
  prediction point.
- **The task is data-limited, and the highest-value intervention is out of
  scope.** The logistic-regression learning curve is still rising at n = 2,791,
  which by Rules-of-ML #41 is the point at which more data beats a better model.
  The marginal Synthea patient costs CPU rather than IRB approval, so "generate
  more patients" dominates every modelling change measured here — and it is
  outside the brief. Every architectural result below should be read against
  that: they are refinements on an axis our own measurement says is not the
  binding one.

No claim here should be read as evidence for clinical use, and this dataset
should not be used to make one.

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

`pytest tests/` covers 80 checks including a grep test that nothing outside
`timeutil` calls `pd.to_datetime`, and one that nothing outside `cohort` and
`sequences` reads the ordering timestamp. The behavioural ones assert effects
rather than flags -- that the fully time-ablated model is bit-identical when
every gap is doubled, for instance, with the positive control asserted beside it
so the probe cannot be vacuous.

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
