# Patient Timeline Forecasting on Synthea

**M31 Research Intern Take-Home**

The task: predict which of 40 conditions are **first diagnosed** in the five years after each
patient's anchor date (last encounter minus five years), using only events before the
anchor. There are 3,514 synthetic patients, split 2,791 train / 365 validation / 358 test.

Code: [github.com/Sallamsaka/M31-Coding-Test](https://github.com/Sallamsaka/M31-Coding-Test) ·
model: [huggingface.co/sallamsaka/M31-Coding-Test](https://huggingface.co/sallamsaka/M31-Coding-Test) ·
training curves: [wandb project](https://wandb.ai/sallamsaka-university-of-toronto/m31-patient-timeline)
(runs [seed 300](https://wandb.ai/sallamsaka-university-of-toronto/m31-patient-timeline/runs/1nyu0iwr),
[seed 301](https://wandb.ai/sallamsaka-university-of-toronto/m31-patient-timeline/runs/s20y3qfg),
[seed 302](https://wandb.ai/sallamsaka-university-of-toronto/m31-patient-timeline/runs/saafguxm))

## Results

The test outcomes are withheld, so **test-set AUROC and mAP cannot be computed here**. They
are estimated by 5-fold cross-validation on the 2,791 training patients: each fold is scored
by models trained on the other four. Mean ± standard deviation across the five folds:

| model | macro AUROC | macro AP (mAP) |
|---|---|---|
| logistic regression (LR) | 0.7595 ± 0.0056 | 0.2077 ± 0.0192 |
| gradient-boosted trees (GBDT) | 0.7640 ± 0.0081 | 0.2260 ± 0.0185 |
| transformer (3 seeds) | 0.7661 ± 0.0065 | 0.2203 ± 0.0157 |
| average of the three, uncalibrated | 0.7747 ± 0.0057 | 0.2421 ± 0.0230 |
| **submitted: average of the three, calibrated** | **0.7755 ± 0.0028** | **0.2435 ± 0.0220** |

The submission averages the three models' logits, then applies a per-condition calibration
(Brier score 0.04147 → 0.04021). On the provided validation set, used only for monitoring,
it scores 0.7742 AUROC and 0.2729 AP.

---

## 1. Representation

**One unit of input is one event:** one row from any of the ten tables, as a token such as
`MED_197361` or `COND_44054006`. Codes that at least 5 training patients have form a
1,105-token vocabulary (the 119 codes dropped are 0.075% of events). Each sequence keeps the
most recent 512 events and ends in an `[ANCHOR]` token, where the prediction is read.

**Lab values are fused into the token.** `OBS_8480-6_Q7` is systolic blood pressure in its
7th decile, with deciles computed on training patients. This covers the 57 common labs;
rarer ones (0.76% of numeric events) are followed by a shared `Q0`–`Q9` token. Fusing makes
sequences 37% shorter. In hindsight, each lab-and-decile token has its own embedding, so
neighbouring deciles of the same lab share nothing; building the embedding as
`embedding(lab) + embedding(decile)` would fix that (not tested).

**Time between events** is given by two learned encodings (Time2Vec) on each event: the time
before the anchor, as `log(1 + days)` because gaps reach 37,284 days, and the patient's age
at that event. They are concatenated with the token embedding and projected to 64
dimensions.

![Head 2's attention to a blood-pressure reading, by its age](../outputs/figures/time_attention.png)

The figure shows one head's attention to the same blood-pressure token depending only on how
long before the anchor it was recorded. A reading 100–300 days old gets up to 1.7× the
attention of a 90-day-old one; readings older than two years get 0.3×. The head has learned
to focus on the last annual check-up.

**Fields used.** Each patient is also summarised as 3,320 tabular features, which LR and GBDT
use directly and the transformer joins to its output.

| source | in the event sequence (transformer) | in the tabular features (all three models) |
|---|---|---|
| the ten event tables | one token per row, from its `CODE` (imaging: `MODALITY_CODE`) | how many times each code appeared in the last year, the last five years, and ever |
| observation `VALUE` | numeric: decile, fused into the token; text: not used | last value, decile counts, trend over time |
| `REASONCODE` (why a drug, procedure or care plan was given) | not used (tested, no gain, §2) | counts per reason |
| per-row costs (`BASE_COST`, `TOTAL_CLAIM_COST`) | not used | cost per visit |
| sex, race, ethnicity, marital status | four tokens at the start of the sequence | one-hot columns |
| birth date | age at each event (time encoding) | age at the anchor |

**Refused.** `DEATHDATE` (filled in train, empty in test), the lifetime
`HEALTHCARE_EXPENSES` and `HEALTHCARE_COVERAGE` totals, and any duration built from `STOP`
(blanked after the anchor in test) are refused at load time. Everything fitted (vocabulary,
deciles, scalers, calibration) uses training patients only, and 145 automated tests check
the pipeline for leakage. Genomics and imaging were not used: the structured tables are
enough, and every result here is limited by the number of patients, which neither changes.

---

## 2. Model and training

| component | setting |
|---|---|
| input | token embedding (64) ‖ Time2Vec(time before anchor) (64) ‖ Time2Vec(age at event) (64) → linear 192→64 |
| trunk | 1 transformer layer, width 64, 2 heads of 32, MLP 64→256→64, pre-LayerNorm, no dropout |
| readout | output at `[ANCHOR]` (64) ‖ tabular features projected 3,320→128 with GELU |
| head | linear 192→40, one logit per condition |
| parameters | 565,628 |

With one layer, the causal mask has no effect: `[ANCHOR]` is the last position and sees
every event. The model is effectively attention pooling over (event, time, age) items.

- **Loss:** binary cross-entropy over the 40 conditions. A condition the patient already had
  before the anchor is left out of that patient's loss.
- **Training:** AdamW, learning rate 1.2e-3, weight decay 0.001, warm-up then cosine decay,
  batch 32, up to 30 epochs with early stopping on validation macro AP (patience 4). Three
  seeds, averaged in logit space.
- **LR:** L2-regularised, C = 0.03, on log(1 + count). **GBDT:** 300 trees, learning rate
  0.05, 15 leaves.

Hyperparameters were chosen on an inner split of the training patients, not on validation.

![Training and validation curves](../outputs/figures/training_curves.png)

Early stopping picked epochs 5, 11 and 9 for the three seeds. Validation loss bottoms out
around epochs 5–7 while training loss keeps falling: the model overfits quickly, as expected
with 2,791 patients.

### Experiments

Each fix to a known weakness was tested as one change against the same baseline, on all
2,791 training patients out of fold. The intervals are a paired bootstrap: resample the
patients 300 times, recompute the difference between the two models each time, and keep the
middle 95%. The rule for adopting a change was written down before the sweep ran.

| change vs baseline (transformer alone, 1 seed) | Δ macro AP [95% CI] | Δ macro AUROC [95% CI] | result |
|---|---|---|---|
| **+ age at each event** | **+0.0100 [+0.0023, +0.0177]** | **+0.0098 [+0.0035, +0.0159]** | **adopted** |
| 4 heads of 16 instead of 2 of 32 | −0.0014 [−0.0071, +0.0049] | −0.0017 [−0.0057, +0.0028] | no |
| + embedding of the reason a drug was given | −0.0028 [−0.0105, +0.0029] | −0.0037 [−0.0086, +0.0016] | no |
| every lab fused into its token | +0.0023 [−0.0047, +0.0088] | −0.0032 [−0.0089, +0.0023] | no |
| + text answers (smoking, urinalysis) | +0.0006 [−0.0067, +0.0075] | −0.0036 [−0.0089, +0.0020] | no |
| 2 layers, on top of age (3 seeds) | −0.0029 [−0.0076, +0.0019] | −0.0013 [−0.0048, +0.0024] | no |

Age was confirmed on two new seeds: over three seeds it improves the transformer by +0.0143
AP [+0.0056, +0.0210] and +0.0114 AUROC [+0.0061, +0.0166] (500 resamples), and it helped at
every seed.

Earlier experiments, mostly scored on the validation set, where differences under about
0.01 cannot be told apart:

| experiment | result | kept? |
|---|---|---|
| **feature fusion**: the transformer also reads the tabular features (20-run designed experiment) | **+0.053 AP**, the largest effect in the project | yes |
| learning rate 1.2e-3 with a 128-wide feature projection | +0.022 AP, better at all 3 seeds | yes |
| averaging 3 seeds | +0.013 AP | yes |
| dropout 0.35 vs none | −0.012 AP, too small to tell from zero | no |
| removing time information (2-layer model) | event order plus time vs neither: +0.030 AUROC; a time-gap attention bias added nothing | time encoding kept, gap bias removed |
| extra training examples from earlier cutoffs | no gain or harmful, in 5 tests | no |
| pretraining on next-event prediction, then fine-tuning (as the brief recommends) | no consistent effect, on pre-anchor events or on whole training timelines (AP −0.0041 [−0.0103, +0.0043]) | no |

Pretraining did learn (next-event loss fell from 6.20 to 2.95), but it re-reads the same
2,791 patients, and the number of patients is what limits every result here.

**Combining the models.** The three models disagree about which patients are at risk (the
rank correlation of GBDT and transformer scores averages 0.574 across conditions), so
averaging their logits beats each one alone. Averaging distorts the probabilities (it
predicts only 0.64× as many diagnoses as happen), so each condition gets a Platt correction,
`sigmoid(a · logit + b)`, fitted on out-of-fold predictions.
`python -m src.reproduce --from-hub sallamsaka/M31-Coding-Test` rebuilds `predictions.csv`
from the published files, matching the submission to 7.4e-08.

---

## 3. AI workflow

I used Claude Code as an agent in my terminal. It wrote nearly all of the code, ran every
experiment and searched the literature; I decided what to research, what counted as
evidence and what shipped. The project took 7 days and about 290 of my messages.

**Research.** I ran seven rounds of literature search, each with 3 to 20 research agents in
parallel, covering about 370 arXiv papers. Each round fed a decision: CEHR-BERT is why time
is concatenated to the embedding rather than added to it, EHRSHOT is why LR and GBDT were
built as serious competitors, and Delphi-2M is the precedent for age at each event. I
usually started a round when a plan seemed to rest on intuition.

**Planning.** Every non-trivial step started as a written plan that I approved: 42 plans, 28
sent back with changes. Each plan had to rank its options by expected effect and cost.
Adoption rules were written down before long runs, so a result could not be reinterpreted
afterwards. I rejected its first evaluation design, a fixed private slice of training
patients, in favour of the 5-fold cross-validation used throughout.

**Overnight runs.** Training is CPU-only and a transformer configuration takes 15 to 60
minutes, so it wrote job chains that ran overnight, retried failures and resumed from
checkpoints. It also wrote the 145 tests, the figures, the Hugging Face upload and the
reproduction script.

**Asking until I understood.** I had not trained a transformer from scratch before, so
whenever it used a term, reported a result or offered options I did not understand, I asked
it to explain before deciding. Walking through the model this way is where the adopted
change came from: the model had no sense of the patient's age at each event.

---

## 4. Interpretation

![Per-condition test-fold AUROC](../outputs/figures/per_condition_cv.png)

Per-condition AUROC ranges from 0.44 to 0.99, depending on whether anything recorded before
the anchor predicts the condition. To see what does, I also scored each condition with
single facts: age at the anchor, and whether the patient died (known for training patients,
used only for this check).

**Diseases people die of: 0.90–0.99** (heart failure, heart attack, pneumonia, lung and
prostate cancer). In Synthea these are diagnosed almost only in patients who die: pneumonia
in 133 of its 133 cases, heart failure in 224 of 229. The anchor explains why. For the 43% of
patients who die, the last encounter is a "Death Certification" 1 to 15 days after death, so
their five-year window is their last five years of life. When a record ends is also
visible: alive patients' records run to July 2021, putting their anchor around 2016, while
patients who died have earlier anchors (of those anchored before 2010, 1,027 of 1,033 died).
The model sees no dates, but a logistic regression on its features predicts death at AUROC
0.991. It does more than detect "about to die", though: scored only among patients who die,
it still picks the right disease (prostate cancer 0.970, pneumonia 0.927, heart attack
0.862, lung cancer 0.850, heart failure 0.767). A real deployment would not have the first
part, because nobody knows whether a patient's record will end within five years.

**Set by age or sex: 0.93–0.98.** Normal pregnancy (young women only), obesity, the prostate
conditions (men only).

**More common with age: 0.72–0.86.** Osteoporosis, Alzheimer's, stroke, atrial fibrillation.
Age alone scores 0.75–0.78, and the model adds a little.

**Acute, one-off events: 0.44–0.71 (the sixteen red bars).** Sinusitis, sore throat,
sprains, cuts, concussion. Neither age nor death predicts them, so no model can rank them.
Fractures in older patients do a little better, because age predicts them.

**Why mAP (0.24) is lower than AUROC (0.78).** AP rewards putting true cases at the very top
of the list, and most conditions are rare: the typical one is newly diagnosed in 3.3% of
patients, where ranking at random scores about 0.03 AP.

These patterns come from how Synthea generates patients, so which conditions are easy here
says little about which would be easy in a real hospital.
