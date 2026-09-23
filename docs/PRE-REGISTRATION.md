# Pre-registration

**Written 2026-09-21 02:47, at commit `52446dd`, before the designed experiment,
the LR factorial, the seed replicates, the learning-rate basin and the time
ablation had produced a single number.** That is the only property that makes
this document worth anything. Everything below is falsifiable by results that do
not exist yet.

Its purpose is narrow and defensive: this project has read its validation set
~26 times, and without a record of what was decided in advance, any favourable
result is indistinguishable from a forking path. What follows fixes the
decisions while they are still cheap to fix.

---

## 0. The contamination ledger — what is already compromised

Stated first, because a pre-registration written after the fact about *some*
things and not others is worse than none.

| object | status |
|---|---|
| provided **test** set (358) | **no labels** (0 of 2,511 post-anchor rows). Cannot be scored by us at all. |
| provided **validation** set (365) | read **~26 times**. Winner's-curse bound **+0.038 to +0.051** — larger than every difference ever measured between our models. |
| all feature engineering | done before any carve. Not de-biased by anything below. |
| the 12 search-ledger runs | architecturally superseded (§E10.5) and leak-carrying (§D17/§D18). Excluded from every claim. |
| **locked test set** (358) | carved from train, **never read**. One read, defined in §7. |
| **CV pool** (2,433) | the instrument for everything else. Read-count tracked. |

**Consequence, stated plainly:** no number computed on the provided validation
set is a de-biased estimate of anything, and the report must not present one as
such. The validation set survives in this project as a *descriptive* artifact
and a source of per-label prevalence, not as an arbiter.

---

## 1. Primary and secondary metrics

**Primary: macro average precision (AP).** Unchanged from §G2, deliberately.

**Co-secondaries, pre-specified: macro AUROC, Brier skill score.**

### 1.1 The documented deviation, and why it is documented rather than taken

§E8.5 measured AUROC as the more *discriminating* metric on the one
within-family comparison available, and an earlier draft of the plan proposed
switching the primary to AUROC on that basis. **That switch is not taken.**
Changing the primary metric after seeing which metric flatters the comparison is
a forking path regardless of how well motivated it is, so macro AP remains
primary and the reliability analysis is reported as a deviation *considered and
declined*.

**Selection during development may use whatever metric is measured to select
best** — currently AP, per §E11 — because selection is a design decision, not a
confirmatory claim. Any such use is labelled exploratory in the report. This is
the selection-vs-inference split and it is load-bearing:

> **Intervals govern what we claim. P(A>B) governs what we pick.**

### 1.2 P(A>B) is reported beside every interval

Not as a significance test. A gate on the interval alone false-negatives ~90% of
the time at this resolution against ~30% for P(A>B) at γ=0.75 (Bouthillier et
al., MLSys 2021). Reporting only "not resolvable" discards most of the evidence
in a comparison we already paid for.

### 1.3 Metrics that will NOT be selected on, fixed now

- **`ap_norm`** — `(AP − prev)/(1 − prev)`. Computed, but Richardson et al.
  (Patterns 2024) tested exactly the three obvious normalisations and all three
  fail. It is not a valid cross-label comparison (§E8.3).
- **MCC, at any threshold** — its threshold-free limit is AUROC (φ is Kendall's
  τ_b on a 2×2; letting the cut vary gives Somers' D = 2·AUROC − 1), so it adds
  no ranking information. Reported at two pre-specified thresholds only.
- **Raw macro Brier** — at 1% prevalence the uncertainty term is ≈0.0099 against
  a resolution of ≈2e-4, so real differences are ~1% relative.
- **Somers' D, Gini** — affine in AUROC. One row, not three.

---

## 2. The estimand

For `Y = 1{condition j first diagnosed within 5 years of the anchor}`:

```
E[Y | x] = P(T <= 5y, cause = j | x) = CIF_j(5y | x)
```

the **subdistribution (Fine–Gray) cumulative incidence**. 155 of 365 validation
patients (42.5%) die inside the window, and labelling a year-2 decedent `Y=0` is
the *correct* label for this estimand. With complete follow-up the IPCW weights
are 1. No competing-risks machinery is required and none will be added.

---

## 3. Decision rules, fixed in advance

1. **Act on the shrunken effect estimate, never the argmax, and never on a
   significance gate.** Shrinkage (`effects.shrink`, James–Stein) removes the
   winner's-curse optimism without discarding real-but-small effects, which the
   literature says dominate here (Probst et al. measure individual
   hyperparameters at 0.001–0.002, i.e. below our noise floor).
2. **"Not resolvable" is a reportable result**, not a failure, and is the
   *predicted* outcome for most factors (§R2).
3. **One partition, one K, for anything compared.** Unequal K applies the
   fit-size tax asymmetrically and would let us measure data-hunger and report
   it as inferiority.
4. **Per-fold mean ± SD is primary; pooled OOF is secondary** (§W1), with the
   raw-vs-rank-normalised gap reported as the measured heterogeneity budget.
5. **σ sets N, not the factor count** (§E14). If σ comes back large, the honest
   responses are more runs, a larger target effect, or "not resolvable" — not a
   narrower design.

---

## 4. Multiplicity policy

Two families, treated differently because they have different dependence
structures.

- **The designed experiment and the LR factorial.** Contrasts are orthogonal by
  construction, which is Benjamini–Yekutieli's studentized case. **One-sided BH
  at q = 0.15.** One-sided because we only care about improvement, and because
  it moves us out of the two-sided correlated case the theorem does not cover.
  BH 1995 names 2^k screening as a founding application and asks for q above a
  conventional α.
- **The 40 labels.** Plausibly positively dependent, so BH — but note the
  effective number of tests is **~35, not 40** (one pair has φ = 1.000), so any
  adjustment assuming 40 independent tests is wrong in both directions.
- **Not used:** Storey / q-value / SAM. The adaptive π₀-estimating variants are
  the fragile ones under dependence.

---

## 5. Planned slices (§W4)

Fixed now so they cannot be chosen after seeing which one is flattering:
**died-in-window vs survived** (first), age band, sex, history length,
sequence-truncated vs not, prior-condition count.

**Stated in advance:** the died-in-window split is a *subgroup-performance*
check. It is **not** a test of whether the model is "really predicting death" —
asking whether AUROC is equal within each death stratum is a different question
from whether the anchor rule selects on the future, and only the first is
answered (§E8.2, which corrected an earlier claim of mine).

---

## 6. Planned sensitivity analyses

All pre-specified: excluding decedents; right label edge closed vs open (the 119
boundary positives); at-risk mask on/off; dedup on/off; `min_patients_per_code`;
truncation policy; the de-duplicated ~35-label macro.

---

## 7. Our carved test set — AMENDED: measure freely, never select

> ⚠ **AMENDED.** The "one question, one read" rule below was wrong and is
> withdrawn. The winner's curse comes from **selecting** on a set, not from
> measuring on it: scoring M models leaves every score unbiased, and only
> taking the `argmax` over them costs `sigma*sqrt(2 ln M)`. The rule that replaces it:
> **select on CV folds, measure and report here as often as we like.**
>
> The cost of the old rule was total. Verified: this set carried **zero** numbers
> while every result in the project — the 20-run design, 7 ablations, 5 basin
> runs, 33 Phase B arms — sat on the provided validation set, whose
> winner's-curse bound (+0.027) exceeds every difference being measured. See D30.
>
> **What survives unchanged from below:** the power argument. At 358 patients with
> 3 positives on the rarest label, it **cannot rank** LR against GBDT against the
> transformer (§E6: separating 0.750 from 0.800 at 80% power needs 565+565).
> Report every model here with an interval; choose between them elsewhere.

### Original text, retained so the change is legible

**Question, fixed now:** *does the shipped model's macro AP exceed the
at-risk-mask-only baseline, with an interval?*

**It will not be asked anything else.** 358 patients, 3 positives on the rarest
label. §E6 says separating 0.750 from 0.800 at 80% power needs 565 + 565, so it
**cannot rank LR against GBDT against the transformer** and will not be used to.

**What it does and does not de-bias:** it de-biases the **fit**, not the
**design**. 26 validation reads and all feature engineering preceded the carve,
so Cawley & Talbot's "evaluate the algorithm *plus* its selection procedure"
standard is only partially met. It is disjoint from the dev split by
construction (dev is the first 558 of the seed-12345 permutation, locked is the
last 358) but the patients were inside the training set of the 12 superseded
search runs — weaker contamination than selection membership, but not nothing.

`python -m src.cross_validate --locked` deliberately prints these terms and
**runs nothing**. The read is a separate, deliberate act.

---

## 8. Predictions, recorded so they can be wrong

The point of this section is that it is checkable. Each is a commitment.

| # | prediction | confidence |
|---|---|---|
| 1 | **Most design factors will not be resolvable.** Probst et al. put individual hyperparameter tunability at 0.001–0.002 against our noise floor. | high |
| 2 | **Pretraining will not resolve** at a ~550k-token corpus (75× under Chinchilla). If it *does*, that is evidence about Synthea's rule-based generators, not about the method, and will be reported as such. | ~65% |
| 3 | **Learning rate will be the largest main effect**, per Greff et al.'s fANOVA (>2/3 of variance). It was absent from `search.py` entirely. | moderate |
| 4 | **The LR factorial will find at least one feature block above 0.002** that §C2 called "inside the noise floor" — because §C2's floor was ~0.01 and the factorial's MDE is ~0.0012 (§E12.1). | moderate |
| 5 | **The transformer will not beat LR** on the CV instrument. If it does, the first action is a bug hunt, not a paragraph. | moderate |
| 6 | **The time ablation will not resolve** at one run per arm: its MDE is 0.0333 against a project-wide spread of 0.0148. The `multiset` contrast is the only one plausibly large enough. | high |
| 7 | **Curvature will be detectable in lr** — a two-level contrast straddling a basin reads near zero, which is why centre points are in the design. | low–moderate |
| 8 | **The design's predicted-best cell will UNDERPERFORM its prediction.** Recorded before the runs, alongside the numeric prediction in `outputs/design_ledger.jsonl` (`confirm_prereg`). Reasons below. | moderate |

### 8.1 Prediction 8, stated in full before the runs

`propose_config` ranks all 32 cells of the full factorial from the 16 that were
run, and its top cell on **both** metrics is one of the 16 that were **not**.
Its predictions: **macro AP 0.2500, macro AUROC 0.7816**, against a best
observed corner of 0.2432 / 0.7638.

Three reasons to expect it to fall short, all structural rather than
pessimistic:

1. **The model is saturated.** 16 runs, 16 parameters, zero residual degrees of
   freedom. It reproduces every observed corner exactly, so nothing inside the
   experiment can contradict it.
2. **The extrapolation rests entirely on effect heredity.** Under the generator
   I = ABCDE every two-factor interaction is aliased with a three-factor one,
   and predictions for the complementary fraction assume all three-way and
   higher terms are zero. That assumption is untestable from inside the design
   -- which is the whole reason for running this.
3. **A predicted maximum over 32 cells is itself a selected quantity.** Even
   with shrinkage, the argmax of a fitted surface is biased upward.

**The comparison SE is sqrt(sigma^2/R + sigma^2), not s/sqrt(R).** For a
saturated two-level design Var(prediction) = sigma^2 at every cell, so the
prediction is exactly as noisy as one run and the target is not a fixed number.
At R = 3 seeds that is 1.15*sigma, roughly double the naive 0.58*sigma. Fixed in
advance so a miss cannot be declared on an SE chosen after seeing the result.

**Falsification threshold:** a prediction error beyond 2 SE on the primary
(macro AP) counts as prediction 8 confirmed and, more importantly, as evidence
that `propose_config`'s config recommendation should not be acted on -- the
design's trustworthy output would then be its effects alone.

**A weaker question is also recorded, because it survives either outcome:**
whether the proposed cell beats the **mean of the 16 corners actually run**.
That comparison does not depend on the point prediction being right, only on the
ranking being useful.

**If prediction 5 fails in the transformer's favour, the bug hunt comes first.**
Two of this project's most convincing findings were bugs, and the largest
apparent effect in the search (fusion, +0.0134 AP) turned out to be perfectly
confounded with a leak.

---

## 9. Scope

Every claim is about **pipeline quality on Synthea-generated patients, not
clinical performance on humans.** Synthea conditions come from rule-based
generative modules — the mechanism behind the φ = 1.000 label pair. The anchor
rule selects on the future. The task is data-limited, and the
highest-value intervention (more patients) is out of scope, so every
architectural result here is a refinement on an axis our own measurement says is
not the binding one.

DECIDE-AI, CONSORT-AI, SPIRIT-AI and STARD-AI are out of scope: this is a
methods exercise on synthetic data, not a clinical study.
