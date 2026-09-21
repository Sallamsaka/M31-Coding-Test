# Why the model needs time — derived, not asserted

## 0. Notation

| symbol | meaning |
|---|---|
| $p$ | a patient |
| $T$ | number of events in one patient's pre-anchor timeline |
| $i \in \{0,\dots,T-1\}$ | **position index** — where an event sits in the list |
| $r$ | **recency rank** — $r=0$ is the most recent event before the anchor |
| $t_i$ | wall-clock timestamp of event $i$ |
| $a$ | the anchor date |
| $\Delta_i = a - t_i$ | time before anchor, in days |
| $\Delta_{ij} = \lvert t_i - t_j\rvert$ | elapsed time between two events |
| $d$ | model width (192) |
| $x_i \in \mathbb{R}^d$ | the vector fed into layer 1 at position $i$ |
| $X \in \mathbb{R}^{T\times d}$ | all of them stacked, one row per position |
| $W_Q, W_K, W_V \in \mathbb{R}^{d\times d}$ | learned projection matrices |
| $P$ | a permutation matrix; $PX$ reorders the rows of $X$ |
| $H$, $L$ | number of heads, number of layers |

---

## 1. Without an explicit signal, a transformer cannot see order at all

**Theorem 1 (permutation equivariance).** Self-attention without positional information is permutation-equivariant.

*Proof.* Let $Q = XW_Q$, $K = XW_K$, $V = XW_V$. Permuting the input gives $Q' = PXW_Q = PQ$, and likewise $K' = PK$, $V' = PV$. The score matrix becomes

$$S' = \frac{Q'K'^{\top}}{\sqrt{d_k}} = \frac{(PQ)(PK)^{\top}}{\sqrt{d_k}} = P\,\frac{QK^{\top}}{\sqrt{d_k}}\,P^{\top} = PSP^{\top}.$$

Conjugation by $P$ permutes rows and columns together, and row-wise softmax is applied independently per row, so $\operatorname{softmax}(PSP^{\top}) = P\operatorname{softmax}(S)P^{\top}$. Hence

$$\operatorname{Attn}(PX) = P\operatorname{softmax}(S)P^{\top}PV = P\operatorname{softmax}(S)V = P\cdot\operatorname{Attn}(X).\qquad\blacksquare$$

**Corollary 1.** A transformer with no positional signal computes a function of the **multiset** of tokens. Under any permutation-invariant readout (mean-pooling, or a `[CLS]` token attending to everything), reordering the input returns *exactly the same output*. Order is not merely under-used — it is unobservable.

So injecting a positional signal is not a refinement. It is the thing that makes it a sequence model.

---

## 2. The real choice is *what function of what* gets injected

Write the injected signal at position $i$ as $\pi_i$. Two families:

$$\pi_i = E_{\text{pos}}[\,i\,] \quad \text{(rank)} \qquad\qquad \pi_i = \varphi(\Delta_i) \quad\text{(time)}$$

**Proposition 2.** Under rank injection the network output is measurable with respect to $\sigma(\text{codes},\,\text{ranks})$. Any dependence on $\Delta$ is representable *only* to the extent $\Delta$ is recoverable from rank.

That makes it an empirical question: **how much of $\Delta$ does rank actually determine?**

### 2.1 Measured on the 2,791 training patients

Distribution of $\Delta$ at a *fixed* recency rank, across patients:

| rank $r$ | $n$ patients | median $\Delta$ (days) | p10 | p90 | p90/p10 | IQR (days) |
|---|---|---|---|---|---|---|
| 0 | 2791 | 33 | 7 | 224 | **28.1×** | 89 |
| 1 | 2791 | 35 | 7 | 230 | 28.9× | 90 |
| 2 | 2791 | 35 | 9 | 243 | 24.4× | 102 |
| 5 | 2788 | 70 | 19 | 342 | 17.1× | 175 |
| 10 | 2784 | 84 | 21 | 370 | 16.9× | 207 |
| 20 | 2769 | 183 | 27 | 592 | 21.2× | 360 |
| 50 | 2654 | 546 | 168 | 1120 | 6.6× | 417 |
| 100 | 2238 | 1114 | 399 | 1752 | 4.4× | 706 |
| 200 | 1304 | 1608 | 387 | 10773 | 27.8× | 4554 |

Read the first row: **"the most recent event before the anchor" happened anywhere between a week and eight months ago** — a 28-fold spread, IQR of 89 days. Rank 0 does not correspond to a fixed amount of elapsed time, and the spread never closes.

### 2.2 The same fact in bits

Discretise $\Delta$ into 32 equal-mass bins and rank into 32 bins, over all events with $r < 512$:

$$H(\Delta) = 4.989 \text{ bits},\qquad H(\Delta \mid r) = 4.309 \text{ bits}$$

$$\boxed{\;I(r;\Delta) = 0.680 \text{ bits} = 13.6\%\ \text{of}\ H(\Delta)\;}$$

and $\operatorname{corr}(\log r, \log \Delta) = 0.656$, so $R^2 = 0.430$.

> **A rank-only model is blind to 86.4% of the temporal information in this dataset.**

### 2.3 Does timing predict *by itself*, with no code identity?

Features built purely from timestamps, scored by univariate AUROC against the 40 labels:

| pure-timing feature | mean \|AUC−0.5\| | max \|AUC−0.5\| | labels >0.05 |
|---|---|---|---|
| history span (years) | **0.1181** | **0.3693** | 26 / 40 |
| log #events | 0.0876 | 0.2989 | 26 |
| log #distinct days | 0.0754 | 0.2548 | 23 |
| event rate, last 2y | 0.0644 | 0.2640 | 20 |
| mean inter-event gap | 0.0539 | 0.2119 | 18 |
| lifetime event rate | 0.0514 | 0.2075 | 17 |
| days since last event | 0.0409 | 0.1886 | 12 |
| burstiness ($\sigma/\mu$ of gaps) | 0.0403 | 0.1034 | 12 |

For scale, the mean $|AUC-0.5|$ across all 654 **code-count** features is **0.0143**. History span alone is 8× that on average, and reaches AUC **0.87** on its best label.

**Adversarial reading — and this one is serious.** History span and event count are heavily confounded with **age**, and age predicts disease. This table is therefore *not* clean evidence that inter-event timing matters; it is evidence that temporal aggregate structure matters, of which age is probably the dominant component. The honest decomposition is to regress each feature on age at anchor and re-score the residual. Until that is run:

- treat rows 1–3 as **age proxies**;
- treat rate / gap / burstiness (rows 4–8) as the genuinely timing-specific signal — weaker at $\approx 0.04$–$0.06$, but still 3–4× the mean code feature.

Either way the conclusion for architecture is unchanged: a model that cannot represent $\Delta$ cannot represent any of these.

---

## 3. Four ways to inject time, with derived costs

### (a) Gap tokens — insert a bucketed interval token between consecutive events

**Fidelity.** $\Delta_{i,i+1}$ is directly observable. But $\Delta_{ij}$ for $|i-j| = k$ requires *summing* $k$ intervening bucket values. Attention computes

$$\text{out}_i = \sum_j \alpha_{ij} v_j, \qquad \sum_j \alpha_{ij} = 1,$$

a convex combination — a weighted **average**, not a sum. Representing $\sum_{m} \delta_m$ over a span of $k$ requires either $\alpha \approx 1/k$ with values scaled by $k$ (so the model must first infer $k$), or composition across layers. Expressible, not free.

**Cost.** Measured **+19.5%** sequence length on this dataset. Since the attention term is $O(T^2)$, it grows by $1.195^2 = 1.43\times$. Parameters $\approx 12d = 2{,}304$.

### (b) Additive embedding (BEHRT) — $x_i = e_{\text{code}}(i) + e_{\text{age}}(i)$

**Cost.** Zero length. Parameters = one bucket table.

**Failure mode, and it is derivable.** Expand the attention logit for $x = w + \pi$:

$$q_i^{\top}k_j = (w_i+\pi_i)^{\top}W_QW_K^{\top}(w_j+\pi_j) = \underbrace{w_i^{\top}Mw_j}_{\text{content}} + \underbrace{w_i^{\top}M\pi_j + \pi_i^{\top}Mw_j}_{\text{two cross terms}} + \underbrace{\pi_i^{\top}M\pi_j}_{\text{position}}$$

with $M = W_QW_K^{\top}$. The cross terms compare a *content* vector against a *time* vector through a single shared projection. TUPE's argument is that these are mostly noise; untangling them reached the same GLUE quality **in 30% of the pretraining steps** — a sample-efficiency gain, which is precisely the currency we are short of at $n=2{,}791$.

CEHR-BERT measured the concrete consequence: summing time/age embeddings was **worse than injecting no time at all in 7 of 8 task-metric cells** (discharge-death PR-AUC 52.7 → 31.6), with roughly 3× the variance of any other variant.

### (c) Continuous encoding replacing the positional table (Delphi-2M)

$$x_i = W\big[\,e_{\text{code}}(i) \,\Vert\, \varphi(\Delta_i)\,\big]$$

concatenate-then-project, which avoids the cross terms in (b) by giving content and time disjoint input coordinates before a single learned mixing matrix.

**Cost.** Zero length. One $(d + d_t)\times d$ projection $\approx 20$k parameters, plus the basis.

**Conditioning — this is where the design can silently fail.** $\Delta$ spans $[0,\ \approx\!29{,}200]$ days: 4.5 orders of magnitude. A sinusoidal ladder $\omega_k = \beta^{-2k/d}$ covers a period ratio of $\beta$; with the standard $\beta = 10^4$ that is $10^4 < 10^{4.5}$, so the coarse end saturates. Two fixes:

1. **Log-compress the input.** $u = \log(1+\Delta) \in [0,\ 10.28]$. Dynamic range collapses from $10^{4.5}$ to $\approx 10$, and it encodes the clinical intuition that 1 day → 2 days is comparable in significance to 1 year → 2 years.
2. **Learn the frequencies and keep a non-periodic term** (Time2Vec / mTAND). The linear term is what stops a purely periodic basis aliasing two very different $\Delta$ onto the same code:

$$\varphi(u)_0 = \omega_0 u + \alpha_0,\qquad \varphi(u)_k = \sin(\omega_k u + \alpha_k),\ \ k = 1,\dots,K$$

Parameter cost $2(K+1)$; at $K=32$ that is **66 numbers**.

### (d) Attention bias on pairwise elapsed time

$$\operatorname{score}(i,j) = \frac{q_i^{\top}k_j}{\sqrt{d_k}} + b\big[\operatorname{bucket}_{\log}(\Delta_{ij}),\, h\big]$$

**Fidelity.** The only option that makes **pairwise** elapsed time visible directly — no summation, no cross-layer composition.

**Cost.** Zero length. $32 \times H = 192$ scalars.

**Two structural properties that matter here.** $b(0) = 0$ by construction, so simultaneous events attend at full strength with no order imposed — relevant because **97.7% of events in this dataset share a timestamp with at least one other event** (95.6% before date-only diagnoses were snapped onto the encounter that produced them; see the report, §1). And log-spaced buckets are the correct prior for a heavy-tailed $\Delta$, reproducing the gap-token inductive bias at zero sequence cost.

### Ranking

| | recovers $\Delta_i$? | recovers $\Delta_{ij}$? | length cost | params | external evidence |
|---|---|---|---|---|---|
| **(d) $\Delta_{ij}$ attention bias** | via (c) | **directly** | 0% | ~192 | STAR 0.785→0.832 avg AUC; TALE-EHR 0.919→0.941 |
| **(c) continuous, concatenated** | **exactly** | by subtraction, learnable | 0% | ~20k | Delphi-2M (2.2M params, Nature 2025); CORE-BEHRT +0.6–0.9 AUROC |
| (a) gap tokens | adjacent only | needs summation | **+19.5%** | ~2.3k | CEHR-BERT +1.0–3.7 AUC, but Context Clues found them *harmful at small context* |
| (b) additive embedding | yes | no | 0% | bucket table | **measured worse than no time at all**, 7/8 cells |
| (none) rank only | **13.6%** | no | 0% | 0 | Theorem 1 + the 0.68-bit result above |

**(c) and (d) are complementary, not competing.** (c) tells each position *when it is*; (d) tells each pair *how far apart they are*. Combined: ~20k parameters, 0% length. Adopt both.

---

## 4. What remains unproven

1. The pure-timing table is confounded with age. The residualised version has not been run, and it should be before any of it goes in the report.
2. **No published work ablates (a) vs (b) vs (c) vs (d) against each other on one dataset.** The ranking above composes evidence across separate papers with different data, scale and baselines — strictly weaker than a controlled comparison.
3. Every cited external number comes from a dataset ≥10× ours. Transfer of the *ranking*, not merely the magnitudes, is an assumption.
4. The cheapest resolution is to run it: four arms (rank-only / gap tokens / continuous / continuous+bias) on identical splits and seeds. At ~7 min/epoch that is one afternoon — and it is an ablation nobody has published.
