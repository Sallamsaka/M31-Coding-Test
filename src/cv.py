"""Fold construction and the locked test carve.

Two objects, deliberately separate.

**The locked test set** is 358 of the provided train patients, set aside and
read exactly once, at the end, for one pre-registered question. 358 is not a
round fraction: it is exactly the size of the provided test set, so this set's
precision equals what the real test would give. It exists
because cross-validation cannot answer the question it is usually asked to:
pooled or averaged, CV estimates the expected performance of a *training
procedure* at n = N(K-1)/K, not the performance of the single model we ship.
The locked set estimates the shipped artifact, and it is one line of code, so
it is immune to all ten fit sites, the pooling artifact and every
fold-construction subtlety that the rest of this module has to get right.

**The CV pool** is the remaining 2,433, partitioned into patient-grouped folds
for everything else.

Three honest caveats, recorded here rather than in a footnote.

1. **The locked set de-biases the FIT, not the DESIGN.** 26 validation reads
   and all feature engineering happened before it was carved. It cannot undo
   those. It answers "how does the shipped model do on patients it never
   trained on", not "how would this whole research process do again".
2. **It is disjoint from the dev split but not from past training sets.** The
   dev split is the *first* 558 of the seed-12345 permutation and the locked
   set is the *last* 358, so no patient used for selection is in it. They were
   however inside the training set of the 12 completed search runs. Training
   membership is a much weaker contamination than selection membership, and
   the final model is refit on the pool only -- but it is not nothing.
3. **It is undersized for ranking.** 358 patients, rarest label 3 positives.
   It can confirm one number with an interval. It cannot choose between LR,
   GBDT and the transformer, and it must not be asked to. This is not a defect
   of the carve -- the real 358-patient test set is equally thin, and a proxy
   that were better resolved would be misrepresenting the deliverable.

Run: ``python -m src.cv`` prints the partition and its integrity checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data.cohort import load_cohort
from .data.examples import ExampleConfig, grouped_folds, load_examples

__all__ = ["SPLIT_SEED", "LOCKED_TEST_FRAC", "LOCKED_TEST_N", "locked_test_pids",
           "cv_pool_pids", "trainable_pids", "trainable_row_mask",
           "fold_masks", "main"]

# The one permutation everything in this project derives a split from.
# Deliberately not cfg.seed: every configuration ever compared must see the
# same partition, while training randomness varies.
SPLIT_SEED = 12345
# The provided test set is 358 patients. Ours is sized to match it exactly, so
# its precision equals what the real test would give -- which is the entire
# point of a proxy. 15% (419) was an arbitrary choice that made ours 61
# patients larger than the thing it stands in for.
# RETIRED -- 0, so nothing is carved and the CV pool is the full 2,791.
#
# The carve existed to give a clean fixed holdout. It is redundant: CV already
# rotates the test set, and the provided val is a fixed yardstick we are
# forbidden to train on anyway, so holding THAT out is free. This one cost 12.8%
# of the training pool (2,791 -> 2,433), worth about -0.005 AUROC on every model
# by §B1's learning curve, paid continuously.
#
# What replaces its one real use -- a cheap clean set to screen transformer arms
# on -- is fold 0 of the standard partition: arms train on folds 1-4 and score on
# fold 0. Part of the design rather than a separate carve, and it rotates.
LOCKED_TEST_N = 0
LOCKED_TEST_FRAC = LOCKED_TEST_N / 2791


def _train_pids(root: str = ".") -> np.ndarray:
    coh = load_cohort(root)
    return np.sort(coh.loc[coh.split == "train", "pid"].unique())


def locked_test_pids(root: str = ".", frac: float = LOCKED_TEST_FRAC) -> set[int]:
    """The **last** ``frac`` of the seed-12345 permutation of train patients.

    Taken from the tail, not the head, and that is the whole point: the dev
    split used by every search run is the *first* 558 of the same permutation,
    so drawing from the tail makes the two disjoint by construction rather
    than by a check that someone has to remember to run.

    Verified on this cohort at LOCKED_TEST_N=358: **all 40 labels have at least
    one positive** (rarest 3), and zero overlap with dev. No stratification is
    applied because none is needed for scorability here; if the size changes,
    re-check coverage before trusting it.
    """
    pids = _train_pids(root)
    n = LOCKED_TEST_N if frac == LOCKED_TEST_FRAC else int(round(frac * len(pids)))
    # RETIRED: n is 0, and `order[-0:]` is the WHOLE array, not an empty one --
    # so this must return early rather than slice. Getting that wrong would
    # silently carve out every training patient.
    if n <= 0:
        return set()
    order = np.random.default_rng(SPLIT_SEED).permutation(pids)
    return set(order[-n:].tolist())


def cv_pool_pids(root: str = ".", frac: float = LOCKED_TEST_FRAC) -> np.ndarray:
    """The CV pool: now ALL 2,791 train patients (the carve is retired)."""
    pids = _train_pids(root)
    locked = locked_test_pids(root, frac)
    return np.sort(np.array([p for p in pids if p not in locked]))


def trainable_pids(root: str = ".", frac: float = LOCKED_TEST_FRAC) -> set[int]:
    """Train patients MINUS the locked test set -- what any fit may legally see.

    This is the default universe for every fit in the project, not a
    cross-validation detail. `split == "train"` is 2,791 patients and **includes
    the locked 358**, so using it as a fit mask trains on the set carved to be
    the honest final estimate. Measured before this existed: the locked patients
    were 12.8% of every default fit set, and the only path that excluded them
    was `fold_masks`.
    """
    return set(cv_pool_pids(root, frac).tolist())


def trainable_row_mask(ex, root: str = ".", frac: float = LOCKED_TEST_FRAC):
    """Row mask over an example frame: train rows whose patient is not locked."""
    pool = trainable_pids(root, frac)
    is_train = (ex.split == "train").to_numpy()
    return is_train & np.isin(ex.pid.to_numpy(), list(pool))


def stratified_fold_of(Y, n_splits: int = 5, seed: int = SPLIT_SEED):
    """Assign patients to folds balancing PER-LABEL positive counts.

    Greedy iterative stratification (Sechidis, Tsoumakas & Vlahavas 2011), which
    is the multi-label analogue of a stratified split: a patient carries several
    labels at once, so you cannot stratify on one without unbalancing the rest.

    **The problem it fixes, measured.** Random patient-grouped folds already match
    the provided test set on every distributional axis -- events (KS p = 0.37 to
    0.96), history length, gap-to-anchor, age, sex, race, per-label prevalence --
    so none of that needs correcting. What they do NOT control is the rare-label
    tail: fold 1 of the current partition holds a label with **one positive**. A
    per-label AP on one positive is a coin flip, and macro AP averages all 40 with
    equal weight, so that fold's headline number is part noise. A fold with ZERO
    positives is worse still: the label is dropped and the macro silently averages
    over a different set of labels (§G1).

    Rarest labels are placed first, because they are the constrained ones -- once
    a label with 29 positives is spread, the common labels have enough mass to
    balance around it. Ties break toward the fold holding fewest patients, which
    keeps the folds near-equal in size without a separate balancing pass.

    ⚠ **WRITTEN, MEASURED, AND DELIBERATELY NOT ADOPTED.** `fold_masks` does not
    call this. It raises the per-label floor exactly as intended -- worst label
    across five folds goes 3 -> 5 against an ideal of 5.8, with fold sizes
    unchanged -- and the decision was still to keep random assignment.

    The reason is what the folds are *for*. They simulate the provided test set,
    which is one random draw of 358 patients, and a random draw can perfectly
    well land 1 positive on a rare label. Stratifying removes that possibility,
    so every fold becomes more stable than the thing it stands in for, and any
    spread reported across folds understates the spread we will actually face.
    Lower estimator variance is the right trade when selecting; a faithful
    simulation is the right trade when the number is meant to predict the
    deliverable. Here it is the latter.

    Kept because it is correct and tested, and because the trade flips if the
    folds are ever used purely for selection rather than for prediction.

    Returns an integer fold index per row of ``Y``.
    """
    Y = np.asarray(Y)
    n, L = Y.shape
    rng = np.random.default_rng(seed)
    totals = Y.sum(0)
    remaining = np.tile(totals / n_splits, (n_splits, 1)).astype(float)
    count = np.zeros(n_splits, int)
    assign = np.full(n, -1, int)

    for j in np.argsort(totals):                 # rarest label first
        members = np.flatnonzero((Y[:, j] == 1) & (assign < 0))
        rng.shuffle(members)
        for i in members:
            need = remaining[:, j]
            cand = np.flatnonzero(need == need.max())
            k = int(cand[np.argmin(count[cand])])
            assign[i] = k
            remaining[k] -= Y[i]
            count[k] += 1

    leftover = np.flatnonzero(assign < 0)        # patients with no positives
    rng.shuffle(leftover)
    for i in leftover:
        k = int(np.argmin(count))
        assign[i] = k
        count[k] += 1
    assert (assign >= 0).all()
    return assign


def fold_masks(root: str = ".", n_splits: int = 5, repeat: int = 0,
               ex_cfg: ExampleConfig | None = None,
               frac: float = LOCKED_TEST_FRAC,
               ) -> list[tuple[np.ndarray, np.ndarray]]:
    """``[(fit_rows, oof_rows), ...]``, patient-grouped, over the CV pool.

    ``fit_rows`` excludes the held-out fold **and** the locked test set, so a
    model fitted on it has seen neither. ``oof_rows`` additionally excludes
    augmented rows: rows-per-patient tracks record length, which is the A6
    confound, so scoring augmented rows would weight the macro by the very
    thing that drives the outcome.

    ``repeat`` varies the partition for repeated CV. Repeat 0 is canonical.
    Bouthillier et al. (MLSys 2021) find data resampling is the largest single
    variance source -- larger than weight initialisation -- so a single
    partition understates uncertainty and repeats are how that gets measured
    rather than corrected by formula.

    Note this partition is over the **pool**, not all of train, so K=5 fold 0
    is *not* the dev split. That equality holds for ``grouped_folds`` called
    with the full train pids, and is not inherited here.
    """
    ex = load_examples(root, ex_cfg or ExampleConfig())
    pool = cv_pool_pids(root, frac)
    seed = SPLIT_SEED + 1000 * repeat
    folds = grouped_folds(ex, n_splits, seed=seed, pids=pool)

    is_train = (ex.split == "train").to_numpy()
    in_pool = np.isin(ex.pid.to_numpy(), pool)
    is_real = ex.is_real.to_numpy() if "is_real" in ex.columns else np.ones(len(ex), bool)

    out = []
    for oof in folds:
        fit = is_train & in_pool & ~oof
        out.append((fit, oof & in_pool & is_real))
    return out


def main(root: str = ".") -> None:
    ex = load_examples(root, ExampleConfig())
    pid = ex.pid.to_numpy()
    train = _train_pids(root)
    locked = locked_test_pids(root)
    pool = cv_pool_pids(root)

    print("PARTITION")
    print(f"  provided train patients : {len(train):,}")
    print(f"  locked test             : {len(locked):,}   read ONCE, at the end (sized to the provided test)")
    print(f"  cv pool                 : {len(pool):,}")
    assert len(locked) + len(pool) == len(train)
    assert not (locked & set(pool.tolist()))

    from .train_finetune import inner_split_pids
    dev, _ = inner_split_pids(root, 0.2, 0.0)
    print(f"  locked AND dev overlap  : {len(locked & dev)}   (must be 0)")
    assert not (locked & dev)

    for k in (5, 10):
        fm = fold_masks(root, k)
        sizes = [int(o.sum()) for _, o in fm]
        cov = np.zeros(len(ex), int)
        for _, o in fm:
            cov += o.astype(int)
        leaked = [int((f & np.isin(pid, list(locked))).sum()) for f, _ in fm]
        overlap = any((fm[i][1] & fm[j][1]).any()
                      for i in range(k) for j in range(i + 1, k))
        print(f"\nK={k}")
        print(f"  oof sizes        : {sizes}")
        print(f"  pool rows scored : {cov.sum():,} of {int(np.isin(pid, pool).sum()):,}"
              f"   max per row = {cov.max()} (must be 1)")
        print(f"  locked rows in any fit set : {sum(leaked)} (must be 0)")
        print(f"  two folds share a row      : {overlap} (must be False)")
        assert cov.max() <= 1 and sum(leaked) == 0 and not overlap


if __name__ == "__main__":
    main()
