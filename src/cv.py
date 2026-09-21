"""Fold construction and the locked test carve.

Two objects, deliberately separate.

**The locked test set** is 15% of the provided train patients (419), set aside
and read exactly once, at the end, for one pre-registered question. It exists
because cross-validation cannot answer the question it is usually asked to:
pooled or averaged, CV estimates the expected performance of a *training
procedure* at n = N(K-1)/K, not the performance of the single model we ship.
The locked set estimates the shipped artifact, and it is one line of code, so
it is immune to all ten fit sites, the pooling artifact and every
fold-construction subtlety that the rest of this module has to get right.

**The CV pool** is the remaining 2,372, partitioned into patient-grouped folds
for everything else.

Three honest caveats, recorded here rather than in a footnote.

1. **The locked set de-biases the FIT, not the DESIGN.** 26 validation reads
   and all feature engineering happened before it was carved. It cannot undo
   those. It answers "how does the shipped model do on patients it never
   trained on", not "how would this whole research process do again".
2. **It is disjoint from the dev split but not from past training sets.** The
   dev split is the *first* 558 of the seed-12345 permutation and the locked
   set is the *last* 419, so no patient used for selection is in it. They were
   however inside the training set of the 12 completed search runs. Training
   membership is a much weaker contamination than selection membership, and
   the final model is refit on the pool only -- but it is not nothing.
3. **It is undersized for ranking.** 419 patients, rarest label 5 positives.
   It can confirm one number with an interval. It cannot choose between LR,
   GBDT and the transformer, and it must not be asked to.

Run: ``python -m src.cv`` prints the partition and its integrity checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data.cohort import load_cohort
from .data.examples import ExampleConfig, grouped_folds, load_examples

__all__ = ["SPLIT_SEED", "LOCKED_TEST_FRAC", "LOCKED_TEST_N", "locked_test_pids",
           "cv_pool_pids", "fold_masks", "main"]

# The one permutation everything in this project derives a split from.
# Deliberately not cfg.seed: every configuration ever compared must see the
# same partition, while training randomness varies.
SPLIT_SEED = 12345
# The provided test set is 358 patients. Ours is sized to match it exactly, so
# its precision equals what the real test would give -- which is the entire
# point of a proxy. 15% (419) was an arbitrary choice that made ours 61
# patients larger than the thing it stands in for.
LOCKED_TEST_N = 358
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

    Verified on this cohort at frac=0.15: 419 patients, **all 40 labels have
    at least one positive** (rarest 5), and zero overlap with dev. No
    stratification is applied because none is needed for scorability here; if
    the fraction changes, re-check coverage before trusting it.
    """
    pids = _train_pids(root)
    order = np.random.default_rng(SPLIT_SEED).permutation(pids)
    n = LOCKED_TEST_N if frac == LOCKED_TEST_FRAC else int(round(frac * len(pids)))
    return set(order[-n:].tolist())


def cv_pool_pids(root: str = ".", frac: float = LOCKED_TEST_FRAC) -> np.ndarray:
    """Train patients minus the locked test set. Everything else fits here."""
    pids = _train_pids(root)
    locked = locked_test_pids(root, frac)
    return np.sort(np.array([p for p in pids if p not in locked]))


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
    print(f"  locked test (15%)       : {len(locked):,}   read ONCE, at the end")
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
