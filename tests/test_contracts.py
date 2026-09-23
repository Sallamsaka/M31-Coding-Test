"""The submission contract, and the invariants the example table must hold.

These are the failures that produce a *valid-looking* file that scores zero,
which is why they are tested rather than eyeballed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.cohort import load_cohort, load_target_codes
from src.data.examples import ExampleConfig, build_examples, grouped_folds
from src.predict import check_contract

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def codes():
    return load_target_codes(ROOT)


@pytest.fixture(scope="module")
def anchors():
    return pd.read_csv(ROOT / "test_anchors.csv", dtype=str)


def _valid(codes, anchors) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.uniform(0.01, 0.99, (358, 40)), columns=codes)
    df.insert(0, "patient_id", anchors["Id"].to_numpy())
    return df


# ------------------------------------------------------------------ T15 ----
def test_a_valid_submission_passes(codes, anchors):
    check_contract(_valid(codes, anchors), ROOT)


def test_codes_are_read_as_strings_not_integers(codes):
    """The quietest way to score zero.

    Every target code is numeric text. ``pd.read_csv`` without ``dtype=str``
    turns ``444814009`` into an int64, and ``'444814009' != 444814009``, so a
    grader joining on column names matches nothing and marks all 40 columns
    missing -- while the file still opens and still has 358 rows.
    """
    assert all(isinstance(c, str) for c in codes)
    assert all(c.strip() == c and c.isdigit() for c in codes)
    naive = pd.read_csv(ROOT / "target_conditions.csv")      # no dtype=str
    assert naive["CODE"].dtype != object, "premise of this test no longer holds"
    assert [str(c) for c in naive["CODE"]] == codes


@pytest.mark.parametrize("break_it,msg", [
    (lambda d: d.iloc[:357], "358"),
    (lambda d: d.rename(columns={d.columns[1]: "wrong"}), "header"),
    (lambda d: d.assign(**{d.columns[1]: 0.0}), "0,1"),
    (lambda d: d.assign(**{d.columns[1]: 1.0}), "0,1"),
    (lambda d: d.assign(**{d.columns[1]: np.nan}), "finite"),
    (lambda d: d.iloc[::-1], "order"),
])
def test_each_contract_violation_is_caught(codes, anchors, break_it, msg):
    with pytest.raises(AssertionError) as e:
        check_contract(break_it(_valid(codes, anchors)), ROOT)
    assert msg in str(e.value), f"wrong error for this violation: {e.value}"


def test_written_predictions_file_satisfies_the_contract():
    """Whatever is on disk right now must be submittable."""
    p = ROOT / "outputs" / "predictions.csv"
    if not p.exists():
        pytest.skip("no predictions.csv yet -- run `python -m src.run_baseline`")
    check_contract(pd.read_csv(p, dtype={"patient_id": str}).astype(
        {c: float for c in load_target_codes(ROOT)}), ROOT)


# ----------------------------------------------------- example invariants ---
@pytest.fixture(scope="module")
def ex():
    return build_examples(ROOT, ExampleConfig(stride_years=5))


def test_only_training_patients_are_augmented(ex):
    """Augmenting validation would change what validation measures.

    The graded quantity is one prediction per patient at one cutoff. A val set
    with six rows per patient measures something else, and averages away the
    very case-mix the metric is defined over.
    """
    for split in ("val", "test"):
        rows = ex[ex.split == split]
        assert rows.is_real.all(), f"{split} contains synthetic examples"
        assert rows.pid.is_unique, f"{split} has duplicate patients"


def test_real_examples_come_first_and_are_indexed_by_pid(ex):
    """The frozen label counts index the real block, so it must not move."""
    n = len(load_cohort(ROOT))
    assert ex.is_real.to_numpy()[:n].all()
    assert not ex.is_real.to_numpy()[n:].any()
    assert (ex.eid.to_numpy()[:n] == ex.pid.to_numpy()[:n]).all()


def test_synthetic_cutoffs_are_strictly_before_the_anchor(ex):
    cohort = load_cohort(ROOT)
    anchor = pd.Series(cohort.anchor.values, index=cohort.pid.values)
    syn = ex[~ex.is_real]
    assert (syn.cutoff < syn.pid.map(anchor)).all()
    # ...and land on the stride grid, not somewhere arbitrary.
    assert (syn.k >= 1).all()


def test_a_patient_never_spans_two_cv_folds(ex):
    """Two cutoffs five years apart share most of one record. Splitting rows
    at random puts near-copies on both sides of the boundary and reports a
    memorisation score as a validation score.
    """
    folds = grouped_folds(ex, n_splits=5, seed=0)
    assert sum(f.sum() for f in folds) == len(ex), "folds must partition the rows"
    seen: dict[int, int] = {}
    for i, f in enumerate(folds):
        for p in ex.pid[f].unique():
            assert seen.setdefault(int(p), i) == i, f"pid {p} appears in two folds"
    assert len(seen) == ex.pid.nunique()


def test_augmentation_off_reproduces_the_original_cohort():
    """The control arm must actually be a control.

    An earlier cache key covered `stride_years` and `min_events` only, so
    `augment_splits=()` silently loaded the augmented table and the two arms
    of the comparison were the same run. The key is now derived from the whole
    dataclass; this pins that.
    """
    plain = build_examples(ROOT, ExampleConfig(augment_splits=()))
    cohort = load_cohort(ROOT)
    assert len(plain) == len(cohort)
    assert plain.is_real.all()
    assert (plain.pid.to_numpy() == cohort.pid.to_numpy()).all()
    assert (plain.cutoff.to_numpy() == cohort.anchor.to_numpy()).all()

# ---------------------------------------------- stratified fold assignment ---
def test_stratified_folds_beat_random_on_the_rarest_label():
    """The whole point of stratifying: raise the FLOOR on per-label positives.

    Random patient-grouped folds already match the provided test set on every
    distributional axis measured (events KS p = 0.37-0.96, history length, age,
    sex, per-label prevalence), so there is nothing to correct there. What they
    do not control is the rare tail -- a fold can land 1 positive for a label,
    and a per-label AP on one positive is a coin flip that macro AP then
    averages in with equal weight.

    Asserted as an INEQUALITY against random on the same labels, not against a
    hard-coded number, so the test keeps meaning if the cohort changes.
    """
    import numpy as np

    from src.cv import stratified_fold_of

    rng = np.random.default_rng(0)
    n, L, K = 2000, 20, 5
    # Prevalences spanning the real range: one very rare label plus common ones.
    prev = np.concatenate([[0.012], rng.uniform(0.03, 0.25, L - 1)])
    Y = (rng.random((n, L)) < prev).astype(int)

    strat = stratified_fold_of(Y, K, seed=0)
    rand = rng.permutation(n) * K // n

    worst_s = min(Y[strat == k].sum(0).min() for k in range(K))
    worst_r = min(Y[rand == k].sum(0).min() for k in range(K))
    assert worst_s >= worst_r, (worst_s, worst_r)

    # and it must not achieve that by unbalancing the folds
    sizes = np.array([(strat == k).sum() for k in range(K)])
    assert sizes.max() - sizes.min() <= 1, sizes


def test_stratified_folds_assign_every_patient_exactly_once():
    """A patient in no fold is silently dropped from evaluation; a patient in
    two is scored twice, which breaks `bootstrap_macro`'s promise to resample
    patients rather than rows."""
    import numpy as np

    from src.cv import stratified_fold_of

    rng = np.random.default_rng(1)
    Y = (rng.random((500, 12)) < 0.1).astype(int)
    a = stratified_fold_of(Y, 4, seed=1)
    assert a.shape == (500,)
    assert a.min() >= 0 and a.max() <= 3
    assert np.bincount(a, minlength=4).sum() == 500


def test_stratified_folds_handle_a_label_with_no_positives():
    """An all-zero label must not crash the assignment or starve a fold.

    It cannot be balanced -- there is nothing to spread -- so the requirement is
    only that it is survived, and that the other labels are still stratified.
    """
    import numpy as np

    from src.cv import stratified_fold_of

    rng = np.random.default_rng(2)
    Y = (rng.random((300, 6)) < 0.15).astype(int)
    Y[:, 3] = 0
    a = stratified_fold_of(Y, 3, seed=2)
    assert set(np.unique(a)) == {0, 1, 2}
    sizes = np.array([(a == k).sum() for k in range(3)])
    assert sizes.max() - sizes.min() <= 1, sizes
