"""One row per **example** -- a (patient, cutoff) pair -- rather than per patient.

Why this exists
---------------
Everything downstream trained on 2,791 rows. Measured sample sizes at which
these model families stabilise are **9,960 for gradient boosting and 12,298
for a neural net** (Silvey & Liu 2024), so the whole project was running at
28% of the easier threshold.

A patient's record does not start at their anchor. Rolling the cutoff back
five years and re-deriving both features and labels from that earlier vantage
gives a genuinely new supervised example: different history, different
outcomes, same person. Measured availability at ``min_events=10``::

    no augmentation       ->   3,514 examples   (2,791 training rows)
    cutoff every 5 years  ->  16,378            ( 15,655)
    cutoff every 3 years  ->  26,275            ( 25,552)
    cutoff every 2 years  ->  38,415            ( 37,692)

An earlier estimate put stride 5 at 24,077. That was taken before the event
filter existed; these are the re-derived figures.

**What the row count hides.** Synthetic examples are not more of the same
data. Measured at stride 5, their positive rate is **0.0126 against the real
0.0467** -- augmentation multiplies rows by 4.6x but positives by only 2.2x
(5,214 -> 11,681), because rolling the cutoff back lands on a population 13
years younger and the 40 targets are adult-onset conditions. Anyone reading
"8.6x the data" off the row count is reading the wrong number.

Three properties this module is responsible for
-----------------------------------------------
**Only training patients are augmented.** Validation keeps exactly one example
per patient, at its real cutoff. Augmenting it would change what validation
measures -- the graded metric is one prediction per patient -- and the ReadMe
restricts validation to monitoring and model selection.

**A patient's examples must never span a cross-validation fold.** Two cutoffs
five years apart share most of their history, so scoring one while having
trained on the other is a within-patient leak wearing a between-patient
costume. :func:`grouped_folds` is the only fold generator callers should use.

**Synthetic outcome windows may extend past the anchor, and that is not a
leak.** At stride 5 a window ends exactly at the anchor. At stride 3 it ends
at ``anchor + 2y`` -- real conditions from a training patient's record, which
the ReadMe's "train only on the provided training split" explicitly permits,
because those patients *are* the split. What would be a leak is using a *val
or test* patient's post-anchor data, which is why augmentation is restricted
by split rather than by window arithmetic.

An earlier draft of the plan asserted ``cutoff + 5y <= anchor``. That is not a
leakage condition, it is a restatement of ``stride >= 5``, and enforcing it
would have silently forbidden the stride sweep. The real guards are
:func:`_assert_invariants`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.io import cached_parquet
from ..utils.timeutil import label_window_end
from .cohort import _deps, load_cohort, load_events

__all__ = ["ExampleConfig", "build_examples", "load_examples", "grouped_folds"]


@dataclass(frozen=True)
class ExampleConfig:
    stride_years: int = 5
    """Spacing between synthetic cutoffs.

    5 is the default because it is the point at which consecutive outcome
    windows stop overlapping, so every extra example contributes a genuinely
    fresh set of outcomes rather than re-weighting ones already counted.
    Shorter strides are legitimate and buy more examples; they trade
    independence for gradient signal, which is what the sweep measures.
    """

    min_events: int = 10
    """A cutoff is kept only with at least this many events strictly before it.

    The filter is on **events, not age**. Synthea starts every patient at
    birth, so a cutoff placed by calendar arithmetic alone lands in infancy
    for the older patients and yields a row with nothing in it to learn from.
    Filtering on events handles age as a side effect, measured at stride 5::

        min_events   examples   synthetic p5 age   under 18
                 5     20,684             18.1        4.9%
                10     16,378             21.0        2.9%
                25     12,411             25.5        1.8%
                50     10,008               --          --

    10 is the default: under-18 contamination falls below 3% while keeping
    16,378 examples, a third above the 12,298 at which a neural net's AUROC
    stabilises. Both this and ``stride_years`` are cheap to sweep on logistic
    regression (4 s a fit) and should be swept jointly.

    What the filter cannot fix is that synthetic examples are **7x sparser**
    (median 26 events before ``anchor - 5y`` against 189 before the anchor)
    and **13 years younger** (median age 50 against 64). That shift is
    intrinsic to rolling the cutoff back, and it is the measured reason to
    fine-tune on the real cutoffs last rather than treating all examples
    as exchangeable.
    """

    window_years: int = 5

    augment: bool = False
    """Off by default, because measured it makes things worse.

    Naive pooling cost 0.009 macro AUROC at stride 5 and 0.015 at stride 3,
    with C retuned per arm so it is not a regularisation artifact. The cause
    is measured too: trained on synthetic rows alone at matched n=2,791, val
    AUROC is 0.6772 against the real set's 0.7679. Earlier cutoffs are a
    different task, not more of the same one.

    The machinery is kept because the un-tested variant -- train broad, then
    fine-tune on real cutoffs only -- is a different proposition from pooling,
    and because a negative result with working code behind it is worth more
    than an untried idea.
    """

    augment_splits: tuple[str, ...] = ("train",)
    max_extra_per_patient: int | None = None

    @property
    def active_splits(self) -> tuple[str, ...]:
        return self.augment_splits if self.augment else ()


def _candidate_cutoffs(anchor: pd.Series, k: int, stride: int) -> pd.Series:
    """``anchor - k*stride`` years, calendar-correct and floored to midnight.

    ``DateOffset``, never ``Timedelta``: a fixed day count drifts by one or two
    days depending on how many leap years the span crosses, and every label
    boundary in this project sits exactly on a midnight.
    """
    return (anchor - pd.DateOffset(years=k * stride)).dt.normalize()


def build_examples(root: Path | str = ".",
                   cfg: ExampleConfig | None = None) -> pd.DataFrame:
    cfg = cfg or ExampleConfig()
    cohort, events = load_cohort(root), load_events(root)

    # Real examples first, so eid == pid on this block and the un-augmented
    # pipeline is row-for-row what it was. The frozen label counts stay
    # checkable against `examples.is_real`, which is worth the ordering
    # constraint on its own.
    real = pd.DataFrame({
        "pid": cohort.pid.to_numpy(np.int32),
        "cutoff": cohort.anchor.to_numpy(),
        "split": cohort.split.to_numpy(),
        "is_real": True,
        "k": 0,
    })

    extra: list[pd.DataFrame] = []
    if cfg.active_splits:
        sel = cohort[cohort.split.isin(cfg.active_splits)]
        anchor = pd.Series(sel.anchor.values, index=sel.pid.values)
        split_of = pd.Series(sel.split.values, index=sel.pid.values)

        ev = events.loc[events.pid.isin(anchor.index), ["pid", "ts"]]
        ev_pid, ev_ts = ev.pid.to_numpy(), ev.ts.to_numpy()

        k = 1
        while True:
            cut = _candidate_cutoffs(anchor, k, cfg.stride_years)
            # Events strictly before each candidate cutoff, counted per patient.
            n_before = pd.Series(ev_ts < cut.reindex(ev_pid).to_numpy()).groupby(
                ev_pid).sum().reindex(anchor.index).fillna(0)
            keep = n_before >= cfg.min_events
            if not keep.any():
                break
            pids = anchor.index[keep.to_numpy()]
            extra.append(pd.DataFrame({
                "pid": np.asarray(pids, np.int32),
                "cutoff": cut.loc[pids].to_numpy(),
                "split": split_of.loc[pids].to_numpy(),
                "is_real": False,
                "k": k,
            }))
            k += 1

    if extra and cfg.max_extra_per_patient is not None:
        extra = [d[d.k <= cfg.max_extra_per_patient] for d in extra]

    ex = pd.concat([real, *extra], ignore_index=True)
    ex["window_end"] = label_window_end(ex.cutoff, cfg.window_years)
    ex.insert(0, "eid", np.arange(len(ex), dtype=np.int32))
    _assert_invariants(ex, cohort, cfg)
    return ex


def _assert_invariants(ex: pd.DataFrame, cohort: pd.DataFrame,
                       cfg: ExampleConfig) -> None:
    n = len(cohort)
    assert (ex.eid.to_numpy()[:n] == cohort.pid.to_numpy()).all(), \
        "the real block must occupy eid == pid so the frozen counts still index it"
    assert ex.is_real.to_numpy()[:n].all() and not ex.is_real.to_numpy()[n:].any()
    assert ex.eid.is_unique and (ex.eid.to_numpy() == np.arange(len(ex))).all()

    anchor = pd.Series(cohort.anchor.values, index=cohort.pid.values)
    a = ex.pid.map(anchor)
    assert (ex.loc[ex.is_real, "cutoff"] == a[ex.is_real]).all(), \
        "a real example's cutoff must be the patient's own anchor"
    assert (ex.loc[~ex.is_real, "cutoff"] < a[~ex.is_real]).all(), \
        "a synthetic cutoff must be strictly earlier than the anchor"

    # The guard that actually matters: nothing outside `augment_splits` is
    # duplicated, so every evaluated patient contributes exactly one row.
    dup = ex[~ex.split.isin(cfg.active_splits)]
    assert dup.pid.is_unique, \
        f"a non-augmented split has duplicate examples: {dup.pid.duplicated().sum()}"


def config_tag(cfg: ExampleConfig) -> str:
    """Cache key covering **every** field of the config.

    An earlier version keyed on ``stride_years`` and ``min_events`` only. The
    `augment_splits=()` control arm then silently loaded the augmented table
    and reported the augmented model's score as the baseline -- the two arms
    of the comparison were the same run. Enumerating fields by hand is exactly
    how that happens, so the key is derived from the dataclass.
    """
    blob = json.dumps(asdict(cfg), sort_keys=True, default=str)
    return f"s{cfg.stride_years}_m{cfg.min_events}_{hashlib.sha1(blob.encode()).hexdigest()[:8]}"


def load_examples(root: Path | str = ".", cfg: ExampleConfig | None = None,
                  rebuild: bool = False) -> pd.DataFrame:
    cfg = cfg or ExampleConfig()
    return cached_parquet(f"examples_{config_tag(cfg)}",
                          lambda: build_examples(root, cfg), _deps(Path(root)),
                          rebuild=rebuild)


def grouped_folds(ex: pd.DataFrame, n_splits: int = 5, seed: int = 12345,
                  pids: np.ndarray | None = None,
                  contiguous: bool = True) -> list[np.ndarray]:
    """Fold assignment by **patient**, returned as row masks over ``ex``.

    Not ``KFold`` over rows. A patient with six cutoffs shares five-sixths of
    their history between consecutive examples, so splitting rows at random
    puts near-copies of the same record on both sides of the fold boundary and
    reports a validation score that is partly a memorisation score.

    Two corrections over the original, both of which mattered.

    **``pids`` restricts the universe.** The original assigned folds across all
    3,514 patients, so using ``fold != k`` as a fit mask pulled in 588 val and
    test rows -- i.e. it trained on validation. Pass the train pids.

    **``contiguous`` deals the permutation in blocks rather than round-robin.**
    Round-robin (``% n_splits``) means a K=10 partition is unrelated to a K=5
    one, and fold 0 bears no relation to anything else in the project. Dealing
    in blocks off the same seed-12345 permutation makes **K=5 fold 0 exactly
    the dev split** ``train_finetune.inner_split_pids`` produces, and makes
    K=10 folds {0,1} equal K=5 fold 0. Both properties are free and both are
    needed for a search ledger to stay comparable with fold-based work.

    Sizes stay within one patient of each other either way.
    """
    universe = np.sort(ex.pid.unique() if pids is None else np.unique(pids))
    rng = np.random.default_rng(seed)
    order = rng.permutation(universe)
    n = len(order)
    if contiguous:
        # Boundaries by ROUNDING i*n/K, not by integer division. Integer
        # division hands the remainder to fold 0, making it 559 where
        # `inner_split_pids` -- which uses round(frac*n) -- produces 558, and
        # the two would then miss coinciding by a single patient. Rounding
        # makes fold 0 exactly the dev split and keeps K=10 {0,1} == K=5 {0}.
        edges = [int(round(i * n / n_splits)) for i in range(n_splits + 1)]
        fold_id = np.empty(n, int)
        for f in range(n_splits):
            fold_id[edges[f]:edges[f + 1]] = f
    else:
        fold_id = np.arange(n) % n_splits
    assign = pd.Series(fold_id, index=order)
    fold_of = ex.pid.map(assign).to_numpy()          # NaN for pids outside `universe`
    return [np.asarray(fold_of == f) for f in range(n_splits)]
