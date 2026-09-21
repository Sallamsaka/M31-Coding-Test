"""Incident-condition labels, and the at-risk mask that goes with them.

The whole module is four lines of logic and they have to be exactly these::

    first     = conditions.groupby(["pid","code"])["ts"].min()   # FIRST-EVER
    y         = (first >= anchor) & (first <= anchor + 5y)
    prevalent = (first <  anchor)

The ``groupby.min()`` runs over the patient's **entire record**, not over the
label window. This is the single most consequential line in the codebase.
Synthea re-records recurring conditions, so computing the minimum *inside* the
window would turn every recurrence into a false positive -- our patient
``8b0484cd`` has "normal pregnancy" in 2011, 2015 and 2020, and only the 2011
occurrence makes her a prevalent case rather than an incident one. Getting it
wrong inflates training positives by roughly 48%.

Window edges, both settled by measurement rather than taste:

* **Left edge closed.** ``anchor <= first`` makes features and labels an exact
  partition of the timeline with no gap. Affects 2 pairs.
* **Right edge closed.** 119 first-diagnoses land exactly on ``anchor + 5y``
  and zero land beyond it. That is structural: ``anchor + 5y`` is by
  construction the day of the last encounter, and conditions carry a date with
  no time, so a diagnosis at that final visit lands precisely on the boundary.
  An exclusive edge silently discards 2% of all positives.

Prevalent patients are kept in the cohort with ``y = 0``. They are *not* the
same as negatives: they cannot be incident cases at all, which is why
:func:`at_risk_mask` exists and why the evaluation reports both denominators.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.io import cached_npz
from .cohort import TABLES, _read_table, load_cohort, load_target_codes, _deps
from .examples import ExampleConfig, config_tag, load_examples

__all__ = ["GOLDEN", "load_labels", "at_risk_mask"]

# Frozen so a future refactor cannot silently change the label definition.
# Reproduced independently twice before being written down.
GOLDEN = {
    "y_train": 5214, "y_val": 685,
    "prevalent_train": 10583, "prevalent_val": 1327,
}


def _build_labels(root: Path, cfg: ExampleConfig) -> dict[str, np.ndarray]:
    cohort = load_cohort(root)
    ex = load_examples(root, cfg)
    codes = load_target_codes(root)
    cidx = {c: j for j, c in enumerate(codes)}
    pid_of = pd.Series(cohort.pid.values, index=cohort.patient_id.values)

    # Conditions from both splits. Test conditions are truncated at the anchor,
    # so test patients can only ever be prevalent -- their outcomes are withheld
    # and their y row stays all-zero by construction.
    con = pd.concat([
        _read_table(root / sub, "conditions", TABLES["conditions"])
        for sub in ("train_val", "test")
    ], ignore_index=True)
    con["pid"] = con.PATIENT.map(pid_of).astype("Int32")
    con = con[con.pid.notna() & con.code.isin(cidx)].copy()
    con["pid"] = con.pid.astype(np.int32)

    # FIRST-EVER, over the whole record. Not restricted to any window, and a
    # patient-level quantity -- so it is computed once and compared against
    # every one of that patient's cutoffs. Raw `ts`, never `ts_seq`: both
    # boundaries are midnight and snapping would push the 119 diagnoses that
    # land exactly on the right edge past it.
    first = con.groupby(["pid", "code"], observed=True)["ts"].min().reset_index()
    first["j"] = first.code.map(cidx).astype(int)

    # One row per (example, first-diagnosis). Many-to-many on pid: a patient
    # with 6 cutoffs and 4 target diagnoses contributes 24 rows.
    m = ex[["eid", "pid", "cutoff", "window_end"]].merge(first, on="pid", how="inner")

    n = len(ex)
    y = np.zeros((n, 40), np.int8)
    prev = np.zeros((n, 40), np.int8)

    is_prev = (m.ts < m.cutoff).to_numpy()
    in_win = (~is_prev) & (m.ts <= m.window_end).to_numpy()
    prev[m.eid.to_numpy()[is_prev], m.j.to_numpy()[is_prev]] = 1
    y[m.eid.to_numpy()[in_win], m.j.to_numpy()[in_win]] = 1

    # An example cannot be both for the same condition.
    assert not (y & prev).any(), "an (example, condition) pair is both prevalent and incident"

    # Nothing may fall past the right edge -- but only for REAL examples,
    # where `window_end` is the day of the last encounter and so is the end of
    # the record. A synthetic cutoff's window closes mid-record on purpose,
    # and a diagnosis after it is an ordinary negative, not a bug.
    real = m.eid.to_numpy() < len(cohort)
    past = (~is_prev) & (m.ts > m.window_end).to_numpy() & real
    assert not past.any(), f"{int(past.sum())} first-diagnoses fall beyond a real anchor+5y"

    return {
        "y": y, "prevalent": prev,
        "eid": ex.eid.to_numpy(np.int32),
        "pid": ex.pid.to_numpy(np.int32),
        "split": ex.split.to_numpy().astype("U5"),
        "is_real": ex.is_real.to_numpy(),
        "codes": np.asarray(codes, dtype="U12"),
    }


def load_labels(root: Path | str = ".", cfg: ExampleConfig | None = None,
                rebuild: bool = False) -> dict[str, np.ndarray]:
    """Cached label arrays, with the frozen counts verified on every load.

    The golden counts are checked on the **real** examples only. Those are the
    first ``len(cohort)`` rows by construction, so augmentation cannot move
    them, and a drift in the augmented block cannot hide behind a drift in the
    real one.
    """
    root = Path(root)
    cfg = cfg or ExampleConfig()
    tag = f"labels_{config_tag(cfg)}"
    d = cached_npz(tag, lambda: _build_labels(root, cfg), _deps(root), rebuild=rebuild)

    y, prev, split, real = d["y"], d["prevalent"], d["split"], d["is_real"]
    got = {
        "y_train": int(y[real & (split == "train")].sum()),
        "y_val": int(y[real & (split == "val")].sum()),
        "prevalent_train": int(prev[real & (split == "train")].sum()),
        "prevalent_val": int(prev[real & (split == "val")].sum()),
    }
    assert got == GOLDEN, f"label counts drifted: got {got}, expected {GOLDEN}"
    assert y[split == "test"].sum() == 0, "test outcomes are withheld; y must be all-zero"
    assert (~real | np.isin(split, ("train", "val", "test"))).all()

    # Validation may be augmented ONLY when the config says so explicitly,
    # which exists for one purpose: the train/eval x real/synthetic diagnostic
    # that asks whether synthetic examples are broken or merely different.
    # The guard that actually protects the result lives at the point of model
    # selection -- `run_baseline` and `train_finetune` both refuse to score a
    # non-real validation row, unconditionally. This one only catches
    # augmenting val by accident.
    val_aug = bool((~real[split == "val"]).any())
    assert val_aug == ("val" in cfg.active_splits), (
        f"validation augmentation ({val_aug}) disagrees with the config "
        f"({cfg.active_splits}) -- this is never what you want by accident")
    return d


def at_risk_mask(labels: dict[str, np.ndarray]) -> np.ndarray:
    """``True`` where the patient could still become an incident case.

    Used three ways, and they are deliberately separate decisions:

    1. **Training** -- drop not-at-risk rows per label, so capacity goes to the
       hard problem. Raises the mean positive rate 0.0467 -> 0.0575, though the
       gain is concentrated: 2.02x for obesity, 1.04x at the median.
    2. **Inference** -- force masked pairs to the bottom of the ranking. The
       rule is deterministic, so this is strictly better than hoping the model
       learns it.
    3. **Reporting** -- ``AUROC_at-risk`` alongside the grader-consistent
       ``AUROC_all``. Including not-at-risk patients as negatives inflates
       AUROC by ``(prevalent share of negatives) x (1 - A_at_risk)``.

    Note for (1) and (3): macro-AP rises *mechanically* by ``N/(N-a)`` when
    rows are dropped, since the positives are unchanged while the denominator
    shrinks -- up to 1.35x for viral sinusitis. Before/after macro-AP is
    therefore uninterpretable; report AP/prevalence or fix the population.
    """
    return labels["prevalent"] == 0
