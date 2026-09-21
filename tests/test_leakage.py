"""Leakage tests.

The point of these is not code hygiene. Leakage across the anchor is invisible
by inspection -- it produces excellent validation numbers and a garbage test
score -- so every rule that keeps the pipeline honest is written here as an
assertion that fails loudly.

Several tests carry a **non-vacuity check**: a leakage test that passes because
it examined nothing is worse than no test. Where one exists it is called out.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cohort import (  # noqa: E402
    FORBIDDEN_PATIENT_COLS, TABLES, load_cohort, load_splits, load_target_codes,
)
from src.data.labels import GOLDEN, at_risk_mask, load_labels  # noqa: E402
from src.utils.timeutil import to_ts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cohort():
    return load_cohort(ROOT)


@pytest.fixture(scope="module")
def labels():
    return load_labels(ROOT)


@pytest.fixture(scope="module")
def events():
    from src.data.cohort import load_events
    return load_events(ROOT)


# ---------------------------------------------------------------- T1
def test_split_partition(cohort):
    """Splits are the documented sizes, disjoint, and match the patient files."""
    sp = load_splits(ROOT)
    assert sp.value_counts().to_dict() == {"train": 2791, "val": 365, "test": 358}

    tv = set(pd.read_csv(ROOT / "train_val" / "patients.csv", dtype=str)["Id"])
    te = set(pd.read_csv(ROOT / "test" / "patients.csv", dtype=str)["Id"])
    anch = set(pd.read_csv(ROOT / "test_anchors.csv", dtype=str)["Id"])

    train = set(sp[sp == "train"].index)
    val = set(sp[sp == "val"].index)
    test = set(sp[sp == "test"].index)

    assert train & val == set() and train & test == set() and val & test == set()
    assert tv == train | val
    assert te == test == anch


# ---------------------------------------------------------------- T2
def test_anchor_definition(cohort):
    """anchor == normalize(max over rows of (STOP else START)) - 5 calendar years.

    Recomputed here from the raw CSV rather than reusing the pipeline helper,
    so this is an independent check and not a tautology.
    """
    enc = pd.read_csv(ROOT / "train_val" / "encounters.csv",
                      usecols=["START", "STOP", "PATIENT"])
    start, stop = to_ts(enc["START"]), to_ts(enc["STOP"])
    last = stop.where(stop.notna(), start).groupby(enc["PATIENT"]).max()
    expect = (last - pd.DateOffset(years=5)).dt.normalize()

    tv = cohort[cohort.split != "test"].set_index("patient_id")["anchor"]
    assert len(tv) == 3156
    pd.testing.assert_series_equal(
        tv.sort_index(), expect.reindex(tv.index).sort_index(),
        check_names=False, check_freq=False)

    assert (cohort.anchor.dt.normalize() == cohort.anchor).all(), "anchor not at midnight"


# ---------------------------------------------------------------- T3
def test_provided_test_tables_have_no_post_anchor_rows():
    """The organisers' own truncation, measured directly.

    Every time column of every provided test table, against the given anchor.
    This pins one side of the boundary: their cut is not looser than ours.
    """
    anchors = pd.read_csv(ROOT / "test_anchors.csv", dtype=str)
    a = pd.Series(to_ts(anchors["anchor_date"]).values, index=anchors["Id"])

    checked = 0
    for name, spec in TABLES.items():
        df = pd.read_csv(ROOT / "test" / f"{name}.csv", dtype=str)
        anc = df["PATIENT"].map(a)
        for col in {spec.time, spec.stop} - {None}:
            ts = to_ts(df[col])
            late = int(((ts >= anc) & ts.notna()).sum())
            assert late == 0, f"{name}.{col}: {late} rows at or after the anchor"
            checked += 1
    assert checked >= 10, "non-vacuity: too few columns examined"


# ---------------------------------------------------------------- T4
def test_our_truncation_is_a_noop_on_test(cohort, events):
    """Our `ts < anchor` rule removes nothing from the provided test tables.

    T3 pins that their cut is not looser than ours; this pins that ours is not
    stricter than theirs. Together they bracket the boundary exactly.
    """
    test_pids = set(cohort.loc[cohort.split == "test", "pid"])
    et = events[events.pid.isin(test_pids)]
    assert len(et) > 100_000, "non-vacuity: test event frame is suspiciously small"
    assert int((~et.is_pre_anchor).sum()) == 0


# ---------------------------------------------------------------- T5
def test_derived_test_anchor_is_a_strict_lower_bound(cohort):
    """The expected relation is INEQUALITY, not equality. Do not "fix" this.

    The test tables are truncated at the anchor, so the true last encounter is
    unrecoverable and any derived anchor is necessarily early -- measured
    minimum gap 0.0995 days, median 58.9. The complementary invariant is that
    the derivation path is never reachable for test patients.
    """
    enc = pd.read_csv(ROOT / "test" / "encounters.csv",
                      usecols=["START", "STOP", "PATIENT"])
    start, stop = to_ts(enc["START"]), to_ts(enc["STOP"])
    last = stop.where(stop.notna(), start).groupby(enc["PATIENT"]).max()
    derived = (last - pd.DateOffset(years=5)).dt.normalize()

    te = cohort[cohort.split == "test"].set_index("patient_id")
    d = derived.reindex(te.index)
    assert (d < te["anchor"]).all(), "a derived test anchor met or exceeded the given one"
    assert te["anchor_source"].eq("provided").all()


# ---------------------------------------------------------------- T7
def test_label_window_partition(labels):
    """Exactly one of {prevalent, incident} per observed (patient, condition)."""
    y, prev = labels["y"], labels["prevalent"]
    assert int((y & prev).sum()) == 0
    assert int(prev.sum()) > 0, "non-vacuity: no prevalent pairs found"
    assert int(y.sum()) > 0, "non-vacuity: no incident pairs found"


# ---------------------------------------------------------------- T8
def test_prevalent_implies_negative(labels):
    """A patient who already had condition c cannot be an incident case for c."""
    y, prev, split = labels["y"], labels["prevalent"], labels["split"]
    assert not y[prev == 1].any()
    assert int(prev[split == "train"].sum()) == 10583, "non-vacuity guard"


# ---------------------------------------------------------------- T9
def test_labels_golden_counts(labels):
    """Frozen totals. Catches any silent change to the label definition."""
    y, prev, split = labels["y"], labels["prevalent"], labels["split"]
    got = {
        "y_train": int(y[split == "train"].sum()),
        "y_val": int(y[split == "val"].sum()),
        "prevalent_train": int(prev[split == "train"].sum()),
        "prevalent_val": int(prev[split == "val"].sum()),
    }
    assert got == GOLDEN
    assert int(y[split == "test"].sum()) == 0, "test outcomes are withheld"


def test_first_diagnosis_is_over_the_whole_record(labels):
    """The highest-stakes line in the codebase, tested by its consequence.

    Taking the minimum *inside* the window instead of over the whole record
    would reclassify every recurrence as a new diagnosis. Measured here: 1,220
    (patient, condition) pairs across 1,065 patients have a pre-anchor first
    occurrence *and* a later occurrence inside the label window -- a 23.4%
    inflation on top of the 5,214 true positives.

    Every one of those pairs must be prevalent and must not be a positive.
    """
    from src.data.cohort import TABLES, _read_table

    cohort = load_cohort(ROOT)
    codes = load_target_codes(ROOT)
    cidx = {c: j for j, c in enumerate(codes)}
    pid_of = pd.Series(cohort.pid.values, index=cohort.patient_id.values)

    con = pd.concat([_read_table(ROOT / s, "conditions", TABLES["conditions"])
                     for s in ("train_val", "test")], ignore_index=True)
    con["pid"] = con.PATIENT.map(pid_of)
    con = con[con.pid.notna() & con.code.isin(cidx)].copy()
    con["pid"] = con.pid.astype(int)
    con["a"] = con.pid.map(pd.Series(cohort.anchor.values, index=cohort.pid.values))
    con["w"] = con.pid.map(pd.Series(cohort.window_end.values, index=cohort.pid.values))

    first = con.groupby(["pid", "code"], observed=True)["ts"].min().rename("first")
    con = con.join(first, on=["pid", "code"])

    trap = con[(con["first"] < con.a) & con.w.notna()
               & (con.ts >= con.a) & (con.ts <= con.w)]
    pairs = trap.groupby(["pid", "code"]).size().reset_index()
    assert len(pairs) > 1000, f"non-vacuity: only {len(pairs)} trap pairs found"

    rows = pairs.pid.values
    cols = pairs.code.map(cidx).values
    assert (labels["prevalent"][rows, cols] == 1).all(), "recurrence not marked prevalent"
    assert (labels["y"][rows, cols] == 0).all(), "a recurrence was counted as incident"


# ---------------------------------------------------------------- T10
def test_no_forbidden_patient_columns(cohort):
    """Plus a positive control documenting *why* DEATHDATE is banned."""
    assert not (FORBIDDEN_PATIENT_COLS & {c.upper() for c in cohort.columns})

    tv = pd.read_csv(ROOT / "train_val" / "patients.csv", usecols=["DEATHDATE"])
    te = pd.read_csv(ROOT / "test" / "patients.csv", usecols=["DEATHDATE"])
    assert int(te["DEATHDATE"].notna().sum()) == 0, "test DEATHDATE is populated?"
    assert int(tv["DEATHDATE"].notna().sum()) == 1354, "train_val DEATHDATE count changed"


# ---------------------------------------------------------------- T11
def test_no_stop_derived_features():
    """`STOP` is declared on exactly one table, and only the anchor reads it."""
    with_stop = [n for n, s in TABLES.items() if s.stop is not None]
    assert with_stop == ["encounters"], f"STOP declared on {with_stop}"

    # The evidence: post-anchor STOPs exist in train and were blanked in test.
    cohort = load_cohort(ROOT)
    a = cohort.set_index("patient_id")["anchor"]
    med_tv = pd.read_csv(ROOT / "train_val" / "medications.csv",
                         usecols=["START", "STOP", "PATIENT"])
    pre = to_ts(med_tv["START"]) < med_tv["PATIENT"].map(a)
    late = (to_ts(med_tv["STOP"]) > med_tv["PATIENT"].map(a)) & pre
    assert int(late.sum()) > 5000, "expected thousands of post-anchor STOPs in train"

    med_te = pd.read_csv(ROOT / "test" / "medications.csv", usecols=["STOP"])
    assert med_te["STOP"].isna().mean() > 3 * med_tv.loc[pre, "STOP"].isna().mean()


# ---------------------------------------------------------------- at-risk
def test_at_risk_mask_is_consistent_with_labels(labels):
    """Masked pairs must never hide a true positive. Must be exactly zero."""
    ar = at_risk_mask(labels)
    hidden = int(((~ar) & (labels["y"] == 1)).sum())
    assert hidden == 0, f"{hidden} positives would be masked away"
    assert int((~ar).sum()) > 0, "non-vacuity: mask fires nowhere"


# ---------------------------------------------------------------- discipline
def test_only_timeutil_parses_timestamps():
    """One parser. A stray `pd.to_datetime` elsewhere reintroduces the mixed
    date-only / ISO-Z bug class that `to_ts` exists to eliminate."""
    hits = subprocess.run(
        ["grep", "-rn", "--include=*.py", "to_datetime", str(ROOT / "src")],
        capture_output=True, text=True).stdout.strip().splitlines()
    offenders = [h for h in hits if "utils/timeutil.py" not in h.replace("\\", "/")]
    assert not offenders, "pd.to_datetime outside timeutil:\n" + "\n".join(offenders)


# ---------------------------------------------------------------- T20 -------
def test_date_only_events_are_snapped_to_their_encounter(events):
    """A12. Conditions, careplans and allergies carry a date with no time, so
    at midnight they sort BEFORE the encounter that produced them. Measured
    before the fix: 81.4% of conditions had a same-day encounter and 100% of
    those landed earlier than it.

    After snapping they must be *simultaneous* with it -- not merely closer.
    Simultaneous is what "diagnosed at this visit" means under a mask that
    lets events at one instant attend to each other.
    """
    e = events[events.is_pre_anchor]
    cond, enc = e[e.kind == "COND"], e[e.kind == "ENC"]
    first_enc = (enc.assign(d=enc.ts.dt.normalize())
                    .groupby(["pid", "d"], observed=True).ts_seq.min())
    j = cond.assign(d=cond.ts.dt.normalize()).join(first_enc.rename("e"), on=["pid", "d"])
    matched = j.e.notna()

    assert matched.mean() > 0.80, f"only {matched.mean():.1%} matched; expected ~81.4%"
    assert (j.loc[matched, "ts_seq"] == j.loc[matched, "e"]).all(), \
        "a same-day condition is not simultaneous with its encounter"
    # Unmatched ones must be left alone, not nudged.
    assert (j.loc[~matched, "ts_seq"] == j.loc[~matched, "ts"]).all()


def test_snapping_cannot_move_an_event_across_the_anchor(events, cohort):
    """The snap only ever moves an event later, and both the anchor and a
    date-only stamp sit at midnight -- so a snap that crossed the anchor would
    mean it left its own calendar day. Cheap to assert, catastrophic to miss.
    """
    assert (events.ts_seq >= events.ts).all()
    assert (events.ts_seq.dt.normalize() == events.ts.dt.normalize()).all()

    anchor = pd.Series(cohort.anchor.values, index=cohort.pid.values)
    recomputed = events.ts_seq < events.pid.map(anchor)
    n_flipped = int((recomputed != events.is_pre_anchor).sum())
    assert n_flipped == 0, (
        f"{n_flipped} events changed side of the anchor under ts_seq; "
        "is_pre_anchor is derived from raw ts and the two must agree")


def test_labels_never_read_the_ordering_timestamp():
    """`ts_seq` must not reach `labels.py`.

    Both label boundaries are midnight and 119 first-diagnoses land exactly on
    `anchor + 5y`. Snapping one of those forward to an afternoon encounter
    pushes it past the closed right edge and deletes a true positive in
    silence -- no test would fail, the golden counts would simply be re-frozen
    wrong. So the guard is on the source, not the output.
    """
    hits = subprocess.run(
        ["grep", "-rn", "--include=*.py", "ts_seq", str(ROOT / "src")],
        capture_output=True, text=True).stdout.strip().splitlines()
    allowed = ("data/cohort.py", "data/sequences.py")

    def is_comment(line: str) -> bool:
        # `grep -n` emits `path:lineno:content`, and on Windows the path
        # itself contains a colon ("C:\..."), so a plain split(":", 2) hands
        # back the line number instead of the content. Anchor on `:<digits>:`.
        m = re.match(r"^.*?:\d+:(.*)$", line)
        return bool(m) and m.group(1).lstrip().startswith("#")

    # Prose mentioning ts_seq is fine -- labels.py explains why it must not
    # use it, and that explanation should not trip its own guard.
    offenders = [h for h in hits
                 if not any(a in h.replace("\\", "/") for a in allowed)
                 and not is_comment(h)]
    assert not offenders, ("ts_seq read outside cohort/sequences:\n"
                           + "\n".join(offenders))
