"""Raw CSVs -> one patient table and one unified event frame.

This is the only module that knows file paths and CSV schemas. Everything
downstream reads the two cached artifacts it produces and never touches a CSV.

Memory strategy
---------------
``observations.csv`` is 222 MB against 15 GB of RAM, so chunking would add a
second code path and partial-groupby bugs for no benefit. The wins are column
pruning and dtypes:

* **``DESCRIPTION`` is never read.** It is roughly 40% of observations.csv and
  perfectly redundant with ``CODE``. A ``code -> description`` map is extracted
  once, separately, for report tables only.
* ``PATIENT`` is mapped to an ``int32`` row index immediately. Keeping a
  36-character string in a 2.1M-row frame costs ~340 MB of Python objects.
* Codes are interned to an ``int32`` ``code_id`` against a global code table.

The result is ~2.1M rows at 22 bytes each, about 48 MB resident.

Leakage guards that live here rather than only in the tests
-----------------------------------------------------------
``DEATHDATE`` is populated for 1,354 of 3,156 train/val patients and **zero**
of 358 test patients -- the organisers stripped it, so a model using it would
look excellent in training and contribute nothing at test.
``HEALTHCARE_EXPENSES`` is a *lifetime* total present in both splits, so it
encodes post-anchor spend and would silently *improve* the test score while
being unambiguous leakage. Both are refused at build time, not just asserted
about in a test file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.io import cached_parquet
from ..utils.timeutil import anchor_from_encounters, label_window_end, to_ts

__all__ = [
    "ALLOWED_PATIENT_COLS",
    "FORBIDDEN_PATIENT_COLS",
    "TABLES",
    "KIND_ORDER",
    "load_splits",
    "load_target_codes",
    "load_target_descriptions",
    "load_cohort",
    "load_events",
]

# Only these six columns of patients.csv may ever reach a feature.
ALLOWED_PATIENT_COLS = ("Id", "BIRTHDATE", "MARITAL", "RACE", "ETHNICITY", "GENDER")

FORBIDDEN_PATIENT_COLS = frozenset({
    "DEATHDATE",                                    # 1354/3156 train_val, 0/358 test
    "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE",   # lifetime totals, both splits
    "SSN", "DRIVERS", "PASSPORT", "PREFIX", "FIRST", "LAST", "SUFFIX", "MAIDEN",
    "BIRTHPLACE", "ADDRESS", "CITY", "STATE", "COUNTY", "ZIP", "LAT", "LON",
})


@dataclass(frozen=True)
class TableSpec:
    """How to read one Synthea event table.

    ``time`` is the *start/event* column and the only one used for windowing.
    ``stop`` is recorded for the encounters table alone, because the anchor
    definition needs it -- and it is read by exactly one function. 6,078
    pre-anchor training medications have a ``STOP`` after the anchor, and the
    organisers blanked those in test (4.41% null vs 1.17% pre-anchor in train),
    so any duration feature would both leak and shift.
    """
    time: str
    code: str
    kind: str
    stop: str | None = None
    extra: tuple[str, ...] = ()


# REASONCODE says what a treatment was *for*. 78% of medications carry one,
# and it buys the ancestor-credit an ontology would give -- a drug and a
# procedure aimed at the same condition increment one shared column -- from a
# provided column rather than a downloaded file.
#
# Cost columns are per-row and pre-anchor, so they are legitimate. They are
# NOT the same object as `HEALTHCARE_EXPENSES` in patients.csv, which is a
# LIFETIME total present in both splits and is therefore forbidden: a test
# patient's value is ~512x their pre-anchor claims and would raise the test
# score by encoding spend that happens after the cutoff.
TABLES: dict[str, TableSpec] = {
    "encounters":    TableSpec("START", "CODE", "ENC", stop="STOP",
                               extra=("ENCOUNTERCLASS", "REASONCODE",
                                      "TOTAL_CLAIM_COST")),
    "conditions":    TableSpec("START", "CODE", "COND"),
    "medications":   TableSpec("START", "CODE", "MED",
                               extra=("REASONCODE", "BASE_COST")),
    "procedures":    TableSpec("DATE", "CODE", "PROC",
                               extra=("REASONCODE", "BASE_COST")),
    "observations":  TableSpec("DATE", "CODE", "OBS", extra=("VALUE", "UNITS", "TYPE")),
    "immunizations": TableSpec("DATE", "CODE", "IMM", extra=("BASE_COST",)),
    "careplans":     TableSpec("START", "CODE", "CP", extra=("REASONCODE",)),
    "allergies":     TableSpec("START", "CODE", "ALG"),
    "devices":       TableSpec("START", "CODE", "DEV"),
    "imaging_studies": TableSpec("DATE", "MODALITY_CODE", "IMG"),
}

# Whichever cost column a table happens to name.
_COST_COLS = ("TOTAL_CLAIM_COST", "BASE_COST")

# Three tables carry a date with no time: conditions, careplans and allergies
# are stamped at exactly midnight for 100% of their rows (measured; every other
# kind is at 0.0007% or below). Midnight is the earliest instant of the day, so
# a diagnosis sorts BEFORE the encounter that produced it -- measured, 81.4% of
# conditions have an encounter on the same calendar day and 100% of those land
# earlier than it. See `_seq_timestamp`.
DATE_ONLY_KINDS = ("COND", "CP", "ALG")

# Deterministic intra-timestamp ordering. 97.7% of events share a timestamp
# with at least one other once date-only diagnoses are snapped onto their
# encounter (95.6% before), so this order is imposed by us, not observed -- it
# exists for reproducibility only and the model is made invariant to it by
# masking on the timestamp rather than the index.
KIND_ORDER = {"ENC": 0, "COND": 1, "PROC": 2, "MED": 3, "OBS": 4,
              "IMM": 5, "CP": 6, "ALG": 7, "DEV": 8, "IMG": 9}


def load_splits(root: Path | str = ".") -> pd.Series:
    """``Id -> split``. Asserts the documented 2791 / 365 / 358."""
    sp = pd.read_csv(Path(root) / "patient_splits.csv", dtype=str).set_index("Id")["split"]
    counts = sp.value_counts().to_dict()
    assert counts == {"train": 2791, "val": 365, "test": 358}, f"split sizes {counts}"
    return sp


def load_target_codes(root: Path | str = ".") -> list[str]:
    """The 40 codes as **strings in file order**.

    ``dtype=str`` is load-bearing: pandas would read ``444814009`` as int64 and
    every later join would silently match nothing.
    """
    codes = pd.read_csv(Path(root) / "target_conditions.csv", dtype=str)["CODE"].tolist()
    assert len(codes) == 40 == len(set(codes))
    return codes


def load_target_descriptions(root: Path | str = ".") -> dict[str, str]:
    """``code -> human-readable description``, for report tables only.

    Never joined to a feature frame: the description is a deterministic
    function of the code, so it carries no additional signal and would only
    add a text column for a model to accidentally learn from.
    """
    t = pd.read_csv(Path(root) / "target_conditions.csv", dtype=str)
    return dict(zip(t["CODE"], t["DESCRIPTION"]))


def _read_table(split_dir: Path, name: str, spec: TableSpec) -> pd.DataFrame:
    """Read one table with pruned columns and parsed timestamps."""
    cols = [spec.time, "PATIENT", spec.code, *spec.extra]
    if spec.stop:
        cols.append(spec.stop)
    df = pd.read_csv(split_dir / f"{name}.csv", usecols=cols, dtype={spec.code: str})
    df["ts"] = to_ts(df[spec.time])
    if spec.stop:
        df["stop_ts"] = to_ts(df[spec.stop])
    df["kind"] = spec.kind
    df["code"] = df[spec.code]
    return df


def _build_cohort(root: Path) -> pd.DataFrame:
    """One row per patient: split, anchor, window end, demographics, counts."""
    splits = load_splits(root)
    frames = []
    for sub in ("train_val", "test"):
        pat = pd.read_csv(root / sub / "patients.csv", dtype=str)
        leaked = FORBIDDEN_PATIENT_COLS & set(pat.columns) & set(ALLOWED_PATIENT_COLS)
        assert not leaked, f"allowlist and forbidden list overlap: {leaked}"
        frames.append(pat[list(ALLOWED_PATIENT_COLS)].assign(_src=sub))
    pat = pd.concat(frames, ignore_index=True)

    # Hard refusal, not a test-file assertion.
    bad = FORBIDDEN_PATIENT_COLS & set(pat.columns)
    assert not bad, f"forbidden patient columns reached the cohort: {sorted(bad)}"

    pat = pat.rename(columns={"Id": "patient_id"})
    pat["birthdate"] = to_ts(pat["BIRTHDATE"])
    pat["split"] = pat.patient_id.map(splits)
    assert pat.split.notna().all(), "a patient is missing from patient_splits.csv"

    # Train/val anchors are derived. Test anchors are read verbatim -- the test
    # tables are truncated, so the derived value is a strict lower bound (median
    # 58.9 days low) and the derivation path must never be reachable for them.
    # Build a clean frame rather than renaming: _read_table keeps the raw
    # string START/STOP alongside the parsed ts/stop_ts, and renaming onto
    # those names would silently hand the string columns to the groupby.
    enc = pd.concat([
        (lambda d: pd.DataFrame({"PATIENT": d["PATIENT"].values,
                                 "START": d["ts"].values,
                                 "STOP": d["stop_ts"].values}))(
            _read_table(root / sub, "encounters", TABLES["encounters"]))
        for sub in ("train_val", "test")
    ], ignore_index=True)
    derived = anchor_from_encounters(enc)

    given = pd.read_csv(root / "test_anchors.csv", dtype=str)
    given_anchor = pd.Series(to_ts(given["anchor_date"]).values, index=given["Id"])

    # `.where` rather than `np.where` so the datetime64 dtype survives -- the
    # numpy version returns an object array and would need a second parse,
    # which is exactly the rule `to_ts` exists to enforce.
    is_test = pat.split.eq("test")
    pat["anchor"] = pat.patient_id.map(derived).where(
        ~is_test, pat.patient_id.map(given_anchor))
    pat["anchor_source"] = np.where(is_test, "provided", "derived")
    assert pat.loc[is_test, "anchor_source"].eq("provided").all()
    assert pat.anchor.notna().all(), "patient with no encounters"

    pat["window_end"] = label_window_end(pat["anchor"])
    pat.loc[is_test, "window_end"] = pd.NaT           # outcomes withheld
    pat["age_at_anchor"] = (pat.anchor - pat.birthdate).dt.days / 365.25
    pat["pid"] = np.arange(len(pat), dtype=np.int32)

    out = pat[["pid", "patient_id", "split", "birthdate", "anchor", "window_end",
               "anchor_source", "age_at_anchor", "GENDER", "RACE", "ETHNICITY",
               "MARITAL"]].rename(columns=str.lower)
    return out.sort_values("pid").reset_index(drop=True)


def _seq_timestamp(ev: pd.DataFrame) -> pd.Series:
    """The timestamp used for ORDERING and time-encoding. Not for labels.

    A date-only event is snapped forward to the first encounter on its own
    calendar day, making the diagnosis simultaneous with the visit that
    produced it rather than a fraction of a day before it. Simultaneous is the
    right relation, not "just after": under timestamp-masking, events at the
    same instant attend to each other, which is what "this was diagnosed at
    this visit" means. The 18.6% of conditions with no same-day encounter keep
    their midnight stamp.

    **This column must never reach `labels.py`.** Both label boundaries sit at
    midnight, and 119 first-diagnoses land exactly on ``anchor + 5y``. Snapping
    one of those forward to an afternoon encounter would push it past the
    closed right edge and silently delete a positive. Labels read raw ``ts``;
    a grep test enforces it.

    Snapping cannot move an event across the anchor, because it never leaves
    the calendar day it started on and the anchor is itself midnight. Asserted
    below rather than argued.
    """
    enc = ev.loc[ev.kind == "ENC", ["pid", "ts"]]
    first_enc = (enc.assign(d=enc.ts.dt.normalize())
                    .groupby(["pid", "d"], observed=True)["ts"].min())
    key = pd.MultiIndex.from_arrays([ev.pid.values, ev.ts.dt.normalize().values])
    snapped = pd.Series(first_enc.reindex(key).to_numpy(), index=ev.index)
    is_date_only = ev.kind.isin(DATE_ONLY_KINDS).to_numpy()
    return ev.ts.where(~is_date_only | snapped.isna(), snapped)


def _build_events(root: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    """Unified long event frame, ~2.1M rows, ~48 MB."""
    pid_of = pd.Series(cohort.pid.values, index=cohort.patient_id.values)
    anchor_of = pd.Series(cohort.anchor.values, index=cohort.pid.values)

    parts = []
    for name, spec in TABLES.items():
        for sub in ("train_val", "test"):
            df = _read_table(root / sub, name, spec)
            df["pid"] = df.PATIENT.map(pid_of).astype("Int32")
            df = df[df.pid.notna()].copy()
            df["pid"] = df.pid.astype(np.int32)
            df["token"] = df.kind + "_" + df.code
            num = (df["TYPE"].eq("numeric") if "TYPE" in df.columns
                   else pd.Series(False, index=df.index))
            val = (pd.to_numeric(df["VALUE"], errors="coerce") if "VALUE" in df.columns
                   else pd.Series(np.nan, index=df.index))
            reason = (df["REASONCODE"].astype("string")
                      if "REASONCODE" in df.columns
                      else pd.Series(pd.NA, index=df.index, dtype="string"))
            cost = next((pd.to_numeric(df[c], errors="coerce")
                         for c in _COST_COLS if c in df.columns),
                        pd.Series(np.nan, index=df.index))
            parts.append(pd.DataFrame({
                "pid": df.pid.values,
                "ts": df.ts.values,
                "kind": df.kind.values,
                "token": df.token.values,
                "value": np.where(num.values, val.values, np.nan).astype(np.float32),
                "is_numeric": num.values,
                # Synthea writes REASONCODE as a float-looking string when the
                # column is sparse ("444814009.0"), which would never match a
                # target code. Normalise through Int64 before stringifying.
                "reason": np.asarray(
                    pd.array(pd.to_numeric(reason, errors="coerce"),
                             dtype="Int64").astype("string"), dtype=object),
                "cost": cost.to_numpy(np.float32),
            }))
            del df
    ev = pd.concat(parts, ignore_index=True)
    del parts

    ev["anchor"] = ev.pid.map(anchor_of)
    ev["is_pre_anchor"] = ev.ts < ev.anchor          # raw ts, the ONE place written

    # Every date-only kind must actually be date-only, or the snap is wrong.
    do = ev.kind.isin(DATE_ONLY_KINDS)
    assert (ev.loc[do, "ts"] == ev.loc[do, "ts"].dt.normalize()).all(), \
        "a DATE_ONLY_KINDS row carries a time of day"

    ev["ts_seq"] = _seq_timestamp(ev)
    assert (ev.ts_seq >= ev.ts).all(), "snapping moved an event backwards"
    assert (ev.ts_seq.dt.normalize() == ev.ts.dt.normalize()).all(), \
        "snapping crossed a calendar day"
    assert ((ev.ts_seq < ev.anchor) == ev.is_pre_anchor).all(), \
        "snapping moved an event across the anchor"
    ev = ev.drop(columns=["anchor"])

    ev["kind"] = ev.kind.astype("category")
    ev["token"] = ev.token.astype("category")
    ev["reason"] = ev.reason.astype("category")
    ev["ko"] = ev.kind.map(KIND_ORDER).astype(np.int8)
    return ev.sort_values(["pid", "ts_seq", "ko", "token"]).reset_index(drop=True)


def _deps(root: Path) -> list[Path]:
    files = [root / "patient_splits.csv", root / "target_conditions.csv",
             root / "test_anchors.csv"]
    for sub in ("train_val", "test"):
        files += [root / sub / f"{n}.csv" for n in TABLES]
    return files


def load_cohort(root: Path | str = ".", rebuild: bool = False) -> pd.DataFrame:
    root = Path(root)
    return cached_parquet("cohort", lambda: _build_cohort(root), _deps(root),
                          rebuild=rebuild)


def load_events(root: Path | str = ".", rebuild: bool = False) -> pd.DataFrame:
    root = Path(root)
    cohort = load_cohort(root, rebuild=rebuild)
    return cached_parquet("events", lambda: _build_events(root, cohort), _deps(root),
                          rebuild=rebuild)
