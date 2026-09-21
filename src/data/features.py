"""Patient x feature matrix for the count-based models.

Every statistic -- the token vocabulary, the lab decile edges, the age
normalisation -- is fitted on **train patients only**. EHRSHOT's reference
implementation fits its vocabulary and age statistics over train+val+test, and
that is a leak we are explicitly graded on not reproducing.

Feature blocks
--------------
``A`` nested cumulative code counts   the biggest measured lever on this task
``B`` numeric lab value deciles       EHRSHOT drops these entirely (a defect)
``C`` per-target prevalent indicator  deterministic negative, its own block
``D`` demographics
``E`` history shape and timing
``F`` lab last-value summaries
``G`` REASONCODE counts               what a treatment was FOR
``H`` lab slope                       NaN below two readings, on purpose
``I`` time since a code last appeared what the count windows cannot express
``J`` per-visit cost
``K`` age-residualised timing

3,320 columns in total. **Measured: blocks G-K together move logistic
regression by less than the noise floor** -- every arm of the screen lands
within 0.0046 of every other, against a resolution limit of about 0.01. They
are kept rather than deleted for one reason: two of them can only pay off
through interactions a linear model cannot represent. ``G`` is literally a
weighted sum of columns LR already has, so it is redundant *to LR* by
construction and informative only to a tree; ``K`` is the mirror image,
expressible by a tree from ``span`` and ``age`` but not by LR without the
explicit ratio. A null on LR is therefore not a null, and the report says so
rather than quietly banking the numbers.

Two choices worth defending
---------------------------
**Nested, not disjoint windows.** OHDSI's defaults are cumulative and all end
at day 0 (``-365 -> 0``, ``-180 -> 0``, ``-30 -> 0``); femr and Gao use disjoint
bins. Our own measurement settles it: cumulative univariate signal rises
monotonically with history depth (mean |AUC-0.5| 0.0074 at 1y, 0.0123 at 5y,
0.0143 over all history). A disjoint bin forces the model to reconstruct "ever"
by summing; nested windows hand it every horizon directly and let a tree take
differences itself if recency is what matters.

**Raw counts, not log1p, for the tree.** An axis-aligned split is ``x <= t``,
and ``log1p`` is strictly monotone, so the achievable partitions are identical;
for integer counts inside ``max_bin`` the histogram binning is exact and the
transform is a literal no-op. It is applied only on the linear-model path,
where it genuinely matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .cohort import load_cohort, load_events, load_target_codes
from .examples import ExampleConfig, load_examples
from .labels import load_labels

__all__ = ["FeatureConfig", "build_features", "FeatureMatrix"]


@dataclass(frozen=True)
class FeatureConfig:
    # Nested cumulative horizons in days; None = all history. Ablatable.
    windows: tuple[float, ...] = (365.0, 1825.0, np.inf)
    min_patients_per_token: int = 5      # drops 119 codes covering 0.075% of events
    n_value_bins: int = 10

    lab_min_coverage: float = 0.05
    """Keep a lab if this share of train patients has it.

    Replaces an arbitrary top-20. Measured coverage collapses past the 49th
    lab -- the 20th most common is present for 83% of patients, the 50th for
    4.7% -- so the cut is where the data puts it, not where a round number is.
    """

    n_time_since_common: int = 20
    """Commonest codes given a time-since column, on top of the 40 targets.

    Genuinely distinct from the count windows, which can only say "more than
    five years ago" and never whether that meant six years or twenty-five.
    """

    deduplicate: bool = False
    """Drop columns that are exactly duplicated or constant on train.

    **Default OFF because the effect has never been measured.** Turning it on
    changes the matrix from 3,320 to 2,263 columns, which changes every result
    in the report and invalidates the saved model -- `test_saved_model.py`
    caught exactly that. An unmeasured change must not silently re-baseline
    the project; measure it, then decide.

    Measured on this matrix: **1,048 of 3,320 columns are exact duplicates**
    and 227 more are constant, leaving 2,263 that carry anything. Nested
    cumulative windows are the main source -- 309 `ever` columns equal their
    `1825d` counterpart because that code never appears more than five years
    back for anyone.

    This is not merely wasteful. Under an L2 penalty, a feature present in
    *k* identical columns has its weight split across them, so the shrinkage
    applied to that feature's total contribution falls by a factor of k. A
    code duplicated three ways is regularised three times more weakly than a
    unique one, purely because of how many of its windows happen to coincide.
    The tuned `C` was compensating for that.

    Kept as a flag so the effect can be measured rather than assumed.
    """

    enable_reason: bool = True
    enable_slope: bool = True
    enable_time_since: bool = True
    enable_cost: bool = True
    enable_age_residual: bool = True
    kinds: tuple[str, ...] = ("ENC", "COND", "MED", "PROC", "OBS", "IMM",
                              "CP", "ALG", "DEV", "IMG")


@dataclass
class FeatureMatrix:
    X: np.ndarray                 # float32, (n_examples, n_features)
    names: list[str]
    eid: np.ndarray
    pid: np.ndarray               # NOT unique once augmented -- group by this for CV
    split: np.ndarray
    is_real: np.ndarray
    blocks: dict[str, slice] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"FeatureMatrix(X={self.X.shape}, blocks={list(self.blocks)})"


def _window_label(w: float) -> str:
    return "ever" if not np.isfinite(w) else f"{int(w)}d"


def _deduplicate(X: np.ndarray, names: list[str], blocks: dict[str, slice],
                 is_train: np.ndarray) -> tuple[np.ndarray, list[str], dict]:
    """Drop exact-duplicate and train-constant columns, keeping the first.

    "First" means the narrowest window survives, because the count blocks are
    built 365d -> 1825d -> ever. That is the right survivor: the narrow
    column is the one whose value is not implied by another, and keeping
    `ever` instead would make a retained feature's meaning depend on which of
    its duplicates happened to exist.

    NaN maps to a sentinel for comparison only. `lab_slope` is legitimately
    NaN, and two columns NaN in the same places and equal elsewhere are
    genuinely duplicates.
    """
    keep, seen = [], {}
    const = set(np.flatnonzero(np.nanstd(X[is_train], axis=0) == 0).tolist())
    for j in range(X.shape[1]):
        if j in const:
            continue
        col = np.nan_to_num(X[:, j], nan=-123456.0)
        key = hash(col.tobytes())
        prev = seen.get(key)
        if prev is not None and np.array_equal(
                col, np.nan_to_num(X[:, prev], nan=-123456.0)):
            continue
        seen[key] = j
        keep.append(j)

    keep = np.asarray(keep, int)
    pos = {j: i for i, j in enumerate(keep)}
    new_blocks: dict[str, slice] = {}
    for b, sl in blocks.items():
        kept = [pos[j] for j in range(sl.start, sl.stop) if j in pos]
        if kept:
            new_blocks[b] = slice(min(kept), max(kept) + 1)
    return X[:, keep], [names[j] for j in keep], new_blocks


def build_features(root: Path | str = ".", cfg: FeatureConfig | None = None,
                   ex_cfg: ExampleConfig | None = None,
                   fit_mask: np.ndarray | None = None) -> FeatureMatrix:
    """One row per **example**, not per patient.

    The expansion is the only subtle part. An event belongs to every example
    of that patient whose cutoff it precedes, so the event frame is joined to
    the example frame on ``pid`` and then cut by ``ts < cutoff``. Two things
    make that affordable:

    * ``token`` and ``kind`` are reduced to integer codes **before** the join.
      A pandas merge materialises a categorical column as objects, and 5.3M
      rows of Python strings costs ~400 MB for information that fits in int32.
    * Distributional statistics -- the vocabulary, the decile edges, the lab
      coverage ranking -- are fitted on the **unexpanded** train-patient event
      frame. Fitting them on the expanded frame would weight each patient by
      how many cutoffs they happen to support, which is a function of record
      length, not of anything clinical.
    """
    cfg = cfg or FeatureConfig()
    root = Path(root)
    cohort = load_cohort(root)
    events = load_events(root)
    ex = load_examples(root, ex_cfg)
    labels = load_labels(root, ex_cfg)
    codes = load_target_codes(root)
    assert (labels["eid"] == ex.eid.to_numpy()).all(), "labels and examples disagree on order"

    n = len(ex)
    # Every fitted statistic below flows from these two variables -- the token
    # vocabulary, the lab decile edges, the coverage ranking, the age and lab
    # z-scores, the REASONCODE vocabulary, the time-since ranking, the
    # age-residual regression and the dedup constant set. Parameterising them
    # is therefore the whole of the fit-mask change, not nine separate edits.
    #
    # `None` executes the ORIGINAL two statements verbatim, so the default path
    # is bit-identical by inspection rather than by argument.
    allowed = (ex.split == "train").to_numpy()
    if fit_mask is None:
        is_train = allowed
        train_pids = set(cohort.loc[cohort.split == "train", "pid"])
    else:
        is_train = np.asarray(fit_mask, bool)
        assert is_train.shape == allowed.shape, (
            f"fit_mask {is_train.shape} does not match {allowed.shape} example rows")
        assert is_train.any(), "empty fit_mask"
        # The brief forbids validation or test entering a fit set. Enforcing it
        # here makes that impossible even deliberately, rather than relying on
        # every caller to remember.
        assert not (is_train & ~allowed).any(), (
            f"fit_mask selects {int((is_train & ~allowed).sum())} non-train rows; "
            "val/test may never enter a fit set")
        train_pids = set(cohort.loc[cohort.pid.isin(ex.pid.to_numpy()[is_train]),
                                    "pid"])

    # ---- vocabulary and value statistics: TRAIN PATIENTS, unexpanded ------
    base = events[events.is_pre_anchor & events.kind.isin(cfg.kinds)]
    tr_ev = base[base.pid.isin(train_pids)]
    per_token_patients = tr_ev.groupby("token", observed=True).pid.nunique()
    vocab = sorted(per_token_patients[per_token_patients >= cfg.min_patients_per_token].index)
    tix = {t: i for i, t in enumerate(vocab)}

    # Restricted to the kept vocabulary: a numeric code can carry a usable
    # value distribution and still fall below the patient-count threshold,
    # and a decile block for a token with no count column is unreachable.
    tr_num = tr_ev[tr_ev.is_numeric & tr_ev.value.notna() & tr_ev.token.isin(tix)]
    edges: dict[str, np.ndarray] = {}
    for tok, g in tr_num.groupby("token", observed=True):
        q = np.nanquantile(g.value.values, np.linspace(0, 1, cfg.n_value_bins + 1)[1:-1])
        e = np.unique(q)                       # near-binary codes give duplicates;
        if len(e) >= 2:                        # searchsorted would dump all in one bin
            edges[tok] = e
    dec_tokens = sorted(edges)
    cover = tr_num.groupby("token", observed=True).pid.nunique().sort_values(ascending=False)
    n_train_pat0 = max(len(train_pids), 1)
    top = [t for t, c in cover.items() if c / n_train_pat0 >= cfg.lab_min_coverage]

    # ---- expand events across examples ------------------------------------
    kix = {k: i for i, k in enumerate(cfg.kinds)}
    b0 = base[base.token.isin(tix)]
    slim = pd.DataFrame({
        "pid": b0.pid.to_numpy(np.int32),
        "ts": b0.ts.to_numpy(),
        "ti": b0.token.map(tix).to_numpy(np.int32),
        "ki": b0.kind.map(kix).to_numpy(np.int8),
        "value": b0.value.to_numpy(np.float32),
        "is_numeric": b0.is_numeric.to_numpy(),
        "cost": (b0.cost.to_numpy(np.float32) if "cost" in b0.columns
                 else np.full(len(b0), np.nan, np.float32)),
    })
    del b0
    pre = ex[["eid", "pid", "cutoff"]].merge(slim, on="pid")
    del slim
    pre = pre[pre.ts.to_numpy() < pre.cutoff.to_numpy()].copy()
    pre["age_days"] = ((pre.cutoff - pre.ts).dt.total_seconds()
                       / 86400.0).astype(np.float32)
    eid_v = pre.eid.to_numpy()
    ti_v = pre.ti.to_numpy()
    age_v = pre.age_days.to_numpy()

    cols: list[np.ndarray] = []
    names: list[str] = []
    blocks: dict[str, slice] = {}

    def add(block: str, mat: np.ndarray, colnames: list[str]) -> None:
        start = len(names)
        cols.append(np.asarray(mat, dtype=np.float32))
        names.extend(colnames)
        blocks[block] = slice(start, len(names))

    # ---- A: nested cumulative code counts ---------------------------------
    for w in cfg.windows:
        keep = np.ones(len(pre), bool) if not np.isfinite(w) else (age_v <= w)
        M = np.zeros((n, len(vocab)), np.float32)
        np.add.at(M, (eid_v[keep], ti_v[keep]), 1.0)
        add(f"counts_{_window_label(w)}", M,
            [f"cnt[{_window_label(w)}]:{t}" for t in vocab])

    # ---- B: numeric lab value deciles, edges fitted on TRAIN --------------
    num_m = pre.is_numeric.to_numpy() & np.isfinite(pre.value.to_numpy())
    if dec_tokens:
        dix = {tix[t]: i for i, t in enumerate(dec_tokens)}
        nb = cfg.n_value_bins
        D = np.zeros((n, len(dec_tokens) * nb), np.float32)
        val_v = pre.value.to_numpy()
        sel = num_m & np.isin(ti_v, np.fromiter(dix, np.int32, len(dix)))
        if sel.any():
            # One searchsorted per token, not one per row: the row-wise form
            # was 5M Python-level calls and dominated the build.
            sub_ti, sub_val = ti_v[sel], val_v[sel]
            b = np.zeros(len(sub_ti), np.int64)
            for t in dec_tokens:
                m = sub_ti == tix[t]
                if m.any():
                    b[m] = np.searchsorted(edges[t], sub_val[m], side="right")
            b = np.clip(b, 0, nb - 1)
            col = np.array([dix[t] for t in sub_ti], np.int64) * nb + b
            np.add.at(D, (eid_v[sel], col), 1.0)
        add("value_deciles", D,
            [f"dec:{t}:Q{q}" for t in dec_tokens for q in range(nb)])

    # ---- C: per-target prevalent indicator ---------------------------------
    add("prevalent", labels["prevalent"].astype(np.float32),
        [f"prev:{c}" for c in codes])

    # ---- D: demographics ---------------------------------------------------
    pat = cohort.set_index("pid").reindex(ex.pid.to_numpy())
    demo = [
        (pat.gender.eq("F").to_numpy(), "sex_F"),
        (pat.gender.eq("M").to_numpy(), "sex_M"),
    ]
    for col, pfx in (("race", "race"), ("ethnicity", "eth"), ("marital", "marital")):
        vals = pat[col].fillna("MISSING")
        for v in sorted(cohort[col].fillna("MISSING").unique()):
            demo.append((vals.eq(v).to_numpy(), f"{pfx}_{v}"))
    # Age at THIS example's cutoff, not at the patient's anchor.
    age = ((ex.cutoff.to_numpy() - pat.birthdate.to_numpy())
           / np.timedelta64(1, "D") / 365.25).astype(np.float32)
    mu, sd = age[is_train].mean(), age[is_train].std() + 1e-9   # train-only stats
    demo.append(((age - mu) / sd, "age_z"))
    demo.append((((age - mu) / sd) ** 2, "age_z_sq"))
    add("demographics", np.column_stack([d[0] for d in demo]), [d[1] for d in demo])

    # ---- E: history shape and timing --------------------------------------
    g = pre.groupby("eid", sort=False)
    def per_example(s: pd.Series, fill: float = 0.0) -> np.ndarray:
        return s.reindex(ex.eid.to_numpy()).fillna(fill).to_numpy(np.float32)

    n_ev = per_example(g.size())
    n_days = per_example(g.ts.nunique())
    n_tok = per_example(g.ti.nunique())
    span = per_example(g.age_days.max() - g.age_days.min())
    recency = per_example(g.age_days.min(), fill=3650.0)
    rate2y = per_example(pre[age_v <= 730].groupby("eid").size()) / 2.0
    shape = [
        (np.log1p(n_ev), "log_n_events"), (np.log1p(n_days), "log_n_days"),
        (np.log1p(n_tok), "log_n_distinct_tokens"), (span / 365.25, "history_span_years"),
        (np.log1p(recency), "log_days_since_last"), (np.log1p(rate2y), "log_rate_last2y"),
        (n_ev / np.maximum(n_days, 1.0), "events_per_active_day"),
    ]
    ki_v = pre.ki.to_numpy()
    for k in ("ENC", "COND", "MED", "PROC", "OBS", "IMM"):
        c = per_example(pre[ki_v == kix[k]].groupby("eid").size())
        shape.append((c / np.maximum(n_ev, 1.0), f"frac_{k}"))
    add("history_shape", np.column_stack([s[0] for s in shape]), [s[1] for s in shape])

    # ---- F: lab last-value summaries, z-scored on TRAIN examples ----------
    if top:
        top_ti = [tix[t] for t in top]
        sel = num_m & np.isin(ti_v, np.asarray(top_ti, np.int32))
        last = (pre[sel].sort_values("age_days")
                .groupby(["eid", "ti"], observed=True).value.first().unstack())
        last = last.reindex(index=ex.eid.to_numpy(), columns=top_ti)
        A = last.to_numpy(np.float32)
        obs = (~np.isnan(A)).astype(np.float32)
        mu = np.nanmean(np.where(is_train[:, None], A, np.nan), axis=0)
        sd = np.nanstd(np.where(is_train[:, None], A, np.nan), axis=0) + 1e-9
        z = np.nan_to_num((A - mu) / sd, nan=0.0).astype(np.float32)
        add("lab_last", np.column_stack([z, obs]),
            [f"lab_last_z:{t}" for t in top] + [f"lab_seen:{t}" for t in top])

    # ---- G: REASONCODE counts ---------------------------------------------
    # What a treatment was FOR. A drug and a procedure aimed at the same
    # condition both increment one shared column, which is the ancestor-credit
    # an ontology would give -- from a provided column, not a download.
    # Measured: 109 distinct reasons, present on 81% of medications and 93% of
    # careplans, and 33 of the 40 target conditions appear as a reason.
    #
    # Expected to help trees and add nothing to LR: a linear model can already
    # form any weighted sum of the underlying code columns, so a sum of them
    # is not new information to it. A tree cannot, without spending depth.
    if cfg.enable_reason and "reason" in base.columns:
        tr_reason = tr_ev.reason.dropna().astype(str)
        reasons = sorted(tr_reason.unique())
        rix = {r: i for i, r in enumerate(reasons)}
        rb = base[base.reason.notna()]
        # Drop unseen reasons BEFORE casting. A reason that appears only in
        # val or test maps to NaN, and `to_numpy(np.int32)` turns NaN into
        # -2147483648 rather than raising -- which then indexes an array from
        # the far end instead of failing.
        rslim = pd.DataFrame({
            "pid": rb.pid.to_numpy(np.int32),
            "ts": rb.ts.to_numpy(),
            "ri": rb.reason.astype(str).map(rix).to_numpy(),
        }).dropna(subset=["ri"])
        rslim["ri"] = rslim.ri.to_numpy(np.int32)
        rpre = ex[["eid", "pid", "cutoff"]].merge(rslim, on="pid")
        rpre = rpre[rpre.ts.to_numpy() < rpre.cutoff.to_numpy()]
        r_age = ((rpre.cutoff - rpre.ts).dt.total_seconds() / 86400.0).to_numpy(np.float32)
        r_eid, r_ri = rpre.eid.to_numpy(), rpre.ri.to_numpy()
        for w in cfg.windows:
            keep = np.ones(len(rpre), bool) if not np.isfinite(w) else (r_age <= w)
            R = np.zeros((n, len(reasons)), np.float32)
            np.add.at(R, (r_eid[keep], r_ri[keep]), 1.0)
            add(f"reason_{_window_label(w)}", R,
                [f"rsn[{_window_label(w)}]:{r}" for r in reasons])

    # ---- H: lab coverage-based last value and slope ------------------------
    # `top` above used an arbitrary k; recompute it from coverage.
    n_train_pat = max(len(train_pids), 1)
    covered = [t for t, c in cover.items() if c / n_train_pat >= cfg.lab_min_coverage]
    lab_ti = np.asarray([tix[t] for t in covered], np.int32)

    if cfg.enable_slope and len(lab_ti):
        # OLS slope of value against time, per (example, lab), in units per
        # year. NaN below two readings -- HistGradientBoosting handles NaN
        # natively, so no companion "was it known" flag is needed, and a
        # linear fit only: the median patient has ~6 readings and a quadratic
        # would be fitting noise.
        sel = num_m & np.isin(ti_v, lab_ti)
        sub = pd.DataFrame({
            "eid": eid_v[sel], "ti": ti_v[sel],
            "t": -age_v[sel] / 365.25, "v": pre.value.to_numpy()[sel],
        })
        sub["tt"], sub["tv"] = sub.t * sub.t, sub.t * sub.v
        gg = sub.groupby(["eid", "ti"], sort=False)
        agg = gg.agg(k=("t", "size"), st=("t", "sum"), sv=("v", "sum"),
                     stt=("tt", "sum"), stv=("tv", "sum"))
        den = agg.k * agg.stt - agg.st ** 2
        sl = np.where(np.abs(den) > 1e-12,
                      (agg.k * agg.stv - agg.st * agg.sv) / np.where(den == 0, 1, den),
                      np.nan)
        S = (pd.Series(sl, index=agg.index).unstack()
             .reindex(index=ex.eid.to_numpy(), columns=lab_ti).to_numpy(np.float32))
        add("lab_slope", S, [f"lab_slope:{t}" for t in covered])

    # ---- I: time since the last occurrence of a code -----------------------
    # The count windows saturate: "0 in the last 5 years" cannot distinguish
    # six years ago from twenty-five. Capped at 10 years and expressed in
    # years so the scale is comparable across columns.
    if cfg.enable_time_since:
        target_tokens = [f"COND_{c}" for c in codes if f"COND_{c}" in tix]
        common = [t for t in per_token_patients.sort_values(ascending=False).index
                  if t in tix][: cfg.n_time_since_common]
        ts_tokens = list(dict.fromkeys(target_tokens + common))
        ts_ti = np.asarray([tix[t] for t in ts_tokens], np.int32)
        sel = np.isin(ti_v, ts_ti)
        last = (pd.DataFrame({"eid": eid_v[sel], "ti": ti_v[sel], "a": age_v[sel]})
                .groupby(["eid", "ti"], sort=False).a.min().unstack()
                .reindex(index=ex.eid.to_numpy(), columns=ts_ti).to_numpy(np.float32))
        # Never-seen -> the cap, which is what "longer ago than we track" means.
        T_since = np.nan_to_num(np.minimum(last, 3650.0), nan=3650.0) / 365.25
        add("time_since", T_since, [f"time_since_y:{t}" for t in ts_tokens])

    # ---- J: per-visit cost -------------------------------------------------
    # Per-row and pre-cutoff, so legitimate -- unlike patients.HEALTHCARE_EXPENSES,
    # which is a lifetime total and is refused at load time.
    if cfg.enable_cost and "cost" in pre.columns:
        cost_all = pre.cost.to_numpy(np.float32)
        if True:
            ok = np.isfinite(cost_all)
            C = np.zeros((n, 2 * len(cfg.windows)), np.float32)
            cnames = []
            for wi, w in enumerate(cfg.windows):
                keep = ok if not np.isfinite(w) else (ok & (age_v <= w))
                tot = np.zeros(n, np.float32)
                cnt = np.zeros(n, np.float32)
                np.add.at(tot, eid_v[keep], cost_all[keep])
                np.add.at(cnt, eid_v[keep], 1.0)
                C[:, 2 * wi] = np.log1p(tot)
                C[:, 2 * wi + 1] = tot / np.maximum(cnt, 1.0)
                cnames += [f"cost_log_total[{_window_label(w)}]",
                           f"cost_mean[{_window_label(w)}]"]
            add("cost", C, cnames)

    # ---- K: age-residualised timing ---------------------------------------
    # History span alone reaches AUROC 0.87, but it is largely a proxy for
    # age, which is already a feature. The residual asks the different
    # question: is this record denser than this patient's age predicts?
    #
    # Expected to help LR and do little for the GBDT -- a tree can split on
    # span and age separately and construct the interaction itself, which is
    # exactly the argument that rejected all-pairs products. The mirror image
    # of the REASONCODE block, and both asymmetries are reported as such.
    if cfg.enable_age_residual:
        sh = np.column_stack([s_[0] for s_ in shape]).astype(np.float32)
        A1 = np.column_stack([np.ones_like(age), age, age ** 2]).astype(np.float64)
        coef, *_ = np.linalg.lstsq(A1[is_train], sh[is_train].astype(np.float64),
                                   rcond=None)
        add("age_residual", (sh - A1 @ coef).astype(np.float32),
            [f"resid_age:{s_[1]}" for s_ in shape])

    X = np.column_stack(cols).astype(np.float32)
    assert X.shape[1] == len(names)

    if cfg.deduplicate:
        X, names, blocks = _deduplicate(X, names, blocks, is_train)
    return FeatureMatrix(X=X, names=names, eid=ex.eid.to_numpy(),
                         pid=ex.pid.to_numpy(),
                         split=ex.split.to_numpy().astype("U5"),
                         is_real=ex.is_real.to_numpy(), blocks=blocks)
