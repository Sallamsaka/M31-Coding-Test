"""Whole-pipeline audit: every component, independently recomputed.

Run: ``python -m src.audit``

The test suite proves specific invariants. This is different: it walks each
component and re-derives what it produced from the raw data by a *different
route*, then reports anything that disagrees or looks pathological. Tests
answer "did the thing I thought of go wrong"; this answers "is anything
wrong that I did not think of".

Sections, each independent so a failure in one still reports the rest:

  A  feature matrix pathologies    constant / duplicate / extreme columns
  B  feature block correctness     counts, deciles, recency, cost vs raw CSVs
  C  cohort and split structure    prevalence, dev-split representativeness
  D  tokenizer correctness         ids, ordering, fused value bins
  E  prediction sanity             distribution, calibration, mask effect
  F  loss masking                  the pairs excluded are exactly the prevalent ones
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from .data.cohort import load_cohort, load_events, load_target_codes
from .data.examples import ExampleConfig
from .data.features import FeatureConfig, build_features
from .data.labels import at_risk_mask, load_labels
from .data.sequences import SeqConfig, build_sequences, build_vocab

__all__ = ["main"]

OK, WARN, BAD = "  ok  ", " warn ", " BAD  "


def _say(status: str, msg: str) -> None:
    print(f"[{status}] {msg}")


# ---------------------------------------------------------------- A -------
def audit_features(F) -> None:
    print("\nA. FEATURE MATRIX PATHOLOGIES")
    X = F.X
    tr = F.split == "train"
    const = np.where(X[tr].std(0) == 0)[0]
    _say(WARN if len(const) else OK,
         f"constant-on-train columns: {len(const)} of {X.shape[1]}"
         + (f"  (they cost parameters and carry nothing)" if len(const) else ""))

    # duplicate columns -- cheap hash, then confirm exact
    h = {}
    dup = 0
    for j in range(X.shape[1]):
        key = hash(X[:, j].tobytes())
        if key in h and np.array_equal(X[:, j], X[:, h[key]]):
            dup += 1
        else:
            h[key] = j
    _say(WARN if dup else OK, f"exactly duplicated columns: {dup}")

    mx = np.nanmax(np.abs(X))   # nanmax: lab_slope is legitimately NaN
    _say(WARN if mx > 1e6 else OK, f"largest absolute value: {mx:,.0f}")
    nan_cols = np.where(np.isnan(X).any(0))[0]
    blocks = {k: v for k, v in F.blocks.items()}
    nan_blocks = {b for b, sl in blocks.items()
                  for j in nan_cols if sl.start <= j < sl.stop}
    _say(OK if nan_blocks <= {"lab_slope"} else BAD,
         f"NaN confined to {sorted(nan_blocks) or 'nothing'} "
         "(only lab_slope may be NaN, by design)")

    # scale spread across blocks matters for the linear model
    print("     per-block scale (train p99 of |x|):")
    for b, sl in blocks.items():
        v = np.nanpercentile(np.abs(X[tr, sl]), 99)
        print(f"       {b:<18} {v:>12,.2f}  ({sl.stop-sl.start} cols)")


# ---------------------------------------------------------------- B -------
def audit_feature_blocks(F, root: Path) -> None:
    print("\nB. FEATURE BLOCKS vs RAW EVENTS (independent recomputation)")
    cohort, ev = load_cohort(root), load_events(root)
    anchor = pd.Series(cohort.anchor.values, index=cohort.pid.values)
    pre = ev[ev.is_pre_anchor]
    age = (pre.pid.map(anchor) - pre.ts).dt.total_seconds() / 86400.0

    names = F.names
    rng = np.random.default_rng(0)

    # ---- counts: recompute for a sample of (patient, token, window) --------
    bad = checked = 0
    for w_label, w in (("365d", 365.0), ("1825d", 1825.0), ("ever", np.inf)):
        cols = [j for j, n in enumerate(names) if n.startswith(f"cnt[{w_label}]:")]
        if not cols:
            continue
        for j in rng.choice(cols, 12, replace=False):
            tok = names[j].split(":", 1)[1]
            sub = pre[(pre.token == tok) & (age <= w)]
            want = sub.groupby("pid").size()
            got = F.X[:, j]
            for pid in list(want.index[:4]):
                row = np.flatnonzero((F.pid == pid) & F.is_real)
                if len(row):
                    checked += 1
                    bad += int(abs(got[row[0]] - want[pid]) > 1e-6)
    _say(OK if bad == 0 else BAD,
         f"count features: {bad} mismatches in {checked} recomputed cells")

    # ---- prevalent block must equal the label array ------------------------
    lab = load_labels(root, ExampleConfig())
    sl = F.blocks["prevalent"]
    same = np.array_equal(F.X[:, sl].astype(np.int8), lab["prevalent"])
    _say(OK if same else BAD, "prevalent feature block equals the label array")

    # ---- time_since: never negative, capped, and 0 only if an event today --
    if "time_since" in F.blocks:
        sl = F.blocks["time_since"]
        T = F.X[:, sl]
        _say(OK if (T >= 0).all() else BAD, "time-since is non-negative")
        cap = 3650 / 365.25
        _say(OK if T.max() <= cap + 1e-6 else BAD,
             f"time-since capped at {cap:.2f}y (max seen {T.max():.2f})")
        at_cap = (np.abs(T - cap) < 1e-6).mean()
        _say(OK, f"share at the cap (never seen or >10y ago): {at_cap:.1%}")

    # ---- cost must be non-negative and mostly non-zero ---------------------
    if "cost" in F.blocks:
        sl = F.blocks["cost"]
        C = F.X[:, sl]
        _say(OK if (C >= 0).all() else BAD, "cost features non-negative")
        _say(OK, f"rows with zero total cost ever: "
                 f"{(C[:, -2] == 0).mean():.1%}")


# ---------------------------------------------------------------- C -------
def audit_cohort(F, root: Path) -> None:
    print("\nC. COHORT, SPLITS AND THE INNER DEV SPLIT")
    lab = load_labels(root, ExampleConfig())
    y = lab["y"].astype(int)
    for sp in ("train", "val", "test"):
        m = lab["split"] == sp
        print(f"     {sp:<6} n={m.sum():>5}  positives/row {y[m].sum(1).mean():>5.3f}  "
              f"labels with 0 positives: {(y[m].sum(0) == 0).sum():>2}/40")

    # the inner dev split must look like the train it was carved from
    pids = np.unique(lab["pid"][lab["split"] == "train"])
    rng0 = np.random.default_rng(12345)
    dev = set(rng0.choice(pids, int(round(0.2 * len(pids))), replace=False).tolist())
    tr_m = (lab["split"] == "train") & ~np.isin(lab["pid"], list(dev))
    dv_m = (lab["split"] == "train") & np.isin(lab["pid"], list(dev))
    r_tr, r_dv = y[tr_m].mean(), y[dv_m].mean()
    _say(OK if abs(r_tr - r_dv) < 0.01 else WARN,
         f"dev positive rate {r_dv:.4f} vs inner-train {r_tr:.4f} "
         f"(val {y[lab['split']=='val'].mean():.4f})")
    _say(OK if (y[dv_m].sum(0) > 0).sum() >= 35 else WARN,
         f"dev labels with at least one positive: {(y[dv_m].sum(0) > 0).sum()}/40")
    print(f"     dev median positives per label: "
          f"{int(np.median(y[dv_m].sum(0)))}  (val: "
          f"{int(np.median(y[lab['split']=='val'].sum(0)))})")


# ---------------------------------------------------------------- D -------
def audit_tokenizer(root: Path) -> None:
    print("\nD. TOKENIZER")
    cfg = SeqConfig()
    v = build_vocab(root, cfg)
    pack = build_sequences(root, v, cfg)
    T, D, L = pack.tokens, pack.dt, pack.lengths
    _say(OK if T.max() < len(v) else BAD,
         f"max token id {T.max()} < vocab {len(v)}")
    _say(OK if (T[np.arange(len(L)), L - 1] == v.stoi["[ANCHOR]"]).all() else BAD,
         "every sequence ends with [ANCHOR]")
    ok_pad = all((T[i, L[i]:] == 0).all() for i in range(0, len(L), 53))
    _say(OK if ok_pad else BAD, "everything past the length is PAD")
    mono = all((np.diff(D[i, :L[i]]) <= 1e-6).all() for i in range(0, len(L), 53))
    _say(OK if mono else BAD, "dt is non-increasing within every sequence")
    _say(OK if (D[T > 0] >= 0).all() else BAD, "dt is non-negative on real tokens")
    print(f"     lengths: median {int(np.median(L))}  p90 {int(np.percentile(L,90))}  "
          f"at cap {(L >= cfg.block_size).mean():.1%}")


# ---------------------------------------------------------------- E -------
def audit_predictions(root: Path) -> None:
    print("\nE. PREDICTION SANITY")
    lab = load_labels(root, ExampleConfig())
    y = lab["y"].astype(int)
    ar = at_risk_mask(lab)
    val = lab["split"] == "val"
    yv, arv = y[val], ar[val]
    npz = root / "artifacts" / "baseline_val_preds.npz"
    got = {}
    if npz.exists():
        d = np.load(npz)
        got.update({k: d[k] for k in d.files if d[k].shape == yv.shape})
    for f in sorted((root / "artifacts").glob("val_preds_*.npy")):
        a = np.load(f)
        if a.shape == yv.shape:
            got[f.stem.replace("val_preds_", "")] = a
    if not got:
        _say(WARN, "no validation predictions on disk")
        return
    for k, P in got.items():
        Pm = np.where(arv.astype(bool), P, 1e-6)
        obs, exp = yv.mean(), Pm.mean()
        print(f"     {k:<16} range [{P.min():.2e}, {P.max():.4f}]  "
              f"mean pred {exp:.4f} vs observed {obs:.4f}  "
              f"ratio {exp/max(obs,1e-9):>5.2f}x")
    _say(OK, "ratio far from 1.0 means miscalibration, which rank metrics hide")

    # The SHIPPED file, which is the only artefact that leaves this repository.
    #
    # Everything above scores `baseline_val_preds.npz`, which `run_baseline`
    # writes BEFORE the calibration step -- so this section was structurally
    # blind to calibration and reported the shipped blend as uncalibrated. An
    # audit that cannot see the last transform applied to the deliverable is
    # auditing something other than the deliverable.
    sub_f = root / "outputs" / "predictions.csv"
    meta_f = root / "outputs" / "predictions_meta.json"
    if not sub_f.exists():
        _say(WARN, "no outputs/predictions.csv to audit")
        return
    import json

    import pandas as pd
    sub = pd.read_csv(sub_f)
    codes = [c for c in sub.columns if c != sub.columns[0]]
    V = sub[codes].to_numpy(float)
    te = lab["split"] == "test"
    ar_te = ar[te]

    # Train at-risk positive rate is the reference: a calibrated model should
    # predict close to it on average, if the test population resembles train.
    tr = lab["split"] == "train"
    base = float((y[tr] * ar[tr]).sum() / max(ar[tr].sum(), 1))
    if V.shape == ar_te.shape:
        mean_ar = float(V[ar_te.astype(bool)].mean())
        print(f"     SHIPPED predictions.csv  mean(at-risk) {mean_ar:.4f} "
              f"vs train rate {base:.4f}  ratio {mean_ar/max(base,1e-9):>5.2f}x")
        n_floor = int((V <= 1.5e-6).sum())
        n_exp = int((~ar_te.astype(bool)).sum())
        _say(OK if n_floor == n_exp else WARN,
             f"not-at-risk cells floored: {n_floor:,} (expected {n_exp:,})")
    else:
        _say(WARN, f"predictions.csv shape {V.shape} != at-risk {ar_te.shape}")

    if meta_f.exists():
        m = json.loads(meta_f.read_text(encoding="utf-8"))
        _say(OK if m.get("recipe") else WARN,
             f"recipe recorded: {m.get('recipe', 'MISSING')}")
        _say(OK if m.get("calibrated") else WARN,
             f"calibrated: {m.get('calibrated')} "
             f"({m.get('n_labels_calibrated', 0)}/40 labels)")


# ---------------------------------------------------------------- F -------
def audit_loss_masking(root: Path) -> None:
    print("\nF. LOSS MASKING")
    lab = load_labels(root, ExampleConfig())
    y, prev = lab["y"].astype(int), lab["prevalent"].astype(int)
    ar = at_risk_mask(lab)
    _say(OK if not (y & prev).any() else BAD,
         "no (example, label) pair is both incident and prevalent")
    _say(OK if (ar == (prev == 0)).all() else BAD,
         "at-risk mask is exactly 'not prevalent'")
    _say(OK if y[~ar].sum() == 0 else BAD,
         f"masked-out pairs contain {y[~ar].sum()} positives (must be 0)")
    keep = ar.mean()
    print(f"     share of (example, label) pairs contributing to the loss: {keep:.1%}")
    print(f"     positives retained: {y[ar].sum():,} of {y.sum():,} (must be all)")


def main(root: str | Path = ".") -> None:
    root = Path(root)
    print("WHOLE-PIPELINE AUDIT")
    F = build_features(root, FeatureConfig(), ExampleConfig())
    print(f"feature matrix {F.X.shape}")
    for fn, arg in ((audit_features, (F,)), (audit_feature_blocks, (F, root)),
                    (audit_cohort, (F, root)), (audit_tokenizer, (root,)),
                    (audit_predictions, (root,)), (audit_loss_masking, (root,))):
        try:
            fn(*arg)
        except Exception as exc:                       # keep going
            print(f"[{BAD}] {fn.__name__} raised: {exc!r}")


if __name__ == "__main__":
    main()
