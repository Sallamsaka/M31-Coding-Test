"""Data-quality checks whose answers belong in the report, not in a model.

Run: ``python -m src.diagnostics``

``DEATHDATE`` is read here and ONLY here. It is forbidden as a feature --
``cohort.py`` refuses to load it, because it is populated for 1,354 of 3,156
train/val patients and zero of 358 test patients, so a model using it would
look excellent in training and contribute nothing at test. Reading it to
*characterise the cohort* is a different act from feeding it to a model, and
the distinction is worth making explicitly rather than avoiding the column out
of superstition. Nothing in this module returns a feature.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data.cohort import load_cohort, load_events
from .utils.timeutil import to_ts

__all__ = ["censoring_report", "anchor_selection_report",
           "selection_ledger_report"]


# Patient-level bootstrap SD of macro AUROC on the 365 validation patients,
# B=300, measured across all four models on disk: prevalence 0.0017, lr 0.0105,
# P3 0.0102, P4 0.0105. The prevalence model is excluded from the average --
# it is a constant predictor, so its only variation is which patients the
# resample drew, and it is not a candidate anyone selects on.
#
# Reproduce: see outputs/sigma_check.log.
SIGMA_MACRO_AUROC = 0.0104


def _death_dates(root: Path) -> pd.Series:
    """``pid -> DEATHDATE``. Analysis only; never joined to a feature frame."""
    cohort = load_cohort(root)
    pid_of = pd.Series(cohort.pid.values, index=cohort.patient_id.values)
    out = []
    for sub in ("train_val", "test"):
        pat = pd.read_csv(root / sub / "patients.csv", dtype=str,
                          usecols=["Id", "DEATHDATE"])
        pat["pid"] = pat.Id.map(pid_of)
        out.append(pat.dropna(subset=["pid"]))
    pat = pd.concat(out, ignore_index=True)
    return pd.Series(to_ts(pat.DEATHDATE).values,
                     index=pat.pid.astype(int).values).dropna()


def censoring_report(root: Path | str = ".") -> pd.DataFrame:
    """Is anyone's 5-year outcome window cut short?

    Expected answer: no, and for a structural reason. The anchor is *defined*
    as five years before the last recorded encounter, so the window closes
    exactly on that encounter by construction. Follow-up cannot be short.
    Worth asserting rather than assuming, because if it were violated the
    missing follow-up would be informative -- sicker patients have denser
    records -- and every affected patient would be a silent false negative.
    """
    root = Path(root)
    cohort = load_cohort(root)
    ev = load_events(root)
    last = ev.groupby("pid", observed=True).ts.max()
    d = cohort.assign(last_event=cohort.pid.map(last))
    d["follow_up_y"] = (d.last_event - d.anchor).dt.days / 365.25
    d["short"] = d.follow_up_y < 4.99

    obs = d[d.split != "test"]
    print("CENSORING")
    print(f"  follow-up from anchor to last event, train+val: "
          f"median {obs.follow_up_y.median():.2f}y  "
          f"min {obs.follow_up_y.min():.2f}y  max {obs.follow_up_y.max():.2f}y")
    print(f"  windows shorter than 5y: {int(obs.short.sum())} of {len(obs)}")
    print("  test split is truncated at the anchor by the organisers, so its "
          f"follow-up is 0 by construction (median {d[d.split=='test'].follow_up_y.median():.2f}y)")
    return d


def anchor_selection_report(root: Path | str = ".") -> None:
    """What does defining the anchor from the LAST encounter actually select?

    This is the more interesting question, and it is not censoring. Because
    the window ends on a patient's final encounter, a patient who dies is
    anchored at five years before their death. The outcome window is then
    exactly their last five years of life -- a period with an atypically high
    diagnosis rate. For patients who simply run out of simulation, it is an
    ordinary five years.

    So the cohort is a mixture of two regimes, and the mixing proportion is
    knowable only from a column we are forbidden to model on. Anything the
    model learns about "patients whose records are about to end" is learned
    from that mixture. It belongs in the report's Interpretation section, and
    it is a limitation of the task construction rather than of the model.
    """
    root = Path(root)
    cohort = load_cohort(root)
    death = _death_dates(root)
    d = cohort.assign(death=cohort.pid.map(death))
    obs = d[d.split != "test"].copy()
    obs["died"] = obs.death.notna()
    obs["death_minus_window_end_d"] = (obs.death - obs.window_end).dt.days

    print("\nANCHOR SELECTION")
    print(f"  train+val patients with a death date: "
          f"{int(obs.died.sum()):,} of {len(obs):,} ({obs.died.mean():.1%})")
    dd = obs.loc[obs.died, "death_minus_window_end_d"]
    print(f"  death date minus window end (days): median {dd.median():.0f}  "
          f"|within 30d| {(dd.abs() <= 30).mean():.1%}  "
          f"|within 365d| {(dd.abs() <= 365).mean():.1%}")
    in_win = ((obs.death >= obs.anchor) & (obs.death <= obs.window_end))
    print(f"  deaths falling INSIDE the 5-year outcome window: "
          f"{int(in_win.sum()):,} ({in_win.mean():.1%} of all train+val)")
    print("  -> for these patients the outcome window is literally their last "
          "five years of life,\n     which is not an ordinary five years. The "
          "cohort is a mixture of two regimes.")

    print(f"\n  test patients with a death date: "
          f"{int(d[d.split=='test'].death.notna().sum())} of 358  "
          "(organisers stripped the column)")


def selection_ledger_report(root: Path | str = ".") -> int:
    """How many times has the validation set been looked at?

    Taking the best of M candidates on one held-out set inflates the winner by
    roughly ``sigma * sqrt(2 ln M)`` even when every candidate is equally good.
    At the *measured* sigma (see ``SIGMA_MACRO_AUROC``) and M = 20 that is
    **+0.025 macro AUROC of pure noise** -- still larger than every difference
    this project has measured between models, which is the point.

    This used to hardcode ``sigma = 0.02``, giving +0.049 at M = 20 and +0.051
    at M = 26. That constant was never measured and is ~2x the real value, so
    the headline bound was inflated by a factor of two. The conclusion survived
    the correction -- 0.026 still exceeds the 0.0109 LR-vs-P4 gap -- but an
    unmeasured constant had no business sitting among measured numbers in the
    one figure the "nothing is resolvable" argument rests on.

    The defence is not to look less; it is to count, and to report the count so
    a reader can discount the headline themselves. `wandb_shim` appends every
    evaluation to `outputs/metrics.jsonl` for exactly this purpose.
    """
    root = Path(root)
    p = root / "outputs" / "metrics.jsonl"
    if not p.exists():
        print("\nSELECTION LEDGER\n  no metrics.jsonl yet")
        return 0

    import json
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Baseline models log under a prefixed key (`lr/macro_auroc`) while the
    # transformer logs a bare one, so match on the suffix. An earlier version
    # matched the bare key only and reported 2 evaluations when there were
    # dozens -- a ledger that undercounts is worse than no ledger, because it
    # licenses exactly the over-searching it exists to expose.
    def scored(r: dict) -> list[str]:
        return [k for k in r if k.split("/")[-1] == "macro_auroc"]

    evals = [(r, k) for r in rows for k in scored(r)]
    candidates = {f"{r.get('run')}::{k.rsplit('/', 1)[0] if '/' in k else r.get('run')}"
                  for r, k in evals}
    m = max(len(candidates), 1)

    def bound(k: int, sigma: float = SIGMA_MACRO_AUROC) -> float:
        return sigma * np.sqrt(2 * np.log(k)) if k > 1 else 0.0

    print("\nSELECTION LEDGER")
    print(f"  validation evaluations recorded : {len(evals):,}")
    print(f"  distinct model candidates       : {m}")
    print(f"  winner's curse at M={m:<4} (families)  : +{bound(m):.3f} macro AUROC")
    print(f"  winner's curse at M={len(evals):<4} (every look): "
          f"+{bound(len(evals)):.3f} macro AUROC")
    print("  The second figure is the honest upper bound. Keeping the best")
    print("  epoch by validation AP is early stopping, and early stopping IS")
    print("  repeated use of the validation set -- 20 epochs is 20 looks, not")
    print("  one. The truth sits between the two rows, because consecutive")
    print("  epochs are strongly correlated and so are not independent draws;")
    print("  neither bound is tight, and reporting only the smaller one would")
    print("  be choosing the flattering half of a number we know is wrong.")
    return len(evals)


def main() -> None:
    censoring_report(".")
    anchor_selection_report(".")
    selection_ledger_report(".")


if __name__ == "__main__":
    main()
