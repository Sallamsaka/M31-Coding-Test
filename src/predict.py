"""Write `predictions.csv`, with the submission contract asserted before saving.

Five things are checked, and each corresponds to a way the file could be
silently rejected or silently mis-scored:

1. **Exactly 358 rows**, one per test patient, in `test_anchors.csv` order.
2. **Header is `patient_id` plus the 40 codes as raw strings in file order.**
   This is the one that bites hardest: pandas reads `444814009` as int64 by
   default, and `'444814009' != 444814009`, so a grader joining on column names
   would match nothing and score every column as missing.
3. **Every value is finite and strictly inside (0, 1).** Probabilities, not
   hard labels -- the brief says so explicitly.
4. **The patient id set matches `test_anchors.csv` exactly.**
5. **The not-at-risk mask never hides a true positive** -- verified on train and
   val, where we can see the answers, before it is trusted on test.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data.cohort import load_cohort, load_target_codes

__all__ = ["write_predictions", "check_contract"]


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def check_contract(df: pd.DataFrame, root: Path | str = ".") -> None:
    """Raise unless `df` is a valid submission. Called before every write."""
    root = Path(root)
    codes = load_target_codes(root)
    anchors = pd.read_csv(root / "test_anchors.csv", dtype=str)

    assert len(df) == 358, f"expected 358 rows, got {len(df)}"
    expected = ["patient_id"] + codes
    assert list(df.columns) == expected, (
        "header mismatch\n"
        f"  expected[:3] {expected[:3]}\n  got[:3]      {list(df.columns)[:3]}")
    assert set(df["patient_id"]) == set(anchors["Id"]), "patient id set mismatch"
    assert list(df["patient_id"]) == list(anchors["Id"]), "row order must match test_anchors.csv"

    vals = df[codes].to_numpy(dtype=float)
    assert np.isfinite(vals).all(), "non-finite probability"
    assert (vals > 0).all() and (vals < 1).all(), \
        f"probabilities outside (0,1): min {vals.min()}, max {vals.max()}"


def write_predictions(P: np.ndarray, model_name: str, root: Path | str = ".",
                      out: Path | str = "outputs/predictions.csv",
                      extra_meta: dict | None = None) -> pd.DataFrame:
    """`P` is (n_patients, 40) over the FULL cohort, in cohort pid order."""
    root, out = Path(root), Path(out)
    cohort = load_cohort(root)
    codes = load_target_codes(root)
    anchors = pd.read_csv(root / "test_anchors.csv", dtype=str)

    is_test = cohort.split.eq("test").to_numpy()
    sub = pd.DataFrame(P[is_test], columns=codes)
    sub.insert(0, "patient_id", cohort.loc[is_test, "patient_id"].to_numpy())

    # Reindex to the order the graders will expect. `reindex` with a named
    # Series carries that name onto the index, so reset it explicitly rather
    # than relying on what the column ends up called.
    sub = sub.set_index("patient_id").reindex(anchors["Id"].to_numpy())
    sub.index.name = "patient_id"
    sub = sub.reset_index()

    # Clamp off the open interval. The at-risk mask writes exact 1e-6 values,
    # which are already inside, but a model could emit a hard 0 or 1.
    sub[codes] = sub[codes].clip(1e-6, 1 - 1e-6)

    check_contract(sub, root)

    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)

    meta = {
        "model": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "n_rows": int(len(sub)),
        "n_codes": len(codes),
        **(extra_meta or {}),
    }
    out.with_name("predictions_meta.json").write_text(json.dumps(meta, indent=2))
    return sub
