"""One-change-at-a-time transformer sweep on the canonical 5-fold partition.

Run:  ``python -u -m src.tx_sweep --variant B1_age``        (one arm, ~22 min)
      ``python -m src.tx_sweep --compare B1_age B0_nobias``  (paired analysis)

Why a separate module rather than a flag on ``cross_validate.run_cv``: run_cv
rewrites ``artifacts/cv_oof_preds.npz`` and ``outputs/cv_per_fold.csv``, which
the SHIPPED calibration reads, and it refits every model family. A sweep arm
must touch neither. Everything here is written under its own names.

Protocol (pre-registered in §E57 before any arm ran):
- every arm is the shipped recipe (``cross_validate.TRANSFORMER_CFG``) plus the
  overrides below, trained on each fold's fit rows, one seed (300);
- B0 is the shipped recipe with the dt bias off (the bias was measured inert by
  zeroing it: val AUROC 0.7650 -> 0.7651), so every other arm differs from B0 in
  exactly one field;
- primary comparison: transformer alone, paired, on all 2,791 OOF rows;
  secondary: the LR+GBDT+transformer logit-mean blend on folds 0-2 (1,675 rows),
  the only folds with cached LR/GBDT predictions for this partition;
- an arm is a *candidate* if P(A>B) >= 0.75 on AP and its AUROC point estimate
  is >= 0 (or the reverse); candidates are adopted only if their combination is
  itself positive-or-neutral on both metrics against B0 (§E50.2).

Resolution, derived in §E57: SE of a paired delta ~0.0093 AP / 0.0027 AUROC, so
an arm with a true effect under ~0.005 AP will almost always read "not
resolvable". Selecting the best of K noise-only arms inflates by SE*sqrt(2 ln K).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .cross_validate import TRANSFORMER_CFG, _fit_transformer, _model_sig
from .cv import fold_masks
from .data.examples import ExampleConfig, load_examples
from .data.labels import at_risk_mask, load_labels
from .evaluate import macro_ap, macro_auroc, paired_bootstrap_delta
from .train_baseline import apply_at_risk_mask

__all__ = ["VARIANTS", "run", "compare"]

# Each arm = B0 + exactly one change. B7 is filled in after B1-B3 are read.
VARIANTS: dict[str, dict] = {
    "B0_nobias":  {"use_dt_bias": False},
    "B1_age":     {"use_dt_bias": False, "use_age_encoding": True},
    "B2_heads4":  {"use_dt_bias": False, "n_head": 4},
    "B3_reason":  {"use_dt_bias": False, "use_reason_embedding": True},
    "B7_2layer":  {"use_dt_bias": False, "n_layer": 2},
}
SEEDS = (300,)
CACHE = Path("artifacts/cv_fold_cache")


def _sig(overrides: dict, seeds) -> str:
    spec = (sorted(dict(TRANSFORMER_CFG, **overrides).items()), tuple(seeds))
    return hashlib.sha256(repr(spec).encode()).hexdigest()[:8]


def _fold_key(root: str, fit_rows: np.ndarray) -> str:
    """Identical to run_cv's key, so fold k here is fold k there."""
    pid = load_examples(root, ExampleConfig()).pid.to_numpy()
    blob = ",".join(map(str, sorted(set(pid[fit_rows].tolist()))))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def run(variant: str, root: str = ".", overrides: dict | None = None,
        seeds=SEEDS) -> Path:
    over = dict(VARIANTS[variant] if overrides is None else overrides)
    lab = load_labels(root, ExampleConfig())
    y, ar = lab["y"].astype(int), at_risk_mask(lab)
    folds = fold_masks(root, n_splits=5)
    sig = _sig(over, seeds)
    CACHE.mkdir(parents=True, exist_ok=True)
    oof = np.zeros(y.shape, np.float32)
    scored = np.zeros(len(y), bool)
    print(f"[{variant}] overrides={over} seeds={list(seeds)} sig={sig}", flush=True)
    t0 = time.time()
    for k, (fit_rows, oof_rows) in enumerate(folds):
        path = CACHE / f"sweep_fold{k}_{_fold_key(root, fit_rows)}_{sig}.npz"
        t = time.time()
        if path.exists():
            P = np.load(path)["p"]
            how = "cached"
        else:
            P = _fit_transformer(root, fit_rows, None, y.shape,
                                 overrides=over, seeds=seeds)
            np.savez_compressed(path, p=P)
            how = f"{(time.time() - t) / 60:.1f} min"
        idx = np.flatnonzero(oof_rows)
        Pm = apply_at_risk_mask(P, ar)
        oof[idx], scored[idx] = Pm[idx], True
        print(f"[{variant}] fold {k}: AUROC {macro_auroc(y[idx], Pm[idx])[0]:.4f} "
              f"AP {macro_ap(y[idx], Pm[idx])[0]:.4f}  ({how})", flush=True)
    out = Path(f"artifacts/tx_sweep_{variant}.npz")
    np.savez_compressed(out, p=oof, scored=scored,
                        meta=json.dumps({"variant": variant, "overrides": over,
                                         "seeds": list(seeds), "sig": sig}))
    r = scored
    print(f"[{variant}] OOF n={int(r.sum())}: AUROC {macro_auroc(y[r], oof[r])[0]:.4f} "
          f"AP {macro_ap(y[r], oof[r])[0]:.4f}  total {(time.time() - t0) / 60:.1f} min "
          f"-> {out}", flush=True)
    return out


def _blend_parts(root: str, y_shape):
    """Cached LR and GBDT OOF for the folds that have them (0-2 here)."""
    folds = fold_masks(root, n_splits=5)
    lr = np.zeros(y_shape, np.float32)
    gb = np.zeros(y_shape, np.float32)
    have = np.zeros(y_shape[0], bool)
    for k, (fit_rows, oof_rows) in enumerate(folds):
        fk = _fold_key(root, fit_rows)
        pl = CACHE / f"fold{k}_{fk}_lr_{_model_sig('lr')}.npz"
        pg = CACHE / f"fold{k}_{fk}_gbdt_{_model_sig('gbdt')}.npz"
        if pl.exists() and pg.exists():
            idx = np.flatnonzero(oof_rows)
            lr[idx] = np.load(pl)["p"][idx]
            gb[idx] = np.load(pg)["p"][idx]
            have[idx] = True
    return lr, gb, have


def compare(a: str, b: str, root: str = ".") -> dict:
    lab = load_labels(root, ExampleConfig())
    y, ar = lab["y"].astype(int), at_risk_mask(lab)
    A, B = (np.load(f"artifacts/tx_sweep_{v}.npz") for v in (a, b))
    rows = A["scored"] & B["scored"]
    out = {}

    def _line(tag, d):
        for m in ("ap", "auroc"):
            pt, lo, hi, _ = d[f"d_macro_{m}"]
            res = "resolvable" if (lo > 0 or hi < 0) else "not resolvable"
            print(f"  {tag:<24} d{m.upper():<6} {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
                  f"P(A>B)={d[f'p_a_gt_b_{m}']:.2f}  {res}", flush=True)

    print(f"{a} vs {b}", flush=True)
    d = paired_bootstrap_delta(y[rows], A["p"][rows], B["p"][rows])
    _line(f"transformer, n={int(rows.sum())}", d)
    out["tx"] = d

    lr, gb, have = _blend_parts(root, y.shape)
    br = rows & have
    if br.any():
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        blend = lambda t: apply_at_risk_mask(
            1 / (1 + np.exp(-(lg(lr) + lg(gb) + lg(t)) / 3)), ar)
        db = paired_bootstrap_delta(y[br], blend(A["p"])[br], blend(B["p"])[br])
        _line(f"LR+GBDT+TX blend, n={int(br.sum())}", db)
        out["blend"] = db

    ap_p, au_p = d["p_a_gt_b_ap"], d["p_a_gt_b_auroc"]
    cand = ((ap_p >= 0.75 and d["d_macro_auroc"][0] >= 0)
            or (au_p >= 0.75 and d["d_macro_ap"][0] >= 0))
    print(f"  pre-registered rule: {'CANDIDATE' if cand else 'not a candidate'}",
          flush=True)
    out["candidate"] = bool(cand)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS))
    ap.add_argument("--overrides", default=None,
                    help="JSON overrides replacing the variant's (for combined arms)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    elif a.variant:
        run(a.variant, overrides=json.loads(a.overrides) if a.overrides else None)
    else:
        ap.error("give --variant or --compare")


if __name__ == "__main__":
    main()
