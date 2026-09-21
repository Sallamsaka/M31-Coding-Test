"""Final head-to-head: every model on the same 365 validation patients.

Run: ``python -m src.compare``

Applies the selection rule that was fixed in advance, rather than reading the
table and picking a winner afterwards:

    Rank by validation macro AP. Adopt the winner only if it beats the
    runner-up by more than the paired-bootstrap interval. Within that margin,
    prefer the simpler model:
        prevalence < logistic < boosting < transformer < ensemble
    Tie-break on macro AUROC.

The paired bootstrap is what decides, not the point estimates. Both models
score the same patients, so resampling them together cancels the shared
patient-level noise and the interval on the *difference* is far tighter than
either marginal interval. Overlapping marginal intervals do not imply a
non-significant difference, which is exactly the mistake this avoids.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data.examples import ExampleConfig
from .data.labels import at_risk_mask, load_labels
from .evaluate import macro_ap, macro_auroc, paired_bootstrap_delta

__all__ = ["main"]

# Lower index = simpler. Used only to break ties inside the noise band.
SIMPLICITY = ["prevalence", "lr", "gbdt", "P4", "P3", "P1", "P2", "ensemble"]


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def collect(root: Path) -> dict[str, np.ndarray]:
    """Validation predictions for every model that left an artifact behind."""
    out: dict[str, np.ndarray] = {}
    npz = root / "artifacts" / "baseline_val_preds.npz"
    if npz.exists():
        d = np.load(npz)
        for k in d.files:
            out[k] = d[k]
    for f in sorted((root / "artifacts").glob("val_preds_*.npy")):
        out[f.stem.replace("val_preds_", "")] = np.load(f)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args(argv)
    root = Path(args.root)

    lab = load_labels(root, ExampleConfig())
    y = lab["y"].astype(int)
    ar = at_risk_mask(lab)
    val = lab["split"] == "val"
    yv, arv = y[val], ar[val]

    preds = collect(root)
    preds = {k: v for k, v in preds.items() if v.shape == yv.shape}
    if not preds:
        print("no validation predictions found -- run the baseline and the "
              "transformer arms first")
        return

    # Best feature model + best sequence model, logit-averaged. Built here
    # rather than during training so it always pairs the arms that actually ran.
    feat = [k for k in preds if k in ("lr", "gbdt")]
    seq = [k for k in preds if k.startswith("P")]
    if feat and seq:
        # Score candidates through the SAME at-risk mask the table uses.
        # Ranking on raw predictions here while the table ranks on masked ones
        # paired the ensemble with the second-best sequence arm.
        def masked_ap(k: str) -> float:
            return macro_ap(yv, np.where(arv.astype(bool), preds[k], 1e-6))[0]

        bf = max(feat, key=masked_ap)
        bs = max(seq, key=masked_ap)
        preds["ensemble"] = _sigmoid(0.5 * (_logit(preds[bf]) + _logit(preds[bs])))
        print(f"ensemble = logit-average of {bf} and {bs}\n")

    # The at-risk rule is deterministic and needs no model, so "mask only" is
    # a real baseline and worth its own row: it is the floor any model has to
    # clear before its own predictions have contributed anything.
    if "prevalence" in preds:
        preds["mask_only"] = preds.pop("prevalence")

    rows = []
    for name, P in preds.items():
        Pm = np.where(arv.astype(bool), P, 1e-6)
        a, n = macro_auroc(yv, Pm)
        p, _ = macro_ap(yv, Pm)
        rows.append((name, a, p, n))

    # Unmasked prevalence is not a model, it is a test of the metric code: it
    # must score exactly 0.5, and if it does not, nothing else on the page
    # means anything.
    base = np.tile(yv.mean(0), (len(yv), 1))
    sanity, _ = macro_auroc(yv, base)
    assert abs(sanity - 0.5) < 1e-9, f"metric code is broken: {sanity}"
    print(f"SANITY  prevalence-only, unmasked = {sanity:.6f}  (must be 0.5)\n")

    rows.sort(key=lambda r: -r[2])

    print(f"{'model':<14} {'macroAUROC':>11} {'macroAP':>9} {'labels':>7}")
    for name, a, p, n in rows:
        print(f"{name:<14} {a:>11.4f} {p:>9.4f} {n:>7}")

    if len(rows) < 2:
        return
    win, run = rows[0], rows[1]
    d = paired_bootstrap_delta(
        yv, np.where(arv.astype(bool), preds[win[0]], 1e-6),
        np.where(arv.astype(bool), preds[run[0]], 1e-6), n_boot=args.n_boot)
    pt, lo, hi, _ = d["d_macro_ap"]
    resolved = lo > 0 or hi < 0

    print(f"\npaired bootstrap, {win[0]} - {run[0]}:")
    print(f"  d macro AP {pt:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
          f"{'RESOLVED' if resolved else 'NOT RESOLVABLE'}")

    if resolved:
        choice = win[0]
        why = f"beats {run[0]} by more than the paired-bootstrap interval"
    else:
        cand = [win[0], run[0]]
        def rank(k: str) -> int:
            # Names carry a seed suffix ("P4_seed0"); match on the arm.
            base = k.split("_")[0]
            return SIMPLICITY.index(base) if base in SIMPLICITY else 99

        choice = min(cand, key=rank)
        why = (f"inside the noise band, so the simpler of {cand} is taken; "
               "the difference is not measurable on 365 patients")
    print(f"\nSELECTED: {choice}\n  {why}")


if __name__ == "__main__":
    main()
