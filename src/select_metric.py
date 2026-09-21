"""Which validation metric picks the model that does best on held-out data?

This is a different question from "which metric is least noisy", and the
difference is the whole point. Discriminability -- |delta| / sigma -- measures
how reliably a comparison *resolves*. It says nothing about whether the metric
points at the *better* model. A metric can separate two models sharply and
separate them on the wrong axis.

The decision-theoretic criterion instead:

    for each candidate metric M:
        pick  = argmax_model  M(model, half A)
        score = TARGET(pick, half B)
    choose the M whose picks score best on B.

``B`` informed neither the fit nor the choice, so this measures what we actually
care about: does selecting on M generalise? Regret against the oracle (the model
that was in fact best on B) is the natural summary -- it is zero when M picks
perfectly and grows as M is misled.

This is what ``TrainConfig.holdout_frac``'s own docstring said the split was
for: "choose a configuration by dev AP and by dev AUROC, then see which choice
actually wins on data that informed neither." The machinery existed; the
experiment was never run.

**Limitation, stated up front.** Four models is a thin candidate set -- with so
few, most random halves have the same winner under every metric and the
experiment has little to discriminate. It establishes the method and gives a
first reading. The designed experiment's runs are the candidate set that makes
it decisive, and it should be re-run there.

Run: ``python -m src.select_metric``
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from .data.examples import ExampleConfig
from .data.labels import at_risk_mask, load_labels
from .phase0 import per_label_panel

# Candidates to select ON. Targets to be judged BY are a subset -- we care about
# AUROC and AP because those are what the report leads with.
CANDIDATES = ["auroc", "ap", "bss", "mcc_at_top10", "sens_at_90spec", "brier"]
TARGETS = ["auroc", "ap"]
LOWER_IS_BETTER = {"brier"}
N_SPLITS = 100


def _macro(y, p, ar, codes, rows) -> dict[str, float]:
    t = per_label_panel(y[rows], p[rows], ar[rows], codes)
    s = t.dropna(subset=["auroc"])
    return {c: float(s[c].mean()) for c in CANDIDATES}


def main(root: str = ".", n_splits: int = N_SPLITS, seed: int = 0) -> None:
    lab = load_labels(root, ExampleConfig())
    val = (lab["split"] == "val").to_numpy() if hasattr(lab["split"], "to_numpy") \
        else np.asarray(lab["split"] == "val")
    y = lab["y"][val].astype(int)
    ar = at_risk_mask(lab)[val]
    codes = list(lab["codes"])

    preds: dict[str, np.ndarray] = {}
    d = np.load(f"{root}/artifacts/baseline_val_preds.npz")
    for k in d.files:
        preds[k] = d[k]
    for f in sorted(glob.glob(f"{root}/artifacts/val_preds_*.npy")):
        preds[Path(f).stem.replace("val_preds_", "")] = np.load(f)
    # The prevalence model is a constant predictor, not a candidate anyone would
    # select; including it would flatter every metric equally.
    preds = {k: v for k, v in preds.items() if v.shape == y.shape and k != "prevalence"}
    names = list(preds)
    print(f"candidate models: {names}   n_val={len(y)}   splits={n_splits}\n")

    rng = np.random.default_rng(seed)
    n = len(y)
    regret = {m: {t: [] for t in TARGETS} for m in CANDIDATES}
    agree = {m: 0 for m in CANDIDATES}
    spread = {t: [] for t in TARGETS}

    for _ in range(n_splits):
        perm = rng.permutation(n)
        A, B = perm[: n // 2], perm[n // 2:]
        mA = {k: _macro(y, p, ar, codes, A) for k, p in preds.items()}
        mB = {k: _macro(y, p, ar, codes, B) for k, p in preds.items()}
        for t in TARGETS:
            vals = [mB[k][t] for k in names]
            spread[t].append(max(vals) - min(vals))
        for m in CANDIDATES:
            sign = -1.0 if m in LOWER_IS_BETTER else 1.0
            pick = max(names, key=lambda k: sign * mA[k][m])
            for t in TARGETS:
                best = max(mB[k][t] for k in names)
                regret[m][t].append(best - mB[pick][t])
            # does selecting on M agree with selecting on the target itself?
            if pick == max(names, key=lambda k: mA[k][TARGETS[0]]):
                agree[m] += 1

    print("REGRET against the oracle, averaged over random halvings")
    print("  (0 = always picked the model that was in fact best on the held-out half)")
    print(f"{'select on':<16}{'regret@AUROC':>14}{'regret@AP':>12}{'picks = AUROC pick':>21}")
    rows = []
    for m in CANDIDATES:
        ra = float(np.mean(regret[m]["auroc"]))
        rp = float(np.mean(regret[m]["ap"]))
        rows.append((m, ra, rp, agree[m] / n_splits))
    for m, ra, rp, ag in sorted(rows, key=lambda r: r[1]):
        print(f"{m:<16}{ra:>14.5f}{rp:>12.5f}{ag*100:>20.0f}%")

    print()
    for t in TARGETS:
        print(f"  for scale: mean best-minus-worst {t} across models on a half "
              f"= {np.mean(spread[t]):.4f}")
    print("\n  A regret near that spread means the metric is picking near-randomly;")
    print("  a regret near zero means it is picking the right model almost always.")

    pd.DataFrame(rows, columns=["select_on", "regret_auroc", "regret_ap", "agrees_with_auroc"]
                 ).to_csv(f"{root}/outputs/select_metric.csv", index=False)
    print(f"\nwrote {root}/outputs/select_metric.csv")


if __name__ == "__main__":
    main()
