"""Stage F: confirm the search winner on the real validation set, with seeds.

Run: ``python -m src.confirm --seeds 0 1 2``

This is the only place the 365-patient validation set is read by a tuned
transformer, and it is read once. Everything before it scored on an inner
split of train.

Why seeds are not optional here
-------------------------------
The search produced a winner by comparing point estimates that differ by less
than the run-to-run spread is known to be. Without measuring that spread, the
statement "configuration X beat configuration Y" has no error bar and cannot
be distinguished from "seed 0 was lucky for X". The seed standard deviation
computed here is the measuring stick for every number in the search ledger,
and if it comes out larger than the search's best-to-worst range then the
honest reading is that the search resolved nothing -- which is a result, and
is reported as one rather than buried.

The comparison against logistic regression uses a paired bootstrap on the
same patients, because that is what cancels the shared patient-level noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data.examples import ExampleConfig
from .data.labels import at_risk_mask, load_labels
from .data.sequences import SeqConfig
from .evaluate import macro_ap, macro_auroc, paired_bootstrap_delta
from .search import LEDGER
from .train_finetune import TrainConfig, train

__all__ = ["main"]

TUNABLE = ("n_layer", "n_embd", "n_head", "attn_dropout", "resid_dropout",
           "fusion", "fusion_dim")


def winner(ledger: Path = LEDGER) -> dict:
    """Best configuration by dev macro AP, read back from the search ledger."""
    rows = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit("search ledger is empty -- run `python -m src.search` first")
    best = max(rows, key=lambda r: r["dev_macro_ap"])
    spread = max(r["dev_macro_ap"] for r in rows) - min(r["dev_macro_ap"] for r in rows)
    print(f"search winner: {best['tag']}")
    print(f"  dev AP {best['dev_macro_ap']:.4f}  over {len(rows)} configs")
    print(f"  best-to-worst dev AP range across the search: {spread:.4f}")
    return best


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30,
                    help="raised from the search's 20: several configs peaked "
                         "at the final epoch, meaning they were still improving")
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args(argv)

    best = winner()
    over = {k: best[k] for k in TUNABLE if k in best}
    arm = best.get("arm", "P4")
    print(f"  confirming arm {arm} with {over}\n")

    lab = load_labels(".", ExampleConfig())
    y = lab["y"].astype(int)
    ar = at_risk_mask(lab)
    val = lab["split"] == "val"
    yv, arv = y[val], ar[val]

    runs, preds = [], []
    for seed in args.seeds:
        cfg = TrainConfig(seed=seed, epochs=args.epochs, dev_frac=0.0, **over)
        r = train(arm, ".", cfg, None, SeqConfig(), verbose=False)
        P = np.where(arv.astype(bool), r["preds"], 1e-6)
        a, _ = macro_auroc(yv, P)
        p, _ = macro_ap(yv, P)
        runs.append((seed, a, p, r["epoch"]))
        preds.append(P)
        np.save(f"artifacts/val_preds_tuned_seed{seed}.npy", r["preds"])
        print(f"  seed {seed}: AUROC {a:.4f}  AP {p:.4f}  (best epoch {r['epoch']})",
              flush=True)

    aus = np.array([r[1] for r in runs])
    aps = np.array([r[2] for r in runs])
    print(f"\ntuned transformer over {len(runs)} seeds:")
    print(f"  macro AUROC {aus.mean():.4f} +- {aus.std(ddof=1):.4f}"
          f"   (range {aus.min():.4f}-{aus.max():.4f})")
    print(f"  macro AP    {aps.mean():.4f} +- {aps.std(ddof=1):.4f}"
          f"   (range {aps.min():.4f}-{aps.max():.4f})")
    print("\n  ^ this standard deviation is the measuring stick for the whole")
    print("    search. Any config-to-config difference smaller than it was noise.")

    # Seed-averaged prediction, then the comparison that actually decides.
    mean_pred = np.mean(preds, axis=0)
    d = np.load("artifacts/baseline_val_preds.npz")
    if "lr" in d.files:
        lr = np.where(arv.astype(bool), d["lr"], 1e-6)
        a_lr, _ = macro_auroc(yv, lr)
        p_lr, _ = macro_ap(yv, lr)
        print(f"\nlogistic regression: AUROC {a_lr:.4f}  AP {p_lr:.4f}")
        for metric, key in (("AUROC", "d_macro_auroc"), ("AP", "d_macro_ap")):
            res = paired_bootstrap_delta(yv, mean_pred, lr, n_boot=args.n_boot)
            pt, lo, hi, _ = res[key]
            verdict = "RESOLVED" if (lo > 0 or hi < 0) else "not resolvable"
            print(f"  seed-averaged transformer - LR, {metric:<5} {pt:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]  {verdict}")


if __name__ == "__main__":
    main()
