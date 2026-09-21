"""Staged transformer search, tuned on an inner split of train.

Run: ``python -m src.search --stage A``   (or ``all``)

Why staged rather than one big grid
-----------------------------------
A full grid over capacity x regularisation x width x fusion x pretraining is
~100 configurations. At 14 minutes each that is 23 hours, and -- far worse --
scoring 100 candidates on one held-out set means the winner is chosen by
noise: ``sigma*sqrt(2 ln 100)`` is 3.0 sigma of pure optimism.

So the search is staged, and the staging follows which factors actually
interact rather than being an arbitrary ordering:

* **Capacity and regularisation are searched JOINTLY** (stage A). Depth and
  dropout are the same knob viewed twice -- both trade bias against variance
  -- and optimising one at a fixed value of the other finds the optimum of
  neither. This is the only factorial stage, and it is factorial because the
  interaction is the point.
* **Width, fusion and pretraining are searched greedily** on the stage-A
  winner. They are closer to separable, and each is cheap.
* **Stage E re-tests the stage-A decision at the final configuration.** Greedy
  descent is only valid if early choices survive later ones; this is the one
  check that catches it when they do not. One run.

A known confound, recorded rather than fixed
--------------------------------------------
Every config gets the same 20-epoch budget, but smaller models converge
later: measured, the 1-layer runs peaked at epoch 19 and 20 of 20 -- still
improving when the budget ran out -- while the winning 2-layer/dropout-0.3
run peaked at 14. **The 1-layer scores are therefore a lower bound**, and
this search cannot rule out that one layer would catch up given more epochs.
Equalising by compute rather than by epochs would be the cleaner design; it
was not worth restarting the search for, because 1 layer trails by 0.017 AP
and the gap would have to close entirely.

Everything scores on a **held-out 20% of TRAIN patients**, grouped so a
patient's cutoffs never straddle the split. The real validation set is read
exactly once, in stage F, by the winner -- which is the only way a search
this size leaves the headline number meaning anything.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .data.sequences import SeqConfig
from .train_finetune import ARMS, TrainConfig, train

__all__ = ["main", "LEDGER"]

LEDGER = Path("outputs/search_ledger.jsonl")

# Screening size: 2 layers / d=128 runs 20 epochs in ~14 min against ~63 min
# for 4L/d192, and the EHR literature puts the depth optimum at 1-2 layers,
# so this is a plausible configuration in its own right rather than only a
# cheap proxy. The winner is re-confirmed at full width in stage F.
# `epochs` is now a CEILING, not a budget: patience stops each config when it
# stops improving, so a slow-converging model is not compared against a
# converged one. The first search hit this -- 1-layer runs peaked at epoch 20
# of 20 while the winner peaked at 14 -- and was restarted because of it.
SCREEN = dict(n_layer=2, n_embd=128, n_head=4, epochs=45, patience=7,
              dev_frac=0.2)


def _run(tag: str, arm: str, **over) -> dict:
    cfg = TrainConfig(**{**SCREEN, **over})
    t0 = time.time()
    r = train(arm, ".", cfg, None, SeqConfig(), verbose=False)
    rec = {
        "tag": tag, "arm": arm, "minutes": round((time.time() - t0) / 60, 1),
        "dev_macro_auroc": round(r["macro_auroc"], 4),
        "dev_macro_ap": round(r["macro_ap"], 4),
        "best_epoch": r["epoch"], "epochs_run": r.get("epochs_run"),
        **{k: v for k, v in over.items()},
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"  {tag:<34} AUROC {rec['dev_macro_auroc']:.4f}  "
          f"AP {rec['dev_macro_ap']:.4f}  "
          f"ep{rec['best_epoch']:>3}/{rec.get('epochs_run','?')}  "
          f"[{rec['minutes']:.0f}m]", flush=True)
    return rec


def _best(recs: list[dict]) -> dict:
    return max(recs, key=lambda r: r["dev_macro_ap"])


def stage_a() -> list[dict]:
    """Depth x dropout, jointly. The only factorial stage."""
    print("\n=== STAGE A: capacity x regularisation (factorial) ===", flush=True)
    out = []
    for n_layer in (1, 2, 4):
        for drop in (0.1, 0.3):
            out.append(_run(f"A n_layer={n_layer} drop={drop}", "P4",
                            n_layer=n_layer, attn_dropout=drop,
                            resid_dropout=drop))
    b = _best(out)
    print(f"  -> best: {b['tag']}", flush=True)
    return out


def stage_b(base: dict) -> list[dict]:
    """Width, greedy on the stage-A winner."""
    print("\n=== STAGE B: width ===", flush=True)
    keep = dict(n_layer=base["n_layer"], attn_dropout=base["attn_dropout"],
                resid_dropout=base["resid_dropout"])
    out = []
    for n_embd, n_head in ((96, 4), (192, 6)):
        out.append(_run(f"B n_embd={n_embd}", "P4", n_embd=n_embd,
                        n_head=n_head, **keep))
    return out


def stage_c(base: dict) -> list[dict]:
    """Feature fusion: none (already have it) vs late vs attended."""
    print("\n=== STAGE C: feature fusion ===", flush=True)
    keep = {k: base[k] for k in ("n_layer", "attn_dropout", "resid_dropout")
            if k in base}
    if "n_embd" in base:
        keep["n_embd"] = base["n_embd"]
        keep["n_head"] = base.get("n_head", 4)
    out = []
    for mode in ("readout", "token"):
        out.append(_run(f"C fusion={mode}", "P4", fusion=mode, **keep))
    return out


def stage_d(base: dict) -> list[dict]:
    """Pretraining, on whichever mask the winner uses."""
    print("\n=== STAGE D: pretraining ===", flush=True)
    keep = {k: v for k, v in base.items()
            if k in ("n_layer", "n_embd", "n_head", "attn_dropout",
                     "resid_dropout", "fusion")}
    return [_run("D pretrain=lm", "P1", pretrain_epochs=4, **keep)]


def stage_e(base: dict, a_recs: list[dict]) -> list[dict]:
    """Re-test stage A's decision at the final configuration.

    Greedy search assumes an early choice stays optimal once everything
    downstream has changed. That assumption is usually stated and rarely
    checked. One run checks it: take the runner-up depth from stage A and
    re-run it with every later decision in place. If it now wins, the path
    was wrong and the report says so.
    """
    print("\n=== STAGE E: interaction re-test ===", flush=True)
    order = sorted(a_recs, key=lambda r: -r["dev_macro_ap"])
    runner_up = next((r for r in order if r["n_layer"] != base["n_layer"]), None)
    if runner_up is None:
        print("  no alternative depth to re-test")
        return []
    keep = {k: v for k, v in base.items()
            if k in ("n_embd", "n_head", "attn_dropout", "resid_dropout",
                     "fusion")}
    return [_run(f"E n_layer={runner_up['n_layer']} @ final", "P4",
                 n_layer=runner_up["n_layer"], **keep)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["A", "B", "C", "D", "E", "all"])
    args = ap.parse_args(argv)

    print("Tuning on an inner 20% split of TRAIN. Validation is not read.")
    recs: list[dict] = []
    a = stage_a() if args.stage in ("A", "all") else []
    recs += a
    if args.stage == "A":
        return

    base = _best(recs) if recs else dict(SCREEN)
    for stage, fn in (("B", stage_b), ("C", stage_c), ("D", stage_d)):
        if args.stage in (stage, "all"):
            got = fn(base)
            recs += got
            base = _best(recs)
            print(f"  running best: {base['tag']}", flush=True)
    if args.stage in ("E", "all") and a:
        recs += stage_e(base, a)

    print("\n=== SEARCH COMPLETE ===")
    for r in sorted(recs, key=lambda r: -r["dev_macro_ap"]):
        print(f"  {r['dev_macro_ap']:.4f}  {r['dev_macro_auroc']:.4f}  {r['tag']}")
    print(f"\nledger -> {LEDGER}  ({len(recs)} configs)")
    print("Nothing here has touched the validation set. Confirm the winner "
          "with seeds before believing any of it.")


if __name__ == "__main__":
    main()
