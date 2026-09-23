"""A1: is the transformer step-starved, and is the stopping rule confounding us?

**The finding this tests.** Every one of the 16 design corners was stopped by
`patience`; not one reached the 30-epoch cap. And training length is confounded
with exactly one factor:

    lr high  -> mean best epoch  8.6      lr low -> 15.9    diff -7.2, p = 0.001
    weight_decay / capacity / dropout / fusion:  |diff| <= 1.0, p >= 0.288

The mechanism is not subtle once `min_delta` is compared against the noise it is
supposed to sit above. `min_delta = 0.002` against a measured epoch-to-epoch
sigma of **0.0157** is a band of **0.13 sigma**, so "improvement" is decided by
noise. A higher learning rate jumps further in the first few epochs, sets a
higher running best, then cannot beat it -- and stops sooner. `min_delta`'s own
docstring says the band "should be set from the measured epoch-to-epoch noise in
the metric"; that noise was measured afterwards and the band was never updated.

So the design's **lr main effect (-0.0003) is uninterpretable, not null**: it
conflates "higher lr per step" with "46% fewer steps". The other four factors are
balanced and survive.

This is D8 ("the fixed epoch budget inverted search conclusions") and E19.4
("arms trained for very different lengths") recurring *inside* the design that
everything now rests on.

**The second question, same experiment.** The chosen configuration stops at epoch
7. At 2,433 patients and batch 32 that is 77 steps per epoch, so the whole run is
**539 optimizer steps** for a ~334k-parameter model. That is few enough that
"converged" is not obviously the right word, and if the model is merely
step-starved then the cure is more steps, not more architecture.

**Why one arm and not two.** A 30-epoch run with patience disabled *contains* the
patience-stopped run as a prefix: same seed, same initialisation, same batch
order, same data. So the patience rule can be replayed on the trajectory
afterwards rather than run separately. That makes the comparison exactly paired --
it is literally the same run -- and costs nothing instead of doubling the bill.
The replay is `_replay_patience`, and it is unit-checked against trajectories
whose answer is known by construction rather than trusted.

Run: ``python -u -m src.protocol --seeds 110 111``
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

LEDGER = Path("outputs/protocol_ledger.jsonl")

# Config (1), the design's predicted-best cell, exactly as design_confirm.csv ran
# it. Held fixed here: this experiment varies the PROTOCOL and nothing else.
CONFIG = dict(n_layer=1, n_embd=64, n_head=2, lr=0.0036, weight_decay=0.001,
              dropout=0.0, fusion="readout")
BUDGET = 30          # must exceed the longest observed design run (21 epochs)
BATCH = 32
N_FIT = 2433
SIGMA_EPOCH = 0.0157


def _replay_patience(ap, patience: int = 4, min_delta: float = 0.002) -> dict:
    """What the design's stopping rule WOULD have picked on this trajectory.

    Replicates `train()`'s rule exactly: `improved = ap > best + min_delta`, and
    the break when `(epoch + 1) - best_epoch >= patience`. Kept as a separate
    pure function so it can be checked against trajectories whose answer is known
    by construction, rather than trusted because it looks right.
    """
    best_ap, best_ep, stop_ep = -1.0, 0, len(ap)
    for i, v in enumerate(ap, start=1):
        if v > best_ap + min_delta:
            best_ap, best_ep = v, i
        if patience > 0 and i - best_ep >= patience:
            stop_ep = i
            break
    return {"stopped_at": stop_ep, "picked_epoch": best_ep, "picked_ap": best_ap}


def run(root: str = ".", arm: str = "P4", seeds=(110, 111)) -> pd.DataFrame:
    from .data.examples import ExampleConfig
    from .data.sequences import SeqConfig
    from .train_finetune import TrainConfig, train

    spe = int(np.ceil(N_FIT / BATCH))
    print("A1: fixed budget vs patience, at the design's predicted-best cell")
    print("  %s" % CONFIG)
    print("  budget %d epochs = %d steps (the patience runs stopped at 7 = %d)"
          % (BUDGET, BUDGET * spe, 7 * spe), flush=True)

    rows = []
    for seed in seeds:
        t = time.time()
        # patience=0 disables the early stop entirely; min_delta=0.0 so `best`
        # tracks the true argmax rather than a band-filtered one. Both sit
        # OUTSIDE the resume fingerprint, so this run could in principle collide
        # with the confirm run at the same seed. Verified safe: those finished
        # cleanly and deleted their checkpoints, and a ckpt_last_*.pt on disk is
        # exactly the signal that a run died (D26).
        cfg = TrainConfig(seed=seed, dev_frac=0.0, holdout_frac=0.0,
                          epochs=BUDGET, patience=0, min_delta=0.0,
                          n_layer=CONFIG["n_layer"], n_embd=CONFIG["n_embd"],
                          n_head=CONFIG["n_head"], lr=CONFIG["lr"],
                          weight_decay=CONFIG["weight_decay"],
                          attn_dropout=CONFIG["dropout"],
                          resid_dropout=CONFIG["dropout"],
                          fusion=CONFIG["fusion"])
        res = train(arm, root, cfg, ExampleConfig(), SeqConfig(), verbose=True)
        row = {"seed": seed, "budget": BUDGET, "steps": BUDGET * spe,
               "argmax_epoch": res["epoch"], "argmax_ap": res["macro_ap"],
               "argmax_auroc": res["macro_auroc"],
               "minutes": (time.time() - t) / 60}
        if res.get("preds") is not None:
            Path("artifacts").mkdir(exist_ok=True)
            np.save("artifacts/protocol_preds_s%d.npy" % seed, res["preds"])
        rows.append(row)
        print("  seed %d done in %.1f min" % (seed, row["minutes"]), flush=True)
        pd.DataFrame(rows).to_csv("outputs/protocol_runs.csv", index=False)
        LEDGER.parent.mkdir(exist_ok=True)
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict({"event": "protocol_run"}, **row)) + "\n")
    return pd.DataFrame(rows)


def trajectories(seeds) -> dict:
    """Per-epoch metrics for these seeds, read back out of metrics.jsonl.

    The trajectory is the whole point of the experiment and `train()` returns
    only the selected epoch, so it is recovered from the log rather than
    re-plumbed through the return value. Runs are segmented by `init` events and
    matched on (epochs == BUDGET, seed), never by the `run` string: those are
    reused heavily -- 39 distinct strings across 115 launches -- and matching on
    them would silently merge different runs.
    """
    out, cur = {}, None
    p = Path("outputs/metrics.jsonl")
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("event") == "init":
            c = r.get("config", {}) or {}
            cur = c.get("seed") if c.get("epochs") == BUDGET and \
                c.get("patience") == 0 else None
            if cur is not None:
                out[cur] = []
        elif cur is not None and "epoch" in r and "macro_ap" in r:
            out[cur].append((r["epoch"], r["macro_ap"], r["macro_auroc"],
                             r.get("ema_macro_ap", float("nan"))))
    return {s: sorted(v) for s, v in out.items() if s in seeds and v}


def report(seeds) -> None:
    tr = trajectories(seeds)
    if not tr:
        print("no trajectories found for %s -- has the run finished?" % sorted(seeds))
        return
    spe = int(np.ceil(N_FIT / BATCH))
    deltas = []
    print("\n=== A1: does training past the patience stop help? ===")
    for s, v in sorted(tr.items()):
        ep = [a for a, _, _, _ in v]
        ap = [b for _, b, _, _ in v]
        au = [c for _, _, c, _ in v]
        em = [d for _, _, _, d in v]
        rep = _replay_patience(ap)
        d = ap[-1] - rep["picked_ap"]
        deltas.append(d)
        print("\n  seed %d  (%d epochs logged = %d steps)"
              % (s, len(ap), len(ap) * spe))
        print("    patience WOULD have stopped at epoch %d, picking epoch %d"
              " -> AP %.4f" % (rep["stopped_at"], rep["picked_epoch"],
                               rep["picked_ap"]))
        print("    full budget, LAST epoch %d -> AP %.4f / AUROC %.4f"
              % (ep[-1], ap[-1], au[-1]))
        print("    full budget, argmax epoch %d -> AP %.4f"
              % (ep[int(np.argmax(ap))], max(ap)))
        if np.isfinite(em[-1]):
            print("    full budget, EMA at last epoch -> AP %.4f" % em[-1])
        print("    DELTA (full-budget last - patience pick) = %+.4f  (%.2f sigma)"
              % (d, d / SIGMA_EPOCH))
        k = max(3, len(ap) // 3)
        sl = float(np.polyfit(np.arange(k), ap[-k:], 1)[0])
        print("    slope over last %d epochs: %+.5f AP/epoch = %.2f sigma across"
              " those %d" % (k, sl, sl * k / SIGMA_EPOCH, k))
    if deltas:
        m = float(np.mean(deltas))
        print("\n  mean DELTA over %d seeds: %+.4f (%.2f sigma_epoch)"
              % (len(deltas), m, m / SIGMA_EPOCH))
    print("\n  Reading. A positive DELTA means patience was stopping the model")
    print("  early, and the design's lr effect is confounded with training")
    print("  budget rather than measuring lr. A flat slope at the end means the")
    print("  model is converged and step starvation is NOT the explanation --")
    print("  which is equally a result, and kills B0 (augmentation) with it,")
    print("  because augmentation's main channel here is 6.6x more steps.")


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--seeds", type=int, nargs="+", default=[110, 111])
    # The decisive test for E25. Config (1) sits at lr HIGH (basin*3), and
    # patience stopped it at the true argmax -- so patience is not truncating
    # THERE. The confound claim stands or falls on lr LOW (basin/3 = 4e-4),
    # where the design ran to a mean epoch of 15.9: if that also peaks where
    # patience stopped it, patience is unbiased across lr and E25 is withdrawn.
    ap_.add_argument("--lr", type=float, default=None,
                     help="override the learning rate (4e-4 = the design low level)")
    ap_.add_argument("--report-only", action="store_true")
    a = ap_.parse_args()
    if a.lr is not None:
        CONFIG["lr"] = a.lr
        print("lr overridden to %.1e (design low level is 4e-4)" % a.lr)
    if not a.report_only:
        run(".", seeds=tuple(a.seeds))
    report(set(a.seeds))


if __name__ == "__main__":
    main()
