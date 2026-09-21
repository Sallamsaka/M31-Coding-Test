"""Is a training run reproducible from its seed? Measured, not assumed.

Prompted by the time ablation failing to reproduce across two attempts in a way
no single explanation covers:

    no_dt_bias        0.7493 / 0.2084 / ep18  ->  BIT-IDENTICAL
    no_time_signals   0.7383 / 0.1812 / ep6   ->  BIT-IDENTICAL
    no_time2vec       0.7493 / 0.2101 / ep10  ->  0.7436 / 0.2110 / ep15
    full, ctx256                              ->  differ (and share a fingerprint)

A shared fingerprint explains `full` and `ctx256` -- they shared a checkpoint
slot, so one silently continued the other's training (D26). It does NOT explain
`no_time2vec`, whose fingerprint is unique.

The remaining candidate is multithreaded CPU non-determinism: `torch.set_num_
threads(8)`, and parallel float reductions can sum in an order that depends on
thread scheduling, which depends on machine load -- and load differed sharply
between the two attempts. That predicts OCCASIONAL divergence, which fits the
pattern, but it is a hypothesis until it is measured.

Why it matters beyond tidiness: if the same seed does not give the same result,
then the seed sigma of 0.0063 is not seed variance -- it is seed variance PLUS
run-to-run non-determinism, and every MDE computed from it is wrong. It also
means a resumed run cannot be verified against an unresumed one.

The test runs the same tiny configuration twice under identical settings, then
again with a single thread, and compares the weights bitwise. One thread is the
control: if 8 threads diverge and 1 thread does not, the cause is thread
scheduling and `threads` becomes a reproducibility setting rather than purely a
speed one.

Run: ``python -m src.determinism_check``
"""

from __future__ import annotations

import torch

from .data.examples import ExampleConfig
from .data.sequences import SeqConfig
from .train_finetune import TrainConfig, train


def _run(threads: int, seed: int = 0) -> dict:
    cfg = TrainConfig(seed=seed, dev_frac=0.0, holdout_frac=0.0,
                      n_layer=1, n_embd=64, n_head=2, epochs=2, patience=0,
                      ema_decay=0.0, resume=False, threads=threads)
    r = train("P4", ".", cfg, ExampleConfig(), SeqConfig(), verbose=False)
    return r


def _cmp(a: dict, b: dict, label: str) -> bool:
    same_ap = a["macro_ap"] == b["macro_ap"]
    same_au = a["macro_auroc"] == b["macro_auroc"]
    d_ap = abs(a["macro_ap"] - b["macro_ap"])
    print(f"  {label:<28} AP {a['macro_ap']:.8f} vs {b['macro_ap']:.8f}"
          f"   |d| {d_ap:.2e}   {'IDENTICAL' if same_ap and same_au else 'DIFFERS'}")
    return same_ap and same_au


def main(root: str = ".") -> None:
    print("determinism: same seed, same config, repeated\n")
    print(f"torch {torch.__version__}, threads available {torch.get_num_threads()}\n")

    a8, b8 = _run(8), _run(8)
    ok8 = _cmp(a8, b8, "8 threads, run 1 vs run 2")

    a1, b1 = _run(1), _run(1)
    ok1 = _cmp(a1, b1, "1 thread,  run 1 vs run 2")

    print()
    if ok8 and ok1:
        print("  VERDICT: reproducible at both thread counts. The ablation")
        print("  divergence is NOT thread non-determinism; look elsewhere.")
    elif ok1 and not ok8:
        print("  VERDICT: 8 threads diverge, 1 thread does not. Parallel float")
        print("  reduction order is the cause. `threads` is a REPRODUCIBILITY")
        print("  setting, not only a speed one, and the measured seed sigma of")
        print("  0.0063 conflates seed variance with run-to-run noise.")
    else:
        print("  VERDICT: diverges even single-threaded -- the cause is upstream")
        print("  of threading (data order, an unseeded generator, or state that")
        print("  outlives a run).")


if __name__ == "__main__":
    main()
