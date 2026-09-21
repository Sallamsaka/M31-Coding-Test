"""Does the time signal earn its place? The design-claim experiment.

`docs/01-why-time.md` argues formally that without a positional signal a
transformer sees only a multiset of codes. That argument is the intellectual
centre of this submission and it had never been measured, because until the
ablation switches were added the model had no way to turn the time signals off.
§W7 ranks this above the hyperparameter search for exactly that reason: Lipton &
Steinhardt's "failure to identify sources of empirical gains" applies directly
when the headline claim is architectural.

**The arms, and why there are six rather than the four originally planned.**

| arm | mask | readout | Time2Vec | Δt bias | what it isolates |
|---|---|---|---|---|---|
| `full` | causal | last | on | on | the shipped model |
| `no_dt_bias` | causal | last | on | **off** | graded distance |
| `no_time2vec` | causal | last | **off** | on | per-token time |
| `no_time_signals` | causal | last | **off** | **off** | both, mask retained |
| `bidir_mean` | **bidir** | **mean** | on | on | control for `multiset` |
| `multiset` | **bidir** | **mean** | **off** | **off** | the theoretical floor |

**`no_time_signals` is NOT the multiset baseline, and treating it as one was the
error in the original four-arm plan.** The visibility mask is derived from `dt`,
so a causal model still knows which event came first with both explicit time
signals removed -- permuting positions still moves its output, which
`tests/test_model.py` asserts. Only when the direction constraint and the
position-picking readout also go does the model provably become order-blind:
every position then sees every other with zero bias, and mean pooling is
symmetric. That arm is verified permutation-invariant by test.

`bidir_mean` exists so `multiset` has an honest comparator. Without it the
multiset arm differs from `full` in four things at once and any gap would be
unattributable.

**Stated limitation, because it is the most commonly violated rule in this
literature.** Each arm is run at the same learning rate. Proper ablation
discipline re-tunes nuisance hyperparameters for every variant, since an
ablation where only the full model got its own sweep measures tuning effort
rather than the component. Six arms x a 5-point sweep is 30 runs and is not
affordable here, so the result is reported as "at the shared learning rate" and
the direction of the bias is stated: it favours `full`, because the shared LR
was chosen on a configuration closest to `full`. A negative result for an
ablated arm is therefore weaker evidence than a positive one.

Run: ``python -m src.ablations``          (all six)
     ``python -m src.ablations --arms full multiset``
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

LEDGER = Path("outputs/ablation_ledger.jsonl")

# Screening capacity, matching the seed-sigma configuration so the measured
# sigma applies directly to these differences.
BASE = dict(n_layer=2, n_embd=128, n_head=4, epochs=20, dev_frac=0.2, seed=0)

ARMS: dict[str, dict] = {
    "full":            dict(arm="P4", use_time_encoding=True,  use_dt_bias=True),
    "no_dt_bias":      dict(arm="P4", use_time_encoding=True,  use_dt_bias=False),
    "no_time2vec":     dict(arm="P4", use_time_encoding=False, use_dt_bias=True),
    "no_time_signals": dict(arm="P4", use_time_encoding=False, use_dt_bias=False),
    "bidir_mean":      dict(arm="P3", use_time_encoding=True,  use_dt_bias=True),
    "multiset":        dict(arm="P3", use_time_encoding=False, use_dt_bias=False),
}


def run_arm(name: str, root: str = ".", verbose: bool = True) -> dict:
    from .data.examples import ExampleConfig
    from .data.sequences import SeqConfig
    from .train_finetune import TrainConfig, train

    spec = dict(ARMS[name])
    arm = spec.pop("arm")
    cfg = TrainConfig(**BASE, **spec)
    t0 = time.time()
    r = train(arm, root, cfg, ExampleConfig(), SeqConfig(), verbose=False)
    out = {"ablation": name, "base_arm": arm, **spec,
           "macro_auroc": r["macro_auroc"], "macro_ap": r["macro_ap"],
           "epoch": r["epoch"], "minutes": (time.time() - t0) / 60}
    if verbose:
        print(f"  {name:<17} AUROC {out['macro_auroc']:.4f}  AP {out['macro_ap']:.4f}"
              f"  ep{out['epoch']:<3} [{out['minutes']:.1f} min]", flush=True)
    LEDGER.parent.mkdir(exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(out) + "\n")
    return out


def report(rows: list[dict]) -> None:
    df = pd.DataFrame(rows).set_index("ablation")
    print("\n=== TIME ABLATION ===")
    print(df[["macro_auroc", "macro_ap", "epoch", "minutes"]]
          .to_string(float_format=lambda v: f"{v:.4f}"))

    # sigma at the 558-patient dev split, scaled from the measured 0.0104 at 365.
    sigma = 0.0104 * np.sqrt(365 / 558)
    print(f"\n  sigma(macro AUROC) at dev n=558 ~ {sigma:.4f}; a difference needs "
          f"~{2*sigma:.4f} to resolve")

    def gap(a, b, col="macro_auroc"):
        if a not in df.index or b not in df.index:
            return None
        d = df.loc[a, col] - df.loc[b, col]
        return d, d / sigma

    print("\n  contrasts that answer the design claim:")
    for a, b, what in (
        ("full", "no_dt_bias", "the Δt attention bias is worth"),
        ("full", "no_time2vec", "the per-token Time2Vec is worth"),
        ("full", "no_time_signals", "both explicit time signals are worth"),
        ("bidir_mean", "multiset", "time within the order-blind shape is worth"),
        ("full", "multiset", "the whole time apparatus is worth"),
    ):
        g = gap(a, b)
        if g is None:
            continue
        d, z = g
        verdict = "resolvable" if abs(z) >= 2 else "NOT resolvable"
        print(f"    {what:<44} {d:+.4f}  ({z:+.1f} sigma, {verdict})")

    print("\n  Read at the shared learning rate; no arm got its own sweep, which")
    print("  biases toward `full`. A negative result for an ablated arm is")
    print("  therefore weaker evidence than a positive one.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS),
                    choices=list(ARMS))
    a = ap.parse_args()
    print(f"time ablation: {len(a.arms)} arms at 2L/d128, 20 epochs, dev_frac=0.2",
          flush=True)
    rows = [run_arm(n) for n in a.arms]
    report(rows)


if __name__ == "__main__":
    main()
