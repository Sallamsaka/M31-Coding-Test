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
# patience, not a hard budget: the arms differ architecturally
# (bidirectional vs causal, time signals on or off) and those converge at
# different rates, so a fixed cap would compare a finished arm against an
# unfinished one and charge the difference to the ablated component.
# min_delta=0.002 is the patience BAND, set from measurement rather than
# taste: the observed epoch-to-epoch AP increments on the collapse run's
# plateau were +0.0010 and +0.0033, so 0.002 sits between them and filters
# pure wobble while still admitting a real gain. Provisional -- it should be
# re-derived from the sigma_ap that `design --seeds` measures.
BASE = dict(n_layer=2, n_embd=128, n_head=4, epochs=30, patience=4,
            min_delta=0.002, dev_frac=0.2, holdout_frac=0.1,
            eval_on="dev", seed=0)

ARMS: dict[str, dict] = {
    # `block_size` is a SeqConfig field, not a TrainConfig one; run_arm pulls it
    # out and builds the sequence pack with it.
    #
    # ctx256 tests the PREMISE behind the event-vs-day question rather than
    # building the fix for it. Truncation keeps the most recent tokens, so at
    # block 512 we discard 32.2% of all events -- the OLDEST history of the
    # LONGEST records, i.e. the most comorbid patients. Day-pooling would
    # recover that (5.05x compression), but it is a real modelling change, and
    # it is only worth building if the discarded history matters.
    #
    # Halving the window is strictly CHEAPER than the baseline and bounds the
    # answer monotonically: if 256 ~ 512, older context has sharply diminishing
    # value, 512 -> more would buy little, and day-pooling can be dropped on
    # MEASURED grounds instead of on a literature argument. If 256 << 512, the
    # 32% we are throwing away probably matters and day-pooling (or a larger
    # block) earns its implementation cost.
    #
    # Deliberately testing downward: at block 1024 the attention bias tensor is
    # (B, n_head, T, T) = 1.2 GB at batch 32, against a measured ~2 GB safe
    # budget on this machine (V10). Upward needs token-budget batching first.
    "ctx256":          dict(arm="P4", use_time_encoding=True,  use_dt_bias=True,
                            block_size=256),
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
    block = spec.pop("block_size", None)
    seq_cfg = SeqConfig(block_size=block) if block else SeqConfig()
    cfg = TrainConfig(**BASE, **spec)
    t0 = time.time()
    r = train(arm, root, cfg, ExampleConfig(), seq_cfg, verbose=False)
    out = {"ablation": name, "base_arm": arm, "block_size": seq_cfg.block_size,
           **spec,
           "macro_auroc": r["macro_auroc"], "macro_ap": r["macro_ap"],
           "hold_macro_auroc": r.get("holdout_macro_auroc", float("nan")),
           "hold_macro_ap": r.get("holdout_macro_ap", float("nan")),
           "epoch": r["epoch"], "minutes": (time.time() - t0) / 60}
    if verbose:
        print(f"  {name:<17} AUROC {out['macro_auroc']:.4f}  AP {out['macro_ap']:.4f}"
              f"  ep{out['epoch']:<3} [{out['minutes']:.1f} min]", flush=True)
    LEDGER.parent.mkdir(exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(out) + "\n")
    return out


def _sigma_single_run():
    """Measured seed sigma if the replicates have run, else evaluation noise."""
    led = Path("outputs/design_ledger.jsonl")
    if led.exists():
        hit = None
        for line in led.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("event") == "seed_sigma":
                    hit = r
        if hit:
            return float(hit["sigma_auroc"]), (
                "MEASURED over %d seeds" % hit["n_seeds"])
    # Evaluation noise only, scaled from the measured 0.0104 at n=365. It
    # excludes seed and optimisation variance, so it is a LOWER BOUND.
    return 0.0104 * np.sqrt(365 / 558), "evaluation noise only -- a LOWER BOUND"


def report(rows: list[dict]) -> None:
    df = pd.DataFrame(rows).set_index("ablation")
    print("\n=== TIME ABLATION ===")
    print(df[["macro_auroc", "macro_ap", "epoch", "minutes"]]
          .to_string(float_format=lambda v: f"{v:.4f}"))

    # sigma of a SINGLE run. Prefer the measured seed sigma when the seed
    # replicates have run; otherwise fall back to evaluation noise scaled to
    # the dev split, and SAY SO -- evaluation noise excludes seed and
    # optimisation variance, so the fallback is a lower bound and every
    # verdict computed under it is optimistic.
    sigma, src = _sigma_single_run()

    # These arms are one-run-vs-one-run, so the relevant quantity is the SE
    # of a DIFFERENCE of two independent runs, sigma*sqrt(2) -- NOT sigma.
    # Dividing by sigma overstates every verdict by 41%.
    se_diff = sigma * np.sqrt(2)
    print(f"\n  sigma(single run) ~ {sigma:.4f}  ({src})")
    print(f"  SE of an arm-vs-arm difference = sigma*sqrt(2) = {se_diff:.4f}")
    print(f"  a contrast needs ~{2*se_diff:.4f} to resolve; 80%-power MDE "
          f"= {2.8*se_diff:.4f}")

    def gap(a, b, col="macro_auroc"):
        if a not in df.index or b not in df.index:
            return None
        d = df.loc[a, col] - df.loc[b, col]
        return d, d / se_diff

    print("\n  contrasts that answer the design claim:")
    for a, b, what in (
        ("full", "no_dt_bias", "the Δt attention bias is worth"),
        ("full", "no_time2vec", "the per-token Time2Vec is worth"),
        ("full", "no_time_signals", "both explicit time signals are worth"),
        ("bidir_mean", "multiset", "time within the order-blind shape is worth"),
        ("full", "multiset", "the whole time apparatus is worth"),
        ("full", "ctx256", "doubling the context window 256->512 is worth"),
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
