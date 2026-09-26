"""The transformer experiment: sigma first, then a designed factorial.

**Why not the greedy search this replaces.** `search.py` decided each stage by
comparing two single runs. SE of such a difference is sigma*sqrt(2) ~ 0.013 at
dev n=558, so its detection floor is ~0.036 -- larger than every difference
this project has ever measured (ensemble-LR 0.0061, LR-P4 0.0109, P4-P3 0.0148).
It could not resolve a single decision it was asked to make, and its own stage E
proved it: re-testing depth at the final configuration reversed stage A's
choice.

A balanced N-run design estimates each effect as an N/2 vs N/2 contrast:

    SE(effect) = 2*sigma/sqrt(N)          N=20 -> sigma/2.24 ~ 0.0047

roughly 2.8x more sensitive at equal cost, and it estimates interactions rather
than assuming all ten of them are zero.

**Seeds run first and can cancel the design.** Every SE above is conditional on
sigma, and the only sigma we have is *evaluation* noise from the validation
bootstrap -- it excludes seed and optimisation variance entirely. Required N for
80% power at effect delta is ~31*sigma^2/delta^2:

    sigma = 0.006 -> N ~ 31       sigma = 0.016 -> N ~ 80

So at the pessimistic end a 20-run design has essentially no power and should be
cut to three factors. Measure before spending five hours.

**Factors** (learning rate and weight decay were absent from the greedy search
entirely, despite being the two the literature ranks first and second):

    A learning rate      basin/3 .. basin*3   (from the 1-D sweep)
    B weight decay       1e-3 .. 0.3          (ours is 0.1, the GPT-pretraining
                                               convention; the normalised-decay
                                               rule implies ~1e-3 at our step
                                               count -- 100x apart, unmeasured)
    C capacity           1L/d128 .. 4L/d192   (depth:width ratio held fixed so
                                               this does not confound with E)
    D dropout            0.0 .. 0.35          (the largest measured stage-A main
                                               effect, 1.5 sigma on AUROC)
    E fusion             none .. readout      (gated on the collapse diagnostic)

2^(5-1) with generator E=ABCD is **resolution V**: all five main effects and all
ten two-factor interactions estimable and unaliased, in 16 runs. Plus **4
replicated centre points**, which do double duty -- pure-error degrees of freedom
estimated inside the experiment rather than assumed, and a curvature test. The
curvature test is not optional here: learning rate and weight decay both have
interior optima in log space, and a two-level contrast straddling a basin reads
**exactly zero**, which is indistinguishable from "this factor does not matter".

**Selection rule.** Report effects with intervals; act on the *shrunken*
estimate, not the argmax. Shrinkage removes the optimism of predicting the best
cell from fitted effects without a significance gate, and a gate would be wrong
here anyway -- selection and inference are different problems, and at 0.6 sigma
you still have to pick something.

Run ``python -m src.design --seeds 5`` first. The design runner is deliberately
not written yet: its shape depends on the sigma that command returns, and writing
it now would commit to the 5-factor version before knowing whether 5 factors are
affordable.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

LEDGER = Path("outputs/design_ledger.jsonl")

# Coded -1/+1 levels. `capacity` moves depth and width together so it cannot be
# confounded with the shape factor if one is added later.
FACTORS = {
    "lr":           (None, None),      # filled from the 1-D basin sweep
    "weight_decay": (1e-3, 0.3),
    # (n_layer, n_embd, n_head). Head dim is 32 at BOTH levels and at the
    # centre, so this contrast is pure capacity and not a head-dim change.
    #
    # The range was 1L/128 -> 4L/192 and is shifted DOWN, for a reason and not
    # only for cost. Delphi-2M swept 486 models and found ~2M parameters optimal
    # at 400,799 patients; running that scaling backwards to 2,791 patients
    # implies an optimum one to two orders of magnitude SMALLER than our 2.04M
    # default (§R3). §E13 points the same way -- the median label supports 10.5
    # parameters and we feed 3,320 -- and §E17 found every feature block null,
    # which is what capacity saturation looks like. So the live question is
    # "are we far too big?", and that is answered by testing downward.
    #
    # Cost mattered too, and honestly: the eight 4L/192 corners were 23 h of a
    # 29.6 h design, to test a direction the evidence already argues against.
    "capacity":     ((1, 64, 2), (2, 128, 4)),
    # One knob, driving BOTH TrainConfig.attn_dropout and .resid_dropout. They
    # are separate fields but were never varied independently, and splitting
    # them would spend a column of the design on a distinction nothing has
    # suggested matters.
    "dropout":      (0.0, 0.35),
    "fusion":       ("none", "readout"),
}


def resolution_v_design() -> np.ndarray:
    """2^(5-1), generator E = ABCD. 16 rows of +-1."""
    base = np.array(list(itertools.product([-1, 1], repeat=4)))
    e = base.prod(axis=1, keepdims=True)
    d = np.hstack([base, e])
    assert d.shape == (16, 5)
    # Resolution V check: no main effect aliased with a 2fi, and no two 2fi
    # aliased with each other. Verified rather than asserted in a comment.
    cols = {f"x{i}": d[:, i] for i in range(5)}
    for i, j in itertools.combinations(range(5), 2):
        cols[f"x{i}x{j}"] = d[:, i] * d[:, j]
    M = np.column_stack(list(cols.values()))
    G = M.T @ M / 16
    off = G - np.eye(len(cols))
    assert np.abs(off).max() < 1e-9, "design is not resolution V"
    return d


def seed_sigma(root=".", n_seeds=5, arm="P4", verbose=True) -> dict:
    """Run the same configuration N times, varying ONLY the training seed.

    This is the number every SE in this module is conditional on, and §G5 filed
    it under "ranked remainder". It is a precondition, not a remainder.

    Bouthillier et al. (MLSys 2021) find data resampling contributes more
    variance than weight initialisation, so this is a *lower bound* on the real
    run-to-run spread -- the split is held fixed here by design, so that every
    configuration in the factorial sees the same one.
    """
    from .data.examples import ExampleConfig
    from .data.sequences import SeqConfig
    from .train_finetune import TrainConfig, train

    rows = []
    for s in range(n_seeds):
        t = time.time()
        # patience, not a hard budget -- sigma must be measured under the SAME
        # stopping protocol the experiments use, or it is the wrong ruler.
        cfg = TrainConfig(seed=s, dev_frac=0.0, holdout_frac=0.0,
                          n_layer=2, n_embd=128, n_head=4,
                          epochs=30, patience=4, min_delta=0.002)
        r = train(arm, root, cfg, ExampleConfig(), SeqConfig(), verbose=False)
        if r.get("preds") is not None:
            Path("artifacts").mkdir(exist_ok=True)
            np.save(f"artifacts/seed_preds_{arm}_seed{s}.npy", r["preds"])
        rows.append({"seed": s, "auroc": r["macro_auroc"], "ap": r["macro_ap"],
                     "hold_auroc": r.get("holdout_macro_auroc", float("nan")),
                     "hold_ap": r.get("holdout_macro_ap", float("nan")),
                     "epoch": r["epoch"], "minutes": (time.time() - t) / 60})
        if verbose:
            print(f"  seed {s}: AUROC {r['macro_auroc']:.4f}  AP {r['macro_ap']:.4f}  "
                  f"ep{r['epoch']}  [{rows[-1]['minutes']:.1f} min]", flush=True)
    df = pd.DataFrame(rows)
    out = {
        "n_seeds": n_seeds,
        "sigma_auroc": float(df.auroc.std(ddof=1)),
        "sigma_ap": float(df.ap.std(ddof=1)),
        "mean_auroc": float(df.auroc.mean()),
        "mean_ap": float(df.ap.mean()),
    }
    # SE of a standard deviation estimated from n draws is ~sigma/sqrt(2(n-1)).
    out["sigma_auroc_rel_se"] = 1 / np.sqrt(2 * (n_seeds - 1))
    # The CLEAN sigma: holdout informed neither the fit nor the epoch
    # choice, so it carries no max-over-epochs inflation. If it is
    # materially larger than the dev sigma, epoch selection is doing more
    # work than the model is.
    if df.hold_auroc.notna().any():
        out["sigma_holdout_auroc"] = float(df.hold_auroc.std(ddof=1))
        out["mean_holdout_auroc"] = float(df.hold_auroc.mean())
        out["epoch_selection_inflation"] = float(
            df.auroc.mean() - df.hold_auroc.mean())
    for N in (16, 20, 32):
        out[f"mde_N{N}"] = 2.8 * 2 * out["sigma_auroc"] / np.sqrt(N)
    df.to_csv("outputs/design_seeds.csv", index=False)
    return out


def lr_basin(root=".", arm="P4", points=(1e-4, 3e-4, 6e-4, 1.2e-3, 2e-3),
             verbose=True) -> dict:
    """Locate the learning-rate basin with a 1-D sweep before the design runs.

    Required, not optional. A two-level contrast cannot see an interior optimum:
    if the low and high levels straddle a basin they score the same and the
    effect reads **exactly zero**, which is indistinguishable from "learning rate
    does not matter" -- the wrong conclusion about the factor Greff et al. (2017,
    5,400 runs, fANOVA) measure at more than two-thirds of all variance.

    Range 1e-4 to 2e-3 is the honest one from the scaling arithmetic: 6e-4 is
    GPT-3-125M's value at 0.5M tokens/batch and ours is ~6.3k, so sqrt-scaling
    implies 6.7e-5 and linear 7.5e-6, while nanoGPT-shakespeare and Delphi's own
    demo both sit near 1e-3. Two conventions, an order of magnitude apart.

    Greff's corollary licenses locating the basin at a small config and
    transferring it, so this runs at 2L/d128 rather than at full capacity.
    """
    from .data.examples import ExampleConfig
    from .data.sequences import SeqConfig
    from .train_finetune import TrainConfig, train

    rows = []
    for lr in points:
        t = time.time()
        # A FIXED EPOCH BUDGET WOULD INVALIDATE THIS ENTIRE SWEEP. Learning rate
        # is precisely the control on convergence speed, so at a hard 20-epoch
        # cap the low-lr arms are scored before they have finished and the sweep
        # measures "which lr converges fastest in 20 epochs" rather than "which
        # lr is best". The basin so found would be biased high -- and it sets
        # factor A's levels, so the bias would propagate into the whole design.
        cfg = TrainConfig(seed=0, dev_frac=0.0, holdout_frac=0.0,
                          n_layer=2, n_embd=128, n_head=4,
                          epochs=40, patience=5, min_delta=0.002, lr=lr)
        r = train(arm, root, cfg, ExampleConfig(), SeqConfig(), verbose=False)
        if r.get("preds") is not None:
            Path("artifacts").mkdir(exist_ok=True)
            np.save(f"artifacts/basin_preds_lr{lr:.0e}.npy", r["preds"])
        rows.append({"lr": lr, "auroc": r["macro_auroc"], "ap": r["macro_ap"],
                     "epoch": r["epoch"], "minutes": (time.time() - t) / 60})
        if verbose:
            print(f"  lr {lr:.1e}: AUROC {r['macro_auroc']:.4f}  AP {r['macro_ap']:.4f}"
                  f"  ep{r['epoch']}  [{rows[-1]['minutes']:.1f} min]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("outputs/design_lr_basin.csv", index=False)

    # Select on AP, not AUROC: selecting on AUROC costs 5x the regret even when
    # the target IS AUROC (E11, measured over 100 random val halvings).
    best = df.loc[df.ap.idxmax()]
    out = {"basin_lr": float(best.lr), "basin_ap": float(best.ap),
           "basin_auroc": float(best.auroc),
           "at_edge": bool(best.lr in (points[0], points[-1]))}
    if verbose:
        print(f"\n  basin at lr = {out['basin_lr']:.1e} "
              f"(AP {out['basin_ap']:.4f}, AUROC {out['basin_auroc']:.4f})")
        if out["at_edge"]:
            print("  WARNING: the optimum is at an endpoint, so the basin is NOT "
                  "bracketed.\n           Extend the range before using it to set "
                  "factor A's levels.")
        else:
            print(f"  design factor A levels: {out['basin_lr']/3:.1e} .. "
                  f"{out['basin_lr']*3:.1e}")
    return out


def report_sigma(out: dict) -> None:
    s = out["sigma_auroc"]
    print("\n=== SEED VARIANCE ===")
    print(f"  macro AUROC {out['mean_auroc']:.4f}, sigma = {s:.4f} "
          f"(+-{100*out['sigma_auroc_rel_se']:.0f}% on {out['n_seeds']} seeds)")
    print(f"  macro AP    {out['mean_ap']:.4f}, sigma = {out['sigma_ap']:.4f}")
    if "sigma_holdout_auroc" in out:
        print(f"  CLEAN (holdout, no epoch-selection inflation): "
              f"AUROC {out['mean_holdout_auroc']:.4f}, "
              f"sigma = {out['sigma_holdout_auroc']:.4f}")
        print(f"  epoch-selection inflation (dev - holdout) = "
              f"{out['epoch_selection_inflation']:+.4f}")
    print("\n  minimum detectable effect at 80% power:")
    for N in (16, 20, 32):
        print(f"    N={N:<3} -> {out[f'mde_N{N}']:.4f} macro AUROC")
    print("\n  For scale, the whole measured spread of this project is "
          "0.0061 to 0.0148.")
    if out["mde_N20"] > 0.02:
        print("  VERDICT: a 20-run design detects nothing this project cares about.")
        print("           Cut to 3 factors, or accept 'not resolvable' as the result.")
    else:
        print("  VERDICT: a 20-run design is worth running.")


# ---------------------------------------------------------------------------
# The design itself.
# ---------------------------------------------------------------------------

# Centre point at coded 0. `lr` and `weight_decay` are coded in LOG space, so
# their centre is the geometric mean of the two levels. `fusion` is genuinely
# categorical and has no centre, so centre runs sit at the reference level --
# which makes the curvature test valid only within the fusion="none" half.
# Stated rather than buried; it is the price of putting a categorical factor
# into a design that carries centre points.
CENTRE_CAPACITY = (1, 96, 3)      # n_embd / n_head = 32, as at both corners


def _read_ledger(event):
    """Last record of `event` in the ledger, or None."""
    if not LEDGER.exists():
        return None
    hit = None
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("event") == event:
                hit = row
    return hit


def design_runs(basin_lr):
    """16 resolution-V corners plus 4 replicated centre points."""
    d = resolution_v_design()
    lo_hi = dict(FACTORS)
    lo_hi["lr"] = (basin_lr / 3, basin_lr * 3)

    runs = []
    for i, row in enumerate(d):
        pick = {f: lo_hi[f][0] if c < 0 else lo_hi[f][1]
                for f, c in zip(FACTORS, row)}
        n_layer, n_embd, n_head = pick.pop("capacity")
        runs.append({"run": i, "kind": "corner", "n_layer": n_layer,
                     "n_embd": n_embd, "n_head": n_head, **pick,
                     **{"c_" + f: int(c) for f, c in zip(FACTORS, row)}})

    n_layer, n_embd, n_head = CENTRE_CAPACITY
    wd_lo, wd_hi = FACTORS["weight_decay"]
    for j in range(4):
        runs.append({"run": 16 + j, "kind": "centre", "n_layer": n_layer,
                     "n_embd": n_embd, "n_head": n_head,
                     "lr": basin_lr,
                     "weight_decay": float(np.sqrt(wd_lo * wd_hi)),
                     "dropout": float(np.mean(FACTORS["dropout"])),
                     "fusion": "none",
                     **{"c_" + f: 0 for f in FACTORS}})
    return runs


def run_design(root=".", arm="P4", verbose=True):
    from .data.examples import ExampleConfig
    from .data.sequences import SeqConfig
    from .train_finetune import TrainConfig, train

    basin = _read_ledger("lr_basin")
    sig = _read_ledger("seed_sigma")
    if basin is None:
        raise SystemExit("no lr_basin in the ledger -- run `--lr-basin` first")
    basin_lr = float(basin["basin_lr"])
    runs = design_runs(basin_lr)

    # RANDOMISE THE RUN ORDER. Standard DOE practice against time trends, but
    # the binding reason here is cruder: this is an ~8-hour serial job on a
    # laptop. In design-matrix order the first 8 runs are ALL lr-low, so a crash
    # or an interrupt at run 10 leaves a set with no lr-high runs at all and the
    # lr contrast is unestimable. Randomised, any prefix is roughly balanced in
    # every factor, so a partial design still yields usable (wider) effects --
    # and design_runs.csv is written after every run precisely so a prefix is
    # worth having.
    order = np.random.default_rng(20250921).permutation(len(runs))
    runs = [runs[i] for i in order]

    print("design: %d runs (16 corners + 4 centre), basin lr = %.1e"
          % (len(runs), basin_lr), flush=True)
    if sig:
        s = float(sig["sigma_auroc"])
        se = 2 * s / np.sqrt(len(runs))
        print("  measured sigma = %.4f -> SE(effect) = %.4f, MDE(80%%) = %.4f"
              % (s, se, 2.8 * se), flush=True)
        print("  E14: sigma sets N, not the factor count. All five factors run;"
              " 'not resolvable' is a reportable outcome.", flush=True)
    else:
        print("  WARNING: no seed_sigma in the ledger, so the effects below have"
              " no measured noise scale to be judged against.", flush=True)

    rows = []
    for r in runs:
        t = time.time()
        # The training seed VARIES with the run (Bouthillier et al. 2021): a
        # fixed seed makes sigma cosmetically small and the conclusions
        # seed-specific. The SPLIT seed 12345 stays fixed -- a different thing.
        # patience > 0, NOT a hard epoch budget. TrainConfig.patience's own
        # docstring records why: at a fixed cap the 1-layer configs peaked at
        # epochs 19-20 still improving while 2-layer/dropout-0.3 peaked at 14,
        # so a fixed budget compares a finished run against an unfinished one.
        # Capacity is a FACTOR here, so that bias would land directly on the
        # effect being estimated.
        cfg = TrainConfig(seed=r["run"], dev_frac=0.0, holdout_frac=0.0,
                          epochs=30, patience=4, min_delta=0.002,
                          n_layer=r["n_layer"], n_embd=r["n_embd"],
                          n_head=r["n_head"], lr=r["lr"],
                          weight_decay=r["weight_decay"],
                          attn_dropout=r["dropout"], resid_dropout=r["dropout"],
                          fusion=r["fusion"])
        res = train(arm, root, cfg, ExampleConfig(), SeqConfig(), verbose=False)
        row = dict(r)
        row.update({"macro_auroc": res["macro_auroc"],
                    "macro_ap": res["macro_ap"],
                    "hold_macro_auroc": res.get("holdout_macro_auroc",
                                               float("nan")),
                    "hold_macro_ap": res.get("holdout_macro_ap",
                                            float("nan")),
                    "epoch": res["epoch"],
                    "minutes": (time.time() - t) / 60})
        if res.get("preds") is not None:
            Path("artifacts").mkdir(exist_ok=True)
            np.save(f"artifacts/design_preds_run{r['run']:02d}.npy",
                    res["preds"])
        rows.append(row)
        if verbose:
            print("  run %2d %-7s AUROC %.4f  AP %.4f  [%.1f min]"
                  % (r["run"], r["kind"], row["macro_auroc"], row["macro_ap"],
                     row["minutes"]), flush=True)
        # Written after every run, so a crash at run 17 does not lose 16 runs.
        pd.DataFrame(rows).to_csv("outputs/design_runs.csv", index=False)
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict({"event": "design_run"}, **row)) + '\n')
    return pd.DataFrame(rows)


def propose_config(df, metric="macro_ap", shrink_it=True):
    """Predict the best CELL from the fitted effects, not the best RUN observed.

    W11 asked for this and it was missing: the design reported effects and
    stopped. Reporting effects answers "what matters"; it does not answer "what
    should we run", and those are different questions.

    Why not simply take the best observed run: with 16 runs and SE(effect)
    ~0.0028, the maximum of 16 noisy draws is inflated by
    sigma*sqrt(2 ln 16) ~ 2.4 sigma. The argmax run is the winner's curse in its
    purest form. A model fitted to ALL 16 runs uses every run to estimate each
    effect -- the hidden replication -- so its prediction for a cell is far more
    stable than that cell's single observation.

    Fits main effects plus all ten two-factor interactions (resolution V makes
    them unaliased), SHRINKS the coefficients toward zero, then evaluates all 32
    cells of the full factorial -- including the 16 that were never run, which is
    the point: a good combination can be predicted without being tried.

    Shrinkage is not optional here. Acting on unshrunken coefficients is exactly
    "predict the best cell from fitted effects without a significance gate",
    which the plan names as the thing that manufactures optimism.
    """
    import itertools
    from .effects import shrink

    corners = df[df.kind == "corner"]
    coded = ["c_" + f for f in FACTORS]
    X1 = corners[coded].to_numpy(float)
    y = corners[metric].to_numpy(float)

    pairs = list(itertools.combinations(range(len(coded)), 2))
    X = np.column_stack([np.ones(len(X1)), X1] +
                        [X1[:, a] * X1[:, b] for a, b in pairs])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    if shrink_it:
        # SE of a COEFFICIENT, from pure error -- not std(y)/sqrt(n), which is
        # the SE of the mean of y and contains the signal we are trying to
        # estimate. Using it made tau^2 collapse to 0, shrank every coefficient
        # to zero, and returned 32 identical predictions.
        #
        # Residuals cannot supply sigma here: 16 runs fitting 1 + 5 + 10 = 16
        # parameters leaves ZERO residual degrees of freedom. The replicated
        # centre points exist precisely for this, and with orthogonal +-1 coding
        # SE(coef) = sigma_pure / sqrt(n).
        # DELIBERATELY the centre-only sigma, not the pooled 7-df estimate
        # that `analyse_design` now uses. Shrinkage strength depends on it, so
        # switching would move every predicted cell -- and the top cell's
        # prediction has already been pre-registered in the ledger as the thing the confirmation runs test.
        # Changing the estimator after recording the prediction and before
        # reading the result is a forking path, and a small one is still one.
        # Revisit only after the confirmation has been scored, as a stated
        # deviation.
        centre = df[df.kind == "centre"]
        if len(centre) >= 2:
            sigma_pure = float(centre[metric].std(ddof=1))
        else:                      # no replicates: fall back to Lenth's PSE
            e = np.abs(beta[1:])
            s0 = 1.5 * np.median(e)
            small = e[e < 2.5 * s0] if s0 > 0 else e
            sigma_pure = (1.5 * np.median(small) * np.sqrt(len(y))) if len(small) else 0.0
        se = np.full(len(beta) - 1, sigma_pure / np.sqrt(len(y)))
        # Intercept is not an effect and must not be shrunk toward zero.
        beta = np.concatenate([beta[:1], shrink(beta[1:], se)])

    grid = np.array(list(itertools.product([-1, 1], repeat=len(coded))), float)
    G = np.column_stack([np.ones(len(grid)), grid] +
                        [grid[:, a] * grid[:, b] for a, b in pairs])
    pred = G @ beta

    out = pd.DataFrame(grid, columns=[f.replace("c_", "") for f in coded])
    out["predicted"] = pred
    ran = {tuple(r) for r in X1}
    out["was_run"] = [tuple(r) in ran for r in grid]
    return out.sort_values("predicted", ascending=False).reset_index(drop=True)


def report_proposal(prop, df, metric):
    print(f"\n=== BEST-CONFIG PROPOSAL from the fitted, shrunken model ({metric}) ===")
    cols = [c for c in prop.columns if c not in ("predicted", "was_run")]
    print(f"{'rank':>5}  " + "  ".join(f"{c:>12}" for c in cols)
          + f"{'predicted':>12}{'ran?':>7}")
    for i, r in prop.head(5).iterrows():
        vals = "  ".join(f"{('high' if r[c] > 0 else 'low'):>12}" for c in cols)
        print(f"{i+1:>5}  {vals}{r['predicted']:>12.4f}{str(bool(r['was_run'])):>7}")
    best_obs = df[df.kind == "corner"][metric].max()
    top = prop.predicted.iloc[0]
    print(f"\n  best OBSERVED run:  {best_obs:.4f}")
    print(f"  best PREDICTED cell: {top:.4f}")
    print(f"  {int((~prop.was_run).sum())} of {len(prop)} cells were never run; "
          f"{'the top cell is one of them' if not prop.was_run.iloc[0] else 'the top cell was run'}")
    print("  Predicted, not observed: the maximum of 16 noisy runs is inflated by")
    print("  ~2.4 sigma, while a fit uses all 16 runs to estimate each effect.")


def confirm_runs(root=".", arm="P4", n_cells=2, n_seeds=3, verbose=True):
    """Run the cells the model PREDICTS are best, and check the prediction.

    This is not a victory lap, it is the design's only falsification test.

    A 2^(5-1) fraction spends 16 runs estimating 16 parameters (intercept, 5
    main effects, 10 two-factor interactions), so the fitted model reproduces
    all 16 observed corners **exactly** and has **zero residual degrees of
    freedom**. Nothing inside the experiment can contradict it. Its predictions
    for the 16 cells of the complementary fraction -- which is where BOTH
    metrics put their top-ranked cell -- rest entirely on the assumption that
    every three-, four- and five-factor interaction is zero. Under the generator
    I = ABCDE each two-factor interaction is aliased with a three-factor one
    (AB with CDE, and so on), so that assumption is doing real work here and is
    untestable from inside the design.

    The only way to test it is to run a cell in the other half and see whether
    the prediction lands. The prediction is therefore written to the ledger
    BEFORE the runs start: with it on disk first, a miss cannot be
    reinterpreted afterwards as something we expected all along.

    Replicated, because one run cannot separate "the model extrapolates badly"
    from "this draw was unlucky". sigma is 0.0063 (E18) and the AUROC
    prediction sits +0.018 above the best observed run -- under 3 sigma of a
    single run. Three seeds put the SE of the cell mean near 0.0036 and make
    the comparison interpretable in either direction.

    A miss is worth as much as a hit: it would mean `propose_config`
    extrapolates on an assumption these data cannot support, and that the
    design's trustworthy output is its effects, not its config recommendation.
    """
    from .data.examples import ExampleConfig
    from .data.sequences import SeqConfig
    from .train_finetune import TrainConfig, train

    basin = _read_ledger("lr_basin")
    if basin is None:
        raise SystemExit("no lr_basin in the ledger")
    basin_lr = float(basin["basin_lr"])
    lo_hi = dict(FACTORS)
    lo_hi["lr"] = (basin_lr / 3, basin_lr * 3)

    df = pd.read_csv("outputs/design_runs.csv")
    props = {m: propose_config(df, m) for m in ("macro_ap", "macro_auroc")}

    # Cells are chosen by macro AP -- the PRE-REGISTERED primary (W2). AUROC
    # rides along as the co-secondary and does not get a vote in the choice.
    cells = props["macro_ap"].head(n_cells)
    coded = list(FACTORS)

    preg = []
    for rank, (_, c) in enumerate(cells.iterrows(), start=1):
        key = tuple(int(c[f]) for f in coded)
        pa = props["macro_auroc"]
        m = np.ones(len(pa), bool)
        for f, v in zip(coded, key):
            m &= (pa[f].to_numpy() == v)
        preg.append({"rank": rank, "cell": dict(zip(coded, key)),
                     "was_run": bool(c["was_run"]),
                     "pred_macro_ap": float(c["predicted"]),
                     "pred_macro_auroc": float(pa.loc[m, "predicted"].iloc[0])})

    LEDGER.parent.mkdir(exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "confirm_prereg", "n_seeds": n_seeds,
                             "cells": preg}) + chr(10))
    print("PRE-REGISTERED, on disk before any run starts:", flush=True)
    for q in preg:
        print("  rank %d  %s" % (q["rank"], q["cell"]), flush=True)
        print("          predicts AP %.4f / AUROC %.4f   (was_run=%s)"
              % (q["pred_macro_ap"], q["pred_macro_auroc"], q["was_run"]),
              flush=True)

    # RESUME. The first attempt died of memory exhaustion at run 3 of 6 (D20,
    # violated by running a second job beside it), and `rows` starting empty
    # would have rewritten design_confirm.csv with only the new runs --
    # destroying the two survivors that the per-run write had just saved. The
    # per-run write protects against losing a crash's completed work only if
    # the restart reads it back.
    #
    # Keyed on (rank, seed), which is what identifies a cell-replicate. Seeds
    # are deterministic in rank and index, so a resumed run reproduces exactly
    # the set the original would have produced -- no reshuffling that would
    # quietly change which replicates the mean is over.
    rows = []
    done = set()
    prev = Path("outputs/design_confirm.csv")
    if prev.exists():
        old_df = pd.read_csv(prev)
        # Only rows whose pre-registered prediction still matches are kept. If
        # the proposal moved, the old runs answer a question that is no longer
        # being asked and silently averaging them in would be wrong.
        pred_by_rank = {q["rank"]: q["pred_macro_ap"] for q in preg}
        for _, r in old_df.iterrows():
            want = pred_by_rank.get(int(r["rank"]))
            if want is not None and abs(float(r["pred_macro_ap"]) - want) < 1e-9:
                rows.append(r.to_dict())
                done.add((int(r["rank"]), int(r["seed"])))
        stale = len(old_df) - len(rows)
        print("  resuming: %d completed runs reused%s"
              % (len(rows), ", %d dropped as stale" % stale if stale else ""),
              flush=True)

    for q in preg:
        pick = {f: lo_hi[f][0] if q["cell"][f] < 0 else lo_hi[f][1]
                for f in coded}
        n_layer, n_embd, n_head = pick.pop("capacity")
        for si in range(n_seeds):
            # Seeds start at 100. The design used 0-19 and the resume
            # fingerprint includes the seed, so a collision would silently
            # continue a design run's checkpoint -- that is D26 exactly.
            seed = 100 + q["rank"] * 10 + si
            if (q["rank"], seed) in done:
                continue
            t = time.time()
            cfg = TrainConfig(seed=seed, dev_frac=0.0, holdout_frac=0.0,
                              epochs=30, patience=4, min_delta=0.002,
                              n_layer=n_layer, n_embd=n_embd, n_head=n_head,
                              lr=pick["lr"], weight_decay=pick["weight_decay"],
                              attn_dropout=pick["dropout"],
                              resid_dropout=pick["dropout"],
                              fusion=pick["fusion"])
            res = train(arm, root, cfg, ExampleConfig(), SeqConfig(),
                        verbose=False)
            row = {"rank": q["rank"], "seed": seed, "kind": "confirm",
                   "n_layer": n_layer, "n_embd": n_embd, "n_head": n_head,
                   **pick,
                   **{"c_" + f: q["cell"][f] for f in coded},
                   "pred_macro_ap": q["pred_macro_ap"],
                   "pred_macro_auroc": q["pred_macro_auroc"],
                   "macro_auroc": res["macro_auroc"],
                   "macro_ap": res["macro_ap"], "epoch": res["epoch"],
                   "minutes": (time.time() - t) / 60}
            if res.get("preds") is not None:
                Path("artifacts").mkdir(exist_ok=True)
                np.save("artifacts/confirm_preds_r%ds%d.npy"
                        % (q["rank"], seed), res["preds"])
            rows.append(row)
            if verbose:
                print("  rank %d seed %d  AUROC %.4f  AP %.4f  [%.1f min]"
                      % (q["rank"], seed, row["macro_auroc"], row["macro_ap"],
                         row["minutes"]), flush=True)
            pd.DataFrame(rows).to_csv("outputs/design_confirm.csv", index=False)
            with LEDGER.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict({"event": "confirm_run"}, **row)) + chr(10))
    return pd.DataFrame(rows)


def _pred_sigma(df, metric):
    """SE of a saturated design's prediction at ANY cell -- which is sigma.

    Worth deriving rather than assuming, because the answer is surprising and
    it decides whether a miss is real.

    With +-1 coding the 16 corner columns are orthogonal, so X'X = 16*I and
    Var(beta_j) = sigma^2 / 16 for each of the 16 coefficients (intercept, 5
    main effects, 10 two-factor interactions). A prediction is
    pred = sum_j g_j * beta_j with every g_j = +-1, and the coefficients are
    uncorrelated, so

        Var(pred) = sum_j g_j^2 * sigma^2/16 = 16 * sigma^2/16 = sigma^2.

    The fitted model's prediction is therefore **exactly as noisy as one run**.
    That is the price of saturation: 16 runs spent on 16 parameters buys
    hidden replication for each individual EFFECT (SE = 2*sigma/sqrt(16) =
    sigma/2) but none at all for a whole-cell prediction, because every
    coefficient's error is summed back in.

    Consequence for the comparison: the SE of (observed mean over R seeds minus
    prediction) is sqrt(sigma^2/R + sigma^2), never s/sqrt(R). At R = 3 that is
    1.15*sigma against 0.58*sigma -- a factor of two, and exactly the factor
    that would turn a 2-sigma "the proposal failed" into a 1-sigma "consistent".

    Shrinkage reduces the variance of `pred` and adds bias in exchange, so
    treating it as unshrunken overstates this SE slightly. That is the
    conservative direction and it is taken deliberately: the alternative
    understates the uncertainty of a claim that the method does not work.
    """
    centre = df[df.kind == "centre"]
    s_centre = float(centre[metric].std(ddof=1)) if len(centre) >= 2 else None
    # Centre-only, for the reason given in analyse_design: the seed replicates
    # scored on dev n=558 and these centre points on val n=365, so pooling them
    # mixes two estimands. E24.6 / E27.
    return s_centre if s_centre is not None else float("nan")


def report_confirm(cf, df):
    """Did the predicted cells deliver? Judged against the pre-registered value."""
    print(chr(10) + "=== CONFIRMATION of the predicted-best cells ===")
    for m in ("macro_ap", "macro_auroc"):
        print(chr(10) + "  " + m)
        best_obs = float(df[df.kind == "corner"][m].max())
        sig = _pred_sigma(df, m)
        corner_mean = float(df[df.kind == "corner"][m].mean())
        print("    sigma = %.4f, so SE(prediction) = %.4f -- see _pred_sigma:"
              " a saturated" % (sig, sig))
        print("    design predicts a CELL no more precisely than a single run"
              " measures one.")
        for rank, g in cf.groupby("rank"):
            obs = g[m].to_numpy(float)
            pred = float(g["pred_" + m].iloc[0])
            mean = float(obs.mean())
            R = len(obs)
            s_obs = float(obs.std(ddof=1)) if R > 1 else float("nan")
            # Both terms, not just the seeds' scatter: the thing being compared
            # against is itself an estimate with SE = sigma.
            se = float(np.sqrt(sig ** 2 / R + sig ** 2))
            d = mean - pred
            z = d / se if se > 0 else float("nan")
            print("    rank %d  n=%d  observed %.4f (seed sd %.4f)   predicted"
                  " %.4f" % (rank, R, mean, 0.0 if s_obs != s_obs else s_obs,
                             pred))
            print("            error %+.4f +- %.4f  (%+.1f SE)  %s"
                  % (d, se, z, "MISS" if abs(z) >= 2 else "consistent"))
            print("            vs best corner OBSERVED %+.4f   vs corner mean"
                  " %+.4f" % (mean - best_obs, mean - corner_mean))
        print("    A negative error beyond 2 SE means `propose_config`"
              " extrapolated on effect")
        print("    heredity and heredity did not hold -- a finding about the"
              " METHOD, reported")
        print("    either way. 'vs corner mean' is the weaker but more robust"
              " question:")
        print("    whether the proposed cell beats the average of what was"
              " actually run,")
        print("    which does not depend on the point prediction being right"
              " at all.")


def analyse_design(df, metric="macro_ap"):
    """Main effects from the corners; pure error and curvature from the centre."""
    from .effects import orthogonal_effects, report_orthogonal

    corners = df[df.kind == "corner"].copy()
    centre = df[df.kind == "centre"]

    # Pure error first: it feeds the effect intervals, so it cannot be computed
    # after them. `paired_effects` is deliberately NOT used -- in a 2^(5-1)
    # fraction the fifth factor is determined by the other four, so no run has a
    # twin matching on everything else and pairing returns nothing at all. Found
    # by testing the analysis on synthetic runs before spending 5 CPU-hours.
    s_pure = float(centre[metric].std(ddof=1)) if len(centre) >= 2 else None
    n_pure = len(centre)

    # POOL the centre-point estimate with the seed replicates measured before
    # the design ran. Both estimate the same thing -- total run-to-run sigma at
    # a fixed configuration -- and each on its own is badly under-powered:
    # 4 centre points give 3 df and 5 seed runs give 4, and a variance estimate
    # on 3 df has a 95% interval spanning roughly 0.6x to 2.9x of the truth.
    #
    # That instability is visible in the numbers. On AP the centre estimate is
    # the LARGER of the two (0.0112 vs 0.0080); on AUROC it is the SMALLER
    # (0.0045 vs 0.0063). Two estimates of one quantity disagreeing in opposite
    # directions on two metrics is the signature of noise, not of a real
    # difference between the two configurations.
    #
    # So the pooling is tested rather than assumed: an F-test for equal
    # variances, and pooling only if it does not reject. If it does reject, the
    # two are measuring different things and the LARGER is used, which is the
    # same "use the larger" convention `effects.report` already applies to
    # se_fold vs se_config. Never silently take the smaller -- that is the one
    # choice that manufactures significance.
    # NOT POOLED with the seed-replicate sigma, and that is a reversal.
    #
    # An earlier version pooled the centre points with `seed_sigma` from the
    # ledger, gated on an F-test. It was wrong: `design_seeds.csv` carries
    # populated hold_auroc columns, which only exist when holdout_frac > 0, so
    # the seed runs scored on **dev n=558** while these centre points score on
    # **val n=365**. Two different evaluation sets estimate two different
    # quantities and pooling them was never licensed. For AP it moved sigma
    # 0.0112 -> 0.0095, narrowing fusion's interval in the ANTI-conservative
    # direction.
    #
    # The F-test that licensed it could not have caught this either: at 3 and 4
    # degrees of freedom it only rejects beyond a 3.2x ratio of sigmas, so
    # "p = 0.53, no evidence they differ" was a statement about its own power.
    #
    # Centre-only is the honest estimator here: it is measured on the same
    # evaluation set, at a configuration inside the design, on 3 df. Wide, and
    # correct. See E24.6 and E27.
    if s_pure is not None:
        print("\n  pure error from %d centre replicates (val n=365): sigma = %.4f"
              % (n_pure, s_pure))
        print("    NOT pooled with seed_sigma -- that was measured on dev n=558,"
              " a different")
        print("    evaluation set. See E24.6 / E27.")

    coded = ["c_" + f for f in FACTORS]
    eff = orthogonal_effects(corners, coded, metric,
                             sigma_pure=s_pure, n_pure=n_pure)
    if not eff.empty:
        eff["factor"] = eff.factor.str.replace("c_", "", regex=False)
    report_orthogonal(eff, metric + " (design corners)")
    eff.to_csv("outputs/design_effects_" + metric + ".csv", index=False)

    if len(centre) >= 2:
        # Pure error, estimated INSIDE the experiment rather than assumed
        # (computed above, since the effect intervals depend on it).
        print("  sigma used for the effect intervals: %.4f on %d df"
              % (s_pure, n_pure - 1))
        # Curvature uses the CENTRE-ONLY sigma on purpose. It is a statement
        # about where the centre points sit relative to the corners, so the
        # relevant scatter is the scatter of those centre points; borrowing a
        # variance measured at a different configuration would be assuming part
        # of what the test is meant to check.
        s_centre = float(centre[metric].std(ddof=1))
        d = float(corners[metric].mean() - centre[metric].mean())
        se = s_centre * np.sqrt(1.0 / len(corners) + 1.0 / len(centre))
        z = d / se if se > 0 else float("nan")
        print("  curvature (corners - centre): %+.4f (%+.1f sigma of pure error)"
              % (d, z))
        if abs(z) >= 2:
            print("    => the response is NOT linear over these ranges. A"
                  " two-level contrast straddling an interior optimum reads near"
                  " zero, so no factor here may be called inert.")
        else:
            print("    => no detectable curvature; the two-level reads are"
                  " interpretable at face value.")
        print("  NOTE: centre runs sit at fusion='none', so curvature is tested"
              " only within the non-fusion half.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=0,
                    help="measure sigma with N seed replicates, then stop")
    ap.add_argument("--lr-basin", action="store_true",
                    help="1-D learning-rate sweep; must precede the design")
    ap.add_argument("--show-design", action="store_true")
    ap.add_argument("--run", action="store_true",
                    help="run the full design (needs --lr-basin first)")
    ap.add_argument("--analyse", action="store_true",
                    help="re-analyse outputs/design_runs.csv, no re-run")
    ap.add_argument("--confirm", action="store_true",
                    help="run the PREDICTED-best cells and test the prediction")
    ap.add_argument("--cells", type=int, default=2)
    ap.add_argument("--seeds-per-cell", type=int, default=3)
    a = ap.parse_args()

    if a.show_design:
        d = resolution_v_design()
        print("2^(5-1) resolution V, generator E=ABCD:\n")
        print(pd.DataFrame(d, columns=list(FACTORS)).to_string())
        print("\n  verified: all 5 main effects and all 10 two-factor "
              "interactions mutually orthogonal")
        return

    if a.confirm:
        cf = confirm_runs(".", n_cells=a.cells, n_seeds=a.seeds_per_cell)
        report_confirm(cf, pd.read_csv("outputs/design_runs.csv"))
        return

    if a.analyse:
        df = pd.read_csv("outputs/design_runs.csv")
        for m in ("macro_ap", "macro_auroc"):
            analyse_design(df, m)
            report_proposal(propose_config(df, m), df, m)
        return

    if a.run:
        df = run_design(".")
        for m in ("macro_ap", "macro_auroc"):
            analyse_design(df, m)
            report_proposal(propose_config(df, m), df, m)
        return

    if a.lr_basin:
        print("locating the learning-rate basin (2L/d128, fusion=none, scored on provided val) ...", flush=True)
        out = lr_basin(".")
        LEDGER.parent.mkdir(exist_ok=True)
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": "lr_basin", **out}) + "\n")
        return

    if a.seeds:
        print(f"measuring sigma over {a.seeds} seeds (2L/d128, fusion=none, same protocol as the design) ...",
              flush=True)
        out = seed_sigma(".", a.seeds)
        report_sigma(out)
        LEDGER.parent.mkdir(exist_ok=True)
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": "seed_sigma", **out}) + "\n")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
