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
    print(f"\n  For scale, the whole measured spread of this project is "
          f"0.0061 to 0.0148.")
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

    coded = ["c_" + f for f in FACTORS]
    eff = orthogonal_effects(corners, coded, metric,
                             sigma_pure=s_pure, n_pure=len(centre))
    if not eff.empty:
        eff["factor"] = eff.factor.str.replace("c_", "", regex=False)
    report_orthogonal(eff, metric + " (design corners)")
    eff.to_csv("outputs/design_effects_" + metric + ".csv", index=False)

    if len(centre) >= 2:
        # Pure error, estimated INSIDE the experiment rather than assumed
        # (computed above, since the effect intervals depend on it).
        print("\n  pure error from %d centre replicates: sigma = %.4f"
              % (len(centre), s_pure))
        d = float(corners[metric].mean() - centre[metric].mean())
        se = s_pure * np.sqrt(1.0 / len(corners) + 1.0 / len(centre))
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
    a = ap.parse_args()

    if a.show_design:
        d = resolution_v_design()
        print("2^(5-1) resolution V, generator E=ABCD:\n")
        print(pd.DataFrame(d, columns=list(FACTORS)).to_string())
        print("\n  verified: all 5 main effects and all 10 two-factor "
              "interactions mutually orthogonal")
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
        print("locating the learning-rate basin (2L/d128, dev_frac=0.2) ...", flush=True)
        out = lr_basin(".")
        LEDGER.parent.mkdir(exist_ok=True)
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": "lr_basin", **out}) + "\n")
        return

    if a.seeds:
        print(f"measuring sigma over {a.seeds} seeds (2L/d128, dev_frac=0.2) ...",
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
