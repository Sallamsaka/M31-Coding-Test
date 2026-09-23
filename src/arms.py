"""Phase B: structural arms against config (1), run sequentially overnight.

**Why arms and not another factorial.** The 20-run design measured the
fusion-ON subspace as close to flat: tau (true config spread) is 0.0090 on AP
against sigma 0.0095, and 0.0024 on AUROC against 0.0056. Selecting the argmax
of M candidates gains G = tau^2/sqrt(tau^2+sigma^2)*sqrt(2 ln M) in true quality
and inflates the reported number by B = sigma^2/sqrt(tau^2+sigma^2)*sqrt(2 ln M),
so **G/B = tau^2/sigma^2 = 0.90**: another hyperparameter design would make the
write-up more wrong than it makes the model better. Since G scales with tau^2,
the payoff comes from candidates that are genuinely DIFFERENT, not finely tuned.
Hence structural single-variable arms.

**What they are aimed at.** Two measurements point the same way. At config (1)
the trunk is ~123k parameters and `Linear(3296, 64)` is **211k**, so most of the
model is a linear map over the columns GBDT already eats -- and turning fusion on
compresses the spread over every other hyperparameter **2.9x**, which is what a
dominant tabular path looks like. Separately §E25.2 measured the model as
**overfitting**, not step-starved: training to 30 epochs costs 0.022 AP and
patience already picks the true argmax. So the arms are capacity cuts and
regularisers, plus the two fusion variants that have never been compared cleanly.

**Seed-major run order, deliberately.** Every arm at seed 300, then every arm at
seed 301, and so on -- not arm-by-arm. An overnight chain that dies at 60% then
leaves a BALANCED set of arms rather than complete data on the first few and none
on the rest, and a balanced prefix still supports every contrast at a wider
interval. This is the same reasoning that randomised the design's run order,
where the failure mode was a crash leaving no lr-high runs at all.

**Protocol held fixed at the design's**: 30-epoch cap, patience 4,
min_delta 0.002, scored on the provided validation set. §E25.2 checked that this
rule stops at the true argmax rather than truncating, and keeping it identical is
what makes these numbers comparable to the 20 design runs and the confirm runs.

Run: ``python -u -m src.arms``            (resumes; safe to re-launch)
     ``python -u -m src.arms --report``   (no training)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

LEDGER = Path("outputs/arms_ledger.jsonl")
CSV = Path("outputs/arms_runs.csv")

# Config (1) -- the design's predicted-best cell, confirmed at AP 0.2380/0.2380.
BASE = dict(n_layer=1, n_embd=64, n_head=2, lr=0.0036, weight_decay=0.001,
            attn_dropout=0.0, resid_dropout=0.0, fusion="readout",
            fusion_dim=64, modality_dropout=0.0)

# Ordered by (expected tau) x (confidence the arm cannot be silently wrong).
# Every one is a single-variable change from BASE.
ARMS = [
    ("baseline", {}),
    # Fusion OFF at THIS cell. The design measured fusion averaged over all 16
    # corners (+0.0534); it has never been measured at config (1) itself, and
    # every interaction says the cell matters. It also supplies the one number
    # missing from the sigma story: repeated seeds at fusion OFF with everything
    # else held identical.
    #
    # Why that matters. Run-to-run sigma_AP is 0.0112 at the design's centre
    # points (fusion OFF) and ~0.0006 across the confirm runs (fusion ON) -- an
    # 18x ratio. If real, it has the same explanation as the 2.9x compression of
    # the hyperparameter spread and the 211k-of-334k parameter count: a dominant
    # tabular path over a feature matrix that is IDENTICAL across seeds leaves
    # little for a seed to change. But the centre points differ from config (1)
    # in capacity, lr, weight_decay and dropout as well as fusion, so that
    # comparison is confounded four ways. This arm is the unconfounded version.
    ("nofusion", {"fusion": "none"}),
    # Capacity: the projection is 211k of ~334k parameters and the model
    # overfits, so this is the most direct cut available. Never swept -- 64 was
    # chosen by argument, and small-data work puts optimal projection dims at
    # 14-21, well below it. Bracketed above as well as below so a monotone
    # reading cannot be mistaken for an interior optimum.
    ("fdim16", {"fusion_dim": 16}),
    ("fdim32", {"fusion_dim": 32}),
    ("fdim128", {"fusion_dim": 128}),
    # Regularisers. §E16 named modality dropout as the fix if collapse were
    # found and never built it; §E25.2's overfitting result makes it apt twice.
    ("moddrop25", {"modality_dropout": 0.25}),
    ("moddrop50", {"modality_dropout": 0.50}),
    # The only fusion mechanism comparison that has never carried the §D18 leak.
    # `token` was additionally a silent no-op until §D5 (prepended at dt=0,
    # invisible under causal masking, measured delta exactly 0.0000), so it has
    # effectively never been tested at all.
    ("token", {"fusion": "token"}),
    # The basin value itself. The design tested basin/3 and basin*3 and never
    # the centre, and the basin was located at 2L/d128 with fusion OFF.
    ("lr_basin", {"lr": 0.0012}),
    # A1 measured lr-high (3.6e-3) beating lr-low (4e-4) by +0.0234 AT THIS CELL,
    # confirming the design's lr:fusion interaction (predicted +0.0190). But
    # 3.6e-3 is the EDGE of the range ever tested, and it is winning -- so the
    # optimum has moved well above the old basin of 1.2e-3, which was located at
    # 2L/d128 with fusion OFF. An optimum at a boundary is not an optimum; these
    # two arms are what make it one or move it.
    ("lr_up6e3", {"lr": 0.006}),
    ("lr_up1e2", {"lr": 0.010}),
    # T1: the winners COMBINED. Every arm above is a single-variable change from
    # BASE, so nothing has ever tested whether they add. The three picked are the
    # ones that survived to the test set: lr_basin (+0.0073 test), fdim128
    # (+0.0032), and moddrop25 -- which is 6th as a single model but 1st when its
    # three seeds are averaged, and averaging is what ships.
    #
    # Additivity is the open question and it is not free: lr:capacity was -0.0196
    # in the design and fusion_dim is a capacity knob, so these two have a
    # measured reason to interact. Two combos rather than one so a failure can be
    # attributed -- if the pair works and the triple does not, moddrop25 is what
    # broke it.
    ("combo_lr_fdim", {"lr": 0.0012, "fusion_dim": 128}),
    ("combo_all", {"lr": 0.0012, "fusion_dim": 128, "modality_dropout": 0.25}),
    # T2: augmentation option B on the WINNING cell, not on BASE -- the question
    # is whether it improves the model we would actually ship.
    #
    # Why run it despite the LR screen being a null on AP. The reason
    # augmentation is on the list at all is E33/B1: the transformer's learning
    # curve is still rising while LR's is flat. Screening a data-hunger
    # intervention on the model that is NOT data-limited, then generalising to
    # the one that is, is the wrong direction for the inference -- so LR's null
    # is weak evidence here specifically.
    #
    # Floor 10 is the largest dose (12,864 synthetic rows, 4.7x the fit set);
    # LR's dose-response was monotone in row count in BOTH metrics, so if there
    # is an effect this is where it is largest.
    #
    # Confound, stated rather than controlled: 4.7x rows means 4.7x optimiser
    # steps per epoch, and patience counts epochs. Normally that is fatal to the
    # comparison. Here it is benign in DIRECTION -- A1 measured this model as
    # overfitting rather than step-starved, so extra steps alone should hurt. A
    # win despite them is attributable to the data.
    ("combo_aug10", {"lr": 0.0012, "fusion_dim": 128, "_aug_floor": 10}),

    # ---- Phase 3: the two grid boundaries, under the POST-RE-CARVE scheme ----
    # New names rather than re-running the old ones, because every arm above
    # was trained before `_screen_fit_pids` existed and therefore saw fold 0 --
    # the set they would now be scored on (D36). Their rows stay on disk and
    # remain valid against the 358 carve; they are simply not a baseline here.
    # `v2_combo` is the fresh baseline these two are paired against.
    ("v2_combo", {"lr": 0.0012, "fusion_dim": 128}),

    # fusion_dim 256. The trend 16 -> 32 -> 64 -> 128 is monotone and 128 is
    # both the largest value ever run AND the winner, which E25.4 already named
    # as the trap it fixed for the learning rate: "an optimum sitting at a
    # boundary is not an optimum". fdim128 is half of E32's only resolvable
    # transformer gain, so the winning cell sits on an untested wall.
    ("v2_fdim256", {"lr": 0.0012, "fusion_dim": 256}),

    # Capacity, re-tested at the corrected lr. The design read capacity as a
    # null (-0.0028, not resolvable) -- but at lr levels 4e-4 and 3.6e-3, and
    # E28.3 later located the optimum at 1.2e-3, i.e. the design's HIGH level
    # was already past it. E24.5 measured `lr:capacity` at -0.0196 on AUROC,
    # the second-largest AUROC contrast of fifteen, so by the project's own
    # mechanism that null was measured at the wrong learning rate.
    # Levels are the design's own: (1,64,2) low -> (2,128,4) high, head dim 32
    # at both so the contrast is capacity and not a head-dim change.
    ("v2_capacity", {"lr": 0.0012, "fusion_dim": 128,
                     "n_layer": 2, "n_embd": 128, "n_head": 4}),
]
SEEDS = (300, 301, 302)
EPOCHS, PATIENCE, MIN_DELTA = 30, 4, 0.002


def _done() -> set:
    """(arm, seed) pairs already in the ledger, so a re-launch resumes.

    §D28.1: writing after every run only saves a crash's work if the restart
    reads it back. `confirm_runs` wrote per-run and still began from an empty
    list, which would have overwritten the two survivors it had just saved.
    """
    out = set()
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("event") == "arm_run":
                    out.add((r["arm_name"], int(r["seed"])))
    return out


def run(root: str = ".", arm: str = "P4") -> pd.DataFrame:
    from .data.examples import ExampleConfig
    from .data.sequences import SeqConfig
    from .train_finetune import TrainConfig, train

    done = _done()
    rows = []
    if CSV.exists():
        rows = pd.read_csv(CSV).to_dict("records")
    todo = [(s, n, o) for s in SEEDS for n, o in ARMS if (n, s) not in done]
    print("Phase B: %d arms x %d seeds = %d runs, %d already done, %d to run"
          % (len(ARMS), len(SEEDS), len(ARMS) * len(SEEDS), len(done), len(todo)),
          flush=True)
    print("  seed-major order, so a partial run leaves a balanced set", flush=True)

    t0 = time.time()
    for seed, name, over in todo:
        spec = dict(BASE, **over)
        # `_aug_floor` is an ExampleConfig knob, not a TrainConfig field, so it
        # is popped before the dataclass sees it.
        _aug = spec.pop("_aug_floor", None)
        _ex = (ExampleConfig() if _aug is None
               else ExampleConfig(augment=True, min_events=_aug))
        t = time.time()
        cfg = TrainConfig(seed=seed, dev_frac=0.0, holdout_frac=0.0,
                          epochs=EPOCHS, patience=PATIENCE,
                          min_delta=MIN_DELTA, predict_all=True, **spec)
        # Train on folds 1-4 so fold 0 stays a genuine holdout for scoring.
        # Epoch selection still happens on the provided val set, which is
        # common-mode across arms and cancels in arm-vs-arm contrasts.
        _fit_pids = _screen_fit_pids(root)
        res = train(arm, root, cfg, _ex, SeqConfig(), verbose=False,
                    fold_pids=_fit_pids)
        row = {"arm_name": name, "seed": seed, "aug_floor": _aug, **spec,
               "macro_auroc": res["macro_auroc"], "macro_ap": res["macro_ap"],
               "epoch": res["epoch"], "minutes": (time.time() - t) / 60}
        if res.get("preds") is not None:
            Path("artifacts").mkdir(exist_ok=True)
            np.save("artifacts/arm_preds_%s_s%d.npy" % (name, seed), res["preds"])
        if res.get("all_preds") is not None and _aug is not None:
            # An augmented run predicts over the AUGMENTED example frame, but
            # `_test_rows` masks the canonical 3,514-row frame. Saving the raw
            # array would mean a boolean mask of the wrong length at report
            # time -- after the training cost. Subset to real rows and reorder
            # onto the canonical pid order, with both facts asserted.
            from .data.examples import load_examples
            _exa = load_examples(root, _ex)
            _real = _exa.is_real.to_numpy().astype(bool)
            _pa = _exa.pid.to_numpy()[_real]
            _pr = load_examples(root, ExampleConfig()).pid.to_numpy()
            assert len(_pa) == len(_pr) and set(_pa.tolist()) == set(_pr.tolist()), (
                "augmented real rows must be exactly the canonical rows")
            _pos = {int(q): i for i, q in enumerate(_pa.tolist())}
            _idx = np.array([_pos[int(q)] for q in _pr.tolist()])
            res["all_preds"] = res["all_preds"][_real][_idx]
        if res.get("all_preds") is not None:
            # PROVENANCE. A prediction matrix is only interpretable next to the
            # set that produced it, and this file outlives the definition of
            # "held out" that was current when it was written -- which is
            # exactly what went wrong when the re-carve repointed `_test_rows`
            # at fold 0 while 14 arms on disk had trained on fold 0's patients.
            # Scoring them there was in-sample and silent.
            np.save("artifacts/arm_fitpids_%s_s%d.npy" % (name, seed),
                    np.array(sorted(_fit_pids), dtype=np.int64))
            # Full cohort, so this run can be scored on the held-out fold.
            np.save("artifacts/arm_allpreds_%s_s%d.npy" % (name, seed),
                    res["all_preds"])
        rows.append(row)
        print("  %-10s seed %d  AUROC %.4f  AP %.4f  ep %2d  [%.1f min, %.0f min"
              " elapsed]" % (name, seed, row["macro_auroc"], row["macro_ap"],
                             row["epoch"], row["minutes"], (time.time() - t0) / 60),
              flush=True)
        pd.DataFrame(rows).to_csv(CSV, index=False)
        LEDGER.parent.mkdir(exist_ok=True)
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict({"event": "arm_run"}, **row)) + "\n")
    return pd.DataFrame(rows)


def report() -> None:
    if not CSV.exists():
        print("no outputs/arms_runs.csv yet")
        return
    from scipy import stats

    d = pd.read_csv(CSV)
    base = d[d.arm_name == "baseline"].set_index("seed")
    print("\n=== Phase B arms, paired against config (1) ===")
    print("%-11s %3s %8s %8s %9s %10s %16s %7s" %
          ("arm", "n", "AP", "AUROC", "sd_AP", "dAP", "95% CI", "P(>)"))
    for name, _ in ARMS:
        g = d[d.arm_name == name]
        if g.empty:
            continue
        # sd_AP is the per-arm seed sigma, and it is a RESULT here, not a
        # diagnostic: §E25.5 predicts sigma_AP(nofusion)/sigma_AP(baseline) > 3
        # if the tabular path is what makes the fused model nearly deterministic.
        sd = g.macro_ap.std(ddof=1) if len(g) > 1 else float("nan")
        line = "%-11s %3d %8.4f %8.4f %9.5f" % (
            name, len(g), g.macro_ap.mean(), g.macro_auroc.mean(), sd)
        if name != "baseline" and not base.empty:
            # PAIRED on seed: same initialisation and batch order on both sides,
            # so whatever made a seed lucky cancels. Unpaired would carry the
            # whole between-seed spread -- the error `paired_effects` exists to
            # avoid.
            j = g.set_index("seed").join(base, rsuffix="_b", how="inner")
            dif = (j.macro_ap - j.macro_ap_b).dropna().to_numpy()
            if len(dif) >= 2:
                m, se = dif.mean(), dif.std(ddof=1) / np.sqrt(len(dif))
                tcrit = stats.t.ppf(0.975, len(dif) - 1)
                # P(arm > baseline) from the t posterior on the paired mean.
                # Small differences are reported as "not resolvable";
                # §W0 measured that a significance gate false-negatives ~90% of
                # the time while P(A>B) does so ~30% at the same false-positive
                # rate. Both are printed so neither has to stand alone.
                pgt = (1 - stats.t.cdf(0.0, len(dif) - 1, loc=m, scale=max(se, 1e-12))
                       ) if se > 0 else float(m > 0)
                line += " %+10.4f [%+.4f,%+.4f] %7.2f" % (
                    m, m - tcrit * se, m + tcrit * se, pgt)
            elif len(dif) == 1:
                line += " %+10.4f %16s %7s" % (dif[0], "(n=1)", "-")
        print(line)
    # The E25.5 test, stated explicitly rather than left for the reader to spot.
    b = d[d.arm_name == "baseline"].macro_ap
    nf = d[d.arm_name == "nofusion"].macro_ap
    if len(b) > 1 and len(nf) > 1:
        rb, rn = b.std(ddof=1), nf.std(ddof=1)
        F = (max(rb, rn) / max(min(rb, rn), 1e-12)) ** 2
        df1 = (len(nf) if rn >= rb else len(b)) - 1
        df2 = (len(b) if rn >= rb else len(nf)) - 1
        pF = 2 * min(stats.f.sf(F, df1, df2), stats.f.cdf(F, df1, df2))
        print("\n  §E25.5 TEST -- does fusion collapse sigma?")
        print("    sigma_AP  baseline (fusion ON) %.5f   nofusion (OFF) %.5f"
              % (rb, rn))
        print("    ratio %.1fx  F = %.1f on (%d, %d) df, p = %.3f"
              % (rn / max(rb, 1e-12), F, df1, df2, pF))
        print("    PRE-REGISTERED: ratio > 3 confirms; near 1 refutes and the")
        print("    sigma collapse is not caused by fusion.")
    print("\n  Paired on seed. The wall: an arm must beat baseline by more than")
    print("  1 SE to count, and the transformer's marginal ensemble contribution")
    print("  must clear the +0.0041 the UNTUNED P3 already supplies -- otherwise")
    print("  tuning has bought nothing a worse model did not already give.")


def _logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def _screen_fit_pids(root: str = ".") -> set:
    """Folds 1-4 of the standard partition -- what a screening arm may train on.

    The separate carve is retired (`cv.LOCKED_TEST_N = 0`), so the clean set an
    arm is scored against is fold 0 of the partition that exists anyway. It
    costs no patients, and unlike a fixed carve it rotates if we ever want it to.
    """
    from .cv import cv_pool_pids, fold_masks
    from .data.examples import ExampleConfig, load_examples

    ex = load_examples(root, ExampleConfig())
    pid = ex.pid.to_numpy()
    _, oof0 = fold_masks(root, n_splits=5)[0]
    held = set(np.unique(pid[np.asarray(oof0, bool)]).tolist())
    return {p for p in cv_pool_pids(root).tolist() if p not in held}


def _test_rows(root: str = "."):
    """Fold 0's held-out rows, with labels and the at-risk mask.

    Was the carved 358. The carve is retired -- it cost 12.8% of the training
    pool to duplicate what cross-validation already provides -- so the screening
    holdout is now fold 0, and arms train on folds 1-4 via `_screen_fit_pids`.

    Scores from before the re-carve are NOT comparable to scores after it: the
    set changed. Within-family arm contrasts survive, because both sides move
    together; absolute levels do not.
    """
    from .cv import fold_masks
    from .data.examples import ExampleConfig
    from .data.labels import at_risk_mask, load_labels

    lab = load_labels(root, ExampleConfig())
    _, oof0 = fold_masks(root, n_splits=5)[0]
    m = np.asarray(oof0, bool)
    assert m.sum() > 0, "fold 0 is empty"
    return lab["y"][m].astype(int), at_risk_mask(lab)[m], m


def report_test(root: str = ".") -> None:
    """Every arm, scored on our test set, seed-averaged.

    **Why this matters more than the validation table.** Every ranking in Phase B
    was produced on the provided validation set, which has been read 26+ times and
    carries a winner's-curse bound of about +0.027 -- larger than any difference
    between the arms. This set has been read zero times.

    Reporting all arms here is NOT a second selection: the winner's curse comes
    from *choosing* by a number, not from computing it. Every value below is an
    unbiased estimate of that arm. What must not happen is picking the winner by
    this column -- that is what the validation column and the CV folds are for.
    """
    import glob

    from .evaluate import macro_ap, macro_auroc

    y, ar, m = _test_rows(root)

    # The holdout's patients. Any run whose fit set touches these is being
    # scored in-sample, and an in-sample number is not a worse estimate -- it
    # is a different quantity, and it looks BETTER, so it cannot be caught by
    # noticing that it seems off. It has to be refused structurally.
    from .data.examples import ExampleConfig as _EC
    from .data.examples import load_examples as _le
    _held = set(np.unique(_le(root, _EC()).pid.to_numpy()[m]).tolist())

    val = pd.read_csv(CSV) if CSV.exists() else None
    rows, skipped = [], []
    for name, _ in ARMS:
        fs = sorted(glob.glob("artifacts/arm_allpreds_%s_s*.npy" % name))
        if not fs:
            continue
        _bad = None
        for f in fs:
            fp = f.replace("arm_allpreds_", "arm_fitpids_")
            if not Path(fp).exists():
                _bad = "no fit-set record (predates the provenance guard)"
                break
            ov = _held & set(np.load(fp).tolist())
            if ov:
                _bad = f"trained on {len(ov)} of {len(_held)} holdout patients"
                break
        if _bad:
            skipped.append((name, _bad))
            continue
        per = [np.load(f)[m] for f in fs]
        # Seed-averaged in logit space -- the same rule the ensemble ships, and
        # worth +0.0055 on validation for free.
        avg = 1.0 / (1.0 + np.exp(-sum(_logit(a) for a in per) / len(per)))
        singles = [macro_ap(y, a, mask=ar)[0] for a in per]
        r = {"arm": name, "n": len(fs),
             "test_ap_avg": macro_ap(y, avg, mask=ar)[0],
             "test_auroc_avg": macro_auroc(y, avg, mask=ar)[0],
             "test_ap_1seed": float(np.mean(singles)),
             "test_ap_sd": float(np.std(singles, ddof=1)) if len(singles) > 1
             else float("nan")}
        if val is not None:
            g = val[val.arm_name == name]
            if len(g):
                r["val_ap_1seed"] = float(g.macro_ap.mean())
        rows.append(r)
    if skipped:
        print("  REFUSED -- these arms cannot be scored on this holdout:")
        for nm, why in skipped:
            print(f"    {nm:16s} {why}")
        print("  Their pre-re-carve numbers against the 358 carve remain valid;"
              "\n  they are simply not comparable to anything scored here.")
    if not rows:
        print("no scorable artifacts/arm_allpreds_*.npy -- every arm on disk"
              " either predates the provenance guard or trained on this"
              " holdout. Re-run the arms to compare them here.")
        return
    d = pd.DataFrame(rows).sort_values("test_ap_avg", ascending=False)
    print("\n=== ARMS ON FOLD 0 (held out; the 358-carve is retired) ===")
    print(d.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    d.to_csv("outputs/arms_on_test.csv", index=False)

    if "val_ap_1seed" in d.columns and d.val_ap_1seed.notna().sum() >= 3:
        from scipy import stats
        sub = d.dropna(subset=["val_ap_1seed"])
        rho = stats.spearmanr(sub.val_ap_1seed, sub.test_ap_1seed)
        print("\n  DOES THE VALIDATION RANKING TRANSFER?")
        print("    Spearman(val rank, test rank) over %d arms = %+.3f (p = %.3f)"
              % (len(sub), rho.statistic, rho.pvalue))
        print("    ⚠ CONFOUNDED, and it must not be read as 'the ranking is")
        print("    real'. Both columns come from the SAME trained models, so")
        print("      val  = mu_arm + m_arm + e_val")
        print("      test = mu_arm + m_arm + e_test")
        print("    with m_arm (the luck of those particular seed draws) SHARED.")
        print("    The correlation is Var(mu+m)/(Var(mu+m)+Var(e)) and cannot")
        print("    separate a real arm effect from a lucky seed: eleven arms with")
        print("    identical true means would still correlate strongly.")
        print("    Decompose instead -- sd(m) = mean(test_ap_sd)/sqrt(n_seeds):")
        sd_m = d.test_ap_sd.mean() / np.sqrt(d.n.max())
        obs = d.test_ap_1seed.std(ddof=1)
        mu = max(0.0, obs ** 2 - sd_m ** 2) ** 0.5
        print("      real between-arm sd(mu) %.4f vs shared-seed sd %.4f"
              " (ratio %.1f)" % (mu, sd_m, mu / max(sd_m, 1e-9)))
        print("    Separating them properly needs DIFFERENT seeds for the two")
        print("    columns, which no run has yet done.")
    print("\n  wrote outputs/arms_on_test.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--test-only", action="store_true",
                    help="score existing runs on our test set, train nothing")
    a = ap.parse_args()
    if a.test_only:
        report_test()
        return
    if not a.report:
        run(".")
    report()
    report_test()


if __name__ == "__main__":
    main()
