"""The headline instrument: K=5 grouped CV over the pool, and the locked read.

Everything reported before this was measured on the 365-patient validation set,
which has been read ~26 times and carries a winner's-curse bound of +0.038 to
+0.051 -- larger than every difference this project has ever measured between
models (§G1). This module is what replaces it.

**Per-fold mean +- SD is PRIMARY; pooled OOF is secondary (§W1).** The two are
different estimands and can order models differently. AUROC is a *ranking*
metric, and pooling ranks patient A's score from model 1 against patient B's
score from model 2 -- scores that were never on a common scale. The measured
symptom is the prevalence-model artifact: pooled OOF AUROC for a constant
predictor is **not 0.5**, because a patient's own outcome is excluded from the
fit that scores them, so a fold with more positives hands all its members a
lower predicted rate and score anticorrelates with outcome.

So pooling is reported two ways: raw, and after **within-fold rank
normalisation** (ranked *before* the 1e-6 at-risk floor is applied, or the ties
corrupt the ranks). The gap between them is the measured heterogeneity budget,
not a nuisance to hide.

**One partition, one K, for anything compared (§X).** Unequal K applies the
fit-size tax asymmetrically and hits the more data-hungry family harder -- we
would measure data-hunger and report it as inferiority. Every model here runs on
the *same* `fold_masks` output.

**What the CV number is, stated so the report cannot overclaim.** It estimates
the expected performance of the *procedure* trained on ~1,946 patients, not of
any single shipped model -- each patient is scored by a different one. It is a
lower bound on the shipped model (refit on the whole pool) by the fit-size tax.
The bootstrap interval covers evaluation noise only.

**The locked test set is read ONCE**, for one pre-registered question: does the
shipped model's macro AP exceed the at-risk-mask-only baseline, with an interval?
It is 358 patients with 3 positives on the rarest label, so it **cannot** rank
LR against GBDT against the transformer and must not be asked to (§W3). It also
de-biases the *fit*, not the *design*: 26 validation reads and all feature
engineering preceded the carve.

Run: ``python -m src.cross_validate``              (LR + GBDT + prevalence)
     ``python -m src.cross_validate --locked``     (the single locked read)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .cv import cv_pool_pids, fold_masks
from .data.examples import ExampleConfig, load_examples
from .data.features import build_features
from .data.labels import at_risk_mask, load_labels
from .evaluate import macro_ap, macro_auroc, paired_bootstrap_delta
from .train_baseline import (GBDTConfig, LRConfig, apply_at_risk_mask, fit_gbdt,
                             fit_lr, fit_prevalence)


def _gbdt_cfg() -> "GBDTConfig":
    """The GBDT config for this run, with an optional experiment override.

    Set `CV_GBDT` to a JSON object, e.g. `{"min_samples_leaf": 25}`.

    **Why an env var instead of editing the dataclass default.** The default is
    mirrored in `configs/default.yaml` and `test_baseline_defaults_match`
    asserts the two agree. Editing the default to run an experiment makes the
    repository claim a value is *shipped* when it is only *under test* -- and
    that test caught exactly that mistake. An override keeps the experiment out
    of the declared configuration.

    The override flows into `_model_sig` as well, so an overridden run gets its
    own cache slot and cannot silently reuse the default's fits (D34).
    """
    import json
    import os
    from dataclasses import replace

    raw = os.environ.get("CV_GBDT", "").strip()
    if not raw:
        return GBDTConfig()
    return replace(GBDTConfig(), **json.loads(raw))


MODELS = ("prevalence", "trivial", "lr_default", "lr", "gbdt", "transformer")

# The transformer configuration to cross-validate: config (1) plus the two arm
# wins that survived to the test set (lr 1.2e-3, fusion_dim 128).
#
# ⚠ Why it is here at all. Until now `MODELS` was
# prevalence/trivial/lr_default/lr/gbdt, so LR and GBDT had 2,433-row
# out-of-fold estimates while EVERY transformer number -- the 20-run design, 7
# ablations, 33 Phase B arms -- came off the 365-patient validation set. The two
# families were being compared on instruments differing 6.7x in n, and the
# ensemble membership question was decided entirely on the noisier one.
#
# Cost: ~4.5 min a fold against GBDT's ~18, so this adds ~23 min to a ~95 min
# run. It is the cheapest thing in the loop except LR.
TRANSFORMER_CFG = dict(n_layer=1, n_embd=64, n_head=2, lr=0.0012,
                       weight_decay=0.001, attn_dropout=0.0, resid_dropout=0.0,
                       fusion="readout", fusion_dim=128, modality_dropout=0.0)
TRANSFORMER_SEEDS = (300, 301, 302)


def _fit_transformer(root, fit_rows, ex_cfg, y_shape, verbose=False):
    """Train on this fold's patients and return full-cohort probabilities.

    Seed-averaged over three runs in logit space. That is not a refinement: on
    our test set averaging three seeds is worth **+0.0133 AP** (§E29.3), more
    than any architectural arm in Phase B, and a single-seed CV number would
    understate the model we would actually ship.

    `fold_pids` restricts training to the fold; `predict_all` returns every
    cohort row so the out-of-fold slice can be taken here.
    """
    import numpy as _np

    from .data.examples import ExampleConfig as _EC
    from .data.sequences import SeqConfig
    from .train_finetune import TrainConfig, train

    ex = load_examples(root, ex_cfg or _EC())
    fold_pids = set(_np.unique(ex.pid.to_numpy()[fit_rows]).tolist())

    lg = lambda q: _np.log(_np.clip(q, 1e-6, 1 - 1e-6)
                           / (1 - _np.clip(q, 1e-6, 1 - 1e-6)))
    acc = None
    for sd in TRANSFORMER_SEEDS:
        cfg = TrainConfig(seed=sd, dev_frac=0.0, holdout_frac=0.0, epochs=30,
                          patience=4, min_delta=0.002, predict_all=True,
                          **TRANSFORMER_CFG)
        res = train("P4", root, cfg, ex_cfg or _EC(), SeqConfig(),
                    verbose=verbose, fold_pids=fold_pids)
        P = res.get("all_preds")
        if P is None:
            raise RuntimeError("predict_all returned nothing")
        acc = lg(P) if acc is None else acc + lg(P)
    return (1.0 / (1.0 + _np.exp(-acc / len(TRANSFORMER_SEEDS)))).astype("float32")



# A trivial clinical rule: age, sex, and how much prior record exists. REFORMS 5f
# and MI-CLAIM both require the comparator to be identified and justified, and
# "beats prevalence" is far too low a bar -- a rule a clinician could apply from
# the chart header is the honest floor.
TRIVIAL_COLS = ("age_z", "age_z_sq", "sex_F", "sex_M", "log_n_distinct_tokens")


def _subset(F, names):
    """A FeatureMatrix restricted to `names`, preserving everything else."""
    from dataclasses import replace
    keep = [i for i, nm in enumerate(F.names) if nm in set(names)]
    assert keep, f"none of {names} present in the feature matrix"
    return replace(F, X=F.X[:, keep], names=[F.names[i] for i in keep],
                   blocks={"trivial": slice(0, len(keep))})


def _rank_within(p: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Per-label rank-normalise a fold's scores to [0, 1].

    Applied BEFORE the at-risk floor, so the 1e-6 sentinels do not collapse into
    a tie block that destroys the ranking this is meant to preserve.
    """
    out = np.empty_like(p[rows], dtype=float)
    for j in range(p.shape[1]):
        v = p[rows, j]
        order = np.argsort(np.argsort(v))
        out[:, j] = order / max(len(v) - 1, 1)
    return out


import hashlib
from dataclasses import asdict


def _model_sig(m: str) -> str:
    """Everything BESIDES the fold that determines this model's output.

    Module level ON PURPOSE. D34 recorded as a known residual that this
    lived inside `run_cv` as a closure and therefore had no test asserting
    that a config edit changes the cache path -- the one property it exists
    to guarantee. It closes over nothing but module-level names, so there
    was never a reason for it to be a closure.

    D26 is the silent-wrong-resume failure, and the previous cache key was
    an instance of it waiting to happen: it captured fold membership and the
    model list but **not the model configs**. Editing TRANSFORMER_CFG or
    GBDTConfig would have reloaded stale predictions under an unchanged key
    and reported them as the new configuration's -- a run that succeeds,
    passes every test, and answers a different question than the one asked.
    """
    if m == "transformer":
        spec = (sorted(TRANSFORMER_CFG.items()), TRANSFORMER_SEEDS)
    elif m == "gbdt":
        spec = sorted(asdict(_gbdt_cfg()).items())
    elif m == "lr":
        spec = sorted(asdict(LRConfig()).items())
    elif m == "lr_default":
        spec = sorted(asdict(LRConfig(C=1.0)).items())
    elif m == "trivial":
        spec = (sorted(asdict(LRConfig()).items()), tuple(TRIVIAL_COLS))
    else:
        spec = ()          # prevalence has no hyperparameters
    return hashlib.sha256(repr(spec).encode()).hexdigest()[:8]


def run_cv(root: str = ".", n_splits: int = 5, verbose: bool = True,
           max_folds: int | None = None) -> dict:
    """K-fold CV over the pool.

    ``max_folds`` runs only the first N folds of the K-way partition. That is a
    COVERAGE reduction, not a data reduction: the unscored patients remain in
    the training set of every fold that does run, they simply never receive an
    out-of-fold prediction.

    Why it is worth having. K itself does not buy precision -- every patient is
    held out exactly once at any K -- so the only levers are how many fits you
    pay for and how much data each sees. Running 3 of 5 folds costs 29% more
    noise (eval n 2,433 -> 1,461, sigma_AP 0.0058 -> 0.0075) for 40% less time,
    and for the comparison this instrument exists to settle the difference is
    immaterial: the transformer-vs-LR gap is ~1 SE at three folds and ~1 SE at
    five.

    Prefer this over lowering ``n_splits``. A smaller K shrinks the FIT set
    (1,946 -> 1,622 at K=3), and §X row 1 records that the fit-size tax is only
    common-mode while the model families are equally data-hungry -- which §E33
    is evidence they are not. Cutting K would handicap the transformer in
    precisely the comparison being run.
    """
    ex_cfg = ExampleConfig()
    lab = load_labels(root, ex_cfg)
    y = lab["y"].astype(int)
    ar = at_risk_mask(lab)
    folds = fold_masks(root, n_splits=n_splits)
    if max_folds is not None:
        folds = folds[:max_folds]

    n = len(y)
    oof = {m: np.zeros((n, y.shape[1]), np.float32) for m in MODELS}
    oof_rank = {m: np.zeros((n, y.shape[1]), np.float32) for m in MODELS}
    scored = np.zeros(n, bool)
    per_fold: list[dict] = []

    # PER-FOLD RESUME. A fold is ~30 minutes and this run has now died twice on
    # GBDT with an ArrayMemoryError -- once at fold 3, once at fold 4 -- losing
    # every completed fold both times, because the OOF matrices only reach disk
    # at the very end. `arms.py`, `design.confirm_runs` and `gbdt_programme` all
    # resume; this, the longest job in the project, did not.
    #
    # Each fold's predictions are cached under a key that includes the model
    # list and the fold's own patient membership, so a cache cannot be reused
    # across a changed MODELS tuple or a re-partition (fold_masks(repeat=N)) --
    # which is the silent-wrong-resume failure of D26.
    cache = Path(root) / "artifacts" / "cv_fold_cache"
    cache.mkdir(parents=True, exist_ok=True)

    import hashlib
    from dataclasses import asdict

    def _fold_key(fit_rows) -> str:
        """Identifies the FIT SET only -- deliberately not the MODELS tuple.

        The earlier version hashed `",".join(MODELS)` in as well, which meant
        adding a seventh model threw away all six existing fits. With one file
        per model that coupling is simply wrong: a fold's `gbdt` predictions do
        not depend on whether a transformer is also being scored.
        """
        pid = load_examples(root, ex_cfg or ExampleConfig()).pid.to_numpy()
        blob = ",".join(map(str, sorted(set(pid[fit_rows].tolist()))))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _mpath(k: int, fkey: str, m: str) -> Path:
        return cache / f"fold{k}_{fkey}_{m}_{_model_sig(m)}.npz"

    if verbose and list(cache.glob("fold?_????????????????.npz")):
        print("  note: whole-fold caches from the previous scheme are present "
              "and will be IGNORED -- resume is now per model.", flush=True)

    for k, (fit_rows, oof_rows) in enumerate(folds):
        _fk = _fold_key(fit_rows)
        idx = np.flatnonzero(oof_rows)
        scored[idx] = True

        # PER-MODEL RESUME. The previous version wrote one file per fold, after
        # all six models had finished -- so an OOM inside GBDT (which has now
        # happened four times) discarded that fold's completed fits along with
        # it. One file per model means a crash costs only the model in flight.
        _missing = [m for m in MODELS if not _mpath(k, _fk, m).exists()]
        F = (build_features(root, ex_cfg=ex_cfg, fit_mask=fit_rows)
             if _missing else None)

        # Lazy: a thunk runs only when that model's cache is absent, so a fully
        # cached fold never touches build_features at all.
        _build = {
            "prevalence": lambda: fit_prevalence(F, y, fit_mask=fit_rows),
            # W9's two missing comparators.
            "trivial": lambda: fit_lr(_subset(F, TRIVIAL_COLS), y, ar,
                                      LRConfig(), verbose=False,
                                      fit_mask=fit_rows),
            # Untuned: sklearn's own default C=1.0 against our tuned 0.03. The
            # gap between these two IS the value of the tuning, which nothing
            # else in this project reports.
            "lr_default": lambda: fit_lr(F, y, ar, LRConfig(C=1.0),
                                         verbose=False, fit_mask=fit_rows),
            "lr": lambda: fit_lr(F, y, ar, LRConfig(), verbose=False,
                                 fit_mask=fit_rows),
            "gbdt": lambda: fit_gbdt(F, y, ar, _gbdt_cfg(), verbose=False,
                                     fit_mask=fit_rows),
            "transformer": lambda: _fit_transformer(root, fit_rows, ex_cfg,
                                                    y.shape),
        }
        preds, _cached = {}, []
        for m in MODELS:
            _mp = _mpath(k, _fk, m)
            if _mp.exists():
                preds[m] = np.load(_mp)["p"]
                _cached.append(m)
            else:
                preds[m] = _build[m]()
                # Written the moment THIS model finishes, not when the fold does.
                np.savez_compressed(_mp, p=preds[m])
        if verbose and _cached:
            print(f"  fold {k}: {len(_cached)}/{len(MODELS)} from cache "
                  f"({', '.join(_cached)})", flush=True)
        row = {"fold": k, "n_fit": int(fit_rows.sum()), "n_oof": len(idx)}
        for m, P in preds.items():
            # Rank BEFORE the floor (see _rank_within), then floor for scoring.
            oof_rank[m][idx] = _rank_within(P, idx)
            Pm = apply_at_risk_mask(P, ar)
            oof[m][idx] = Pm[idx]
            au, n_ok = macro_auroc(y[idx], Pm[idx], mask=ar[idx])
            ap, _ = macro_ap(y[idx], Pm[idx], mask=ar[idx])
            row[f"{m}_auroc"] = au
            row[f"{m}_ap"] = ap
            row[f"{m}_n_scored"] = n_ok
        per_fold.append(row)
        if verbose:
            print(f"  fold {k}: fit {row['n_fit']:,} oof {row['n_oof']:,}  "
                  + "  ".join(f"{m} AP {row[f'{m}_ap']:.4f}" for m in MODELS),
                  flush=True)

    df = pd.DataFrame(per_fold)
    df.to_csv("outputs/cv_per_fold.csv", index=False)

    # Save the OOF prediction matrices, not just the summary metrics.
    #
    # Without these, every downstream question needs a full re-run: which models
    # actually disagree (the Spearman that decides whether ensembling can help),
    # whether a GBDT+transformer blend beats either alone, a patient-level
    # bootstrap of a block effect. All of those are seconds of arithmetic on
    # stored predictions and ~40 minutes of CPU without them.
    #
    # Each row is one example and each column one label, so this is the object
    # every later comparison needs and none of them can reconstruct.
    Path("artifacts").mkdir(exist_ok=True)
    np.savez_compressed("artifacts/cv_oof_preds.npz",
                        scored=scored, **{m: oof[m] for m in MODELS},
                        **{f"{m}__rank": oof_rank[m] for m in MODELS})
    print(f"  wrote artifacts/cv_oof_preds.npz "
          f"({len(MODELS)} models x {int(scored.sum()):,} scored rows)", flush=True)

    # Each patient must be held out exactly once, or `bootstrap_macro`'s promise
    # to resample patients rather than rows is silently false.
    pid = lab["pid"] if "pid" in lab else None
    if pid is not None:
        p_oof = np.asarray(pid)[scored]
        assert len(np.unique(p_oof)) == len(p_oof), "a patient is scored twice"

    return {"per_fold": df, "oof": oof, "oof_rank": oof_rank, "scored": scored,
            "y": y, "ar": ar}


def report_cv(res: dict, verbose: bool = True) -> None:
    df, y, ar = res["per_fold"], res["y"], res["ar"]
    rows = np.flatnonzero(res["scored"])

    print("\n=== PRIMARY: per-fold mean +- SD across folds (§W1) ===")
    print(f"{'model':<12}{'macro AP':>20}{'macro AUROC':>22}")
    for m in MODELS:
        ap, au = df[f"{m}_ap"], df[f"{m}_auroc"]
        print(f"{m:<12}{ap.mean():>12.4f} +- {ap.std(ddof=1):.4f}"
              f"{au.mean():>14.4f} +- {au.std(ddof=1):.4f}")

    print("\n=== SECONDARY: pooled OOF, raw vs within-fold rank-normalised ===")
    print(f"{'model':<12}{'AP raw':>10}{'AP rank':>10}{'AUROC raw':>12}"
          f"{'AUROC rank':>12}")
    for m in MODELS:
        P, R = res["oof"][m], res["oof_rank"][m]
        Rm = apply_at_risk_mask(R, ar)
        a1 = macro_ap(y[rows], P[rows], mask=ar[rows])[0]
        a2 = macro_ap(y[rows], Rm[rows], mask=ar[rows])[0]
        u1 = macro_auroc(y[rows], P[rows], mask=ar[rows])[0]
        u2 = macro_auroc(y[rows], Rm[rows], mask=ar[rows])[0]
        print(f"{m:<12}{a1:>10.4f}{a2:>10.4f}{u1:>12.4f}{u2:>12.4f}")

    # The pooling artifact, measured rather than asserted away. A constant
    # predictor pooled across folds does NOT score 0.5, so run_baseline's
    # `assert abs(au - 0.5) < 1e-9` must be replaced here, not reused.
    pv = macro_auroc(y[rows], res["oof"]["prevalence"][rows],
                     mask=ar[rows])[0]
    print(f"\n  pooling artifact: prevalence-only pooled AUROC = {pv:.4f} "
          f"(0.5 only if pooling were benign)")
    print("  the raw-vs-rank gap above is the measured heterogeneity budget.")

    print("\n=== PAIRED COMPARISONS on identical folds ===")
    for a, b in (("lr", "gbdt"), ("lr", "lr_default"), ("lr", "trivial"),
                 ("gbdt", "trivial"), ("trivial", "prevalence")):
        d = paired_bootstrap_delta(y[rows], res["oof"][a][rows],
                                   res["oof"][b][rows])
        # BOTH metrics. paired_bootstrap_delta computes the AUROC delta too and
        # an earlier version printed only AP, discarding it -- which mattered,
        # because macro AP is the pre-registered primary but the ensemble result
        # in B was an AUROC finding.
        for metric in ("ap", "auroc"):
            pt, lo, hi, _sd = d[f"d_macro_{metric}"]
            pg = d.get(f"p_a_gt_b_{metric}")
            tag = "resolvable" if (lo > 0 or hi < 0) else "not resolvable"
            extra = f"  P({a}>{b}) = {pg:.2f}" if pg is not None else ""
            lbl = "dAP   " if metric == "ap" else "dAUROC"
            print(f"  {a:>10} - {b:<12} {lbl} {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                  f"  {tag}{extra}")
    print("\n  P(A>B) is reported beside every interval on purpose: a"
          " significance gate\n  alone false-negatives ~90% of the time at this"
          " resolution (Bouthillier\n  et al. 2021), against ~30% for P(A>B).")


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--folds", type=int, default=5,
                     help="K, the PARTITION size (changes fold membership)")
    ap_.add_argument("--max-folds", type=int, default=None,
                     help="run only the first N folds of the K-way partition "
                          "(coverage cut, keeps fold membership and the cache)")
    a = ap_.parse_args()

    print(f"K={a.folds} grouped CV over the pool, one partition for every model",
          flush=True)
    res = run_cv(".", n_splits=a.folds, max_folds=a.max_folds)
    report_cv(res)


if __name__ == "__main__":
    main()
