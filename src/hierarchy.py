"""Clinical families over the 40 targets, and James-Stein shrinkage toward them.

**Why this exists, and why the reason given was wrong.** The module was built on
the claim that the binding constraint is rare *labels*: the median target has ~11
positives and ten of the 40 sit between 7 and 16, so a per-label score fitted on
8 positives is mostly noise, and macro AP weights all 40 equally.

⚠ **Those are VALIDATION counts (365 patients). The scores are fitted on TRAIN,
2,791 patients**, where the rarest label has **29** positives and the median is
**86** -- 7.6x larger. Quoting a property of one set to argue about estimation
noise in another is what motivated this, and it does not hold up.

§E44 measured the consequence: shrinkage is **monotonically harmful** on all
three models (-0.024 AP for LR and the transformer at strength 4; GBDT's best is
+0.0001). The grouping below is real -- within-family label correlation is 18x
the across-family value -- but a valid grouping is not sufficient when the
per-label estimates are not noisy enough to need borrowing.

The module is kept, tested and wired to `python -m src.hierarchy` because the
negative is worth being able to reproduce, and because the grouping itself may
be useful for something else (per-family reporting, error analysis). It is NOT
part of the shipped model.

Shrinkage is the standard answer and it needs a grouping. §E37.2 established what
we may use: the assignment owner permitted the SNOMED/ICD hierarchy (2026-09-22),
but the data carries SNOMED concept IDs (`162864005`), LOINC (`8302-2`) and
RxNorm (`2001499`) -- opaque integers whose IS-A graph lives in a UMLS-licensed
release we do not have, and ICD-10, the one prefix-structured system, does not
appear in this dataset at all. With only 147 distinct condition codes the
practical answer is a hand-built grouping over the 40 targets, which is auditable
in a way a downloaded graph would not be.

**The grouping is a hypothesis, not a fact**, so `family_coherence` measures it
rather than asserting it: labels in a family should co-occur more than labels
across families. If they do not, the grouping is wrong and shrinkage toward it
would drag every member toward an unrelated mean.
"""
from __future__ import annotations

import argparse

import numpy as np

__all__ = ["FAMILIES", "family_of", "family_matrix", "family_coherence",
           "shrink_to_family", "evaluate"]

# Codes are SNOMED CT, as they appear in `conditions.csv`. Positive counts in
# the comments are validation-set counts (`outputs/per_code_val.csv`) and are
# there to show WHY a family exists -- a family of one rare label buys nothing.
FAMILIES: dict[str, list[int]] = {
    # 5 to 99 positives. The widest spread of any family, which is the point:
    # bacterial sinusitis (5) borrows from viral sinusitis (99).
    "respiratory_infection": [
        444814009,   # Viral sinusitis                     99
        195662009,   # Acute viral pharyngitis             72
        10509002,    # Acute bronchitis                    61
        43878008,    # Streptococcal sore throat           22
        233604007,   # Pneumonia                           17
        36971009,    # Sinusitis                            9
        75498004,    # Acute bacterial sinusitis            5
    ],
    "cardiovascular": [
        230690007,   # Stroke                              35
        88805009,    # Chronic congestive heart failure    29
        22298006,    # Myocardial infarction               19
        53741008,    # Coronary heart disease              15
        49436004,    # Atrial fibrillation                 14
        723857007,   # Silent micro-hemorrhage of brain    11
    ],
    # The largest family and the one with the most to gain: ten labels, none
    # above 16 positives, all of them acute injuries.
    "msk_trauma": [
        65966004,    # Fracture of forearm                 16
        44465007,    # Sprain of ankle                     16
        263102004,   # Fracture subluxation of wrist       10
        16114001,    # Fracture of ankle                    8
        58150001,    # Fracture of clavicle                 8
        70704007,    # Sprain of wrist                      8
        39848009,    # Whiplash injury to neck              8
        283371005,   # Laceration of forearm                7
        284551006,   # Laceration of foot                   7
        62106007,    # Concussion, no loss of consciousness 7
    ],
    # Kept apart from msk_trauma deliberately: these are consequences of bone
    # fragility, not of an accident, so their risk factors are the metabolic
    # ones. Merging them would pull a 14-positive label toward a mean driven by
    # sprained ankles.
    "bone_fragility": [
        443165006,   # Pathological fracture from osteoporosis  14
        64859006,    # Osteoporosis                             12
    ],
    "metabolic": [
        162864005,   # Body mass index 30+ obesity         11
        15777000,    # Prediabetes                         10
    ],
    "colorectal_polyp": [
        68496003,    # Polyp of colon                      18
        713197008,   # Recurrent rectal polyp              10
    ],
    "pregnancy": [
        72892002,    # Normal pregnancy                    10
        156073000,   # Fetus with unknown complication      5
    ],
    # Oncology is split rather than pooled. Prostate and lung cancer share
    # almost no risk factors -- one tracks age and sex, the other smoking -- so
    # a single "cancer" mean would be the average of two unrelated things. Three
    # members each is thin for shrinkage, which is the honest cost of not
    # pretending they are one family.
    "prostate_neoplasm": [
        126906006,   # Neoplasm of prostate                11
        92691004,    # Carcinoma in situ of prostate        8
        314994000,   # Metastasis from prostate tumour      6
    ],
    "lung_neoplasm": [
        162573006,   # Suspected lung cancer               10
        424132000,   # NSCLC TNM stage 1                    9
        254637007,   # Non-small cell lung cancer           9
    ],
    # Not a family in any clinical sense -- three unrelated labels with nothing
    # to borrow from. Named "singleton" rather than "other" so it is obvious
    # that shrinkage must LEAVE THESE ALONE; grouping unlike things together to
    # avoid an awkward leftover bucket is how shrinkage does harm.
    "singleton": [
        271737000,   # Anemia                              15
        26929004,    # Alzheimer's disease                 13
        22325002,    # Abnormal gait                       11
    ],
}


def family_of(codes) -> np.ndarray:
    """Family name per label column, ordered as `codes`."""
    lookup = {c: fam for fam, cs in FAMILIES.items() for c in cs}
    missing = [int(c) for c in codes if int(c) not in lookup]
    assert not missing, f"targets with no family: {missing}"
    return np.array([lookup[int(c)] for c in codes], dtype=object)


def family_matrix(codes) -> np.ndarray:
    """(n_labels, n_labels) bool: same family, excluding the diagonal."""
    fam = family_of(codes)
    same = fam[:, None] == fam[None, :]
    np.fill_diagonal(same, False)
    return same


def family_coherence(y: np.ndarray, at_risk: np.ndarray, codes) -> dict:
    """Do family members actually co-occur more than non-members?

    The grouping is clinical judgement, and clinical judgement about what
    co-occurs in **Synthea** is worth less than it would be in real data --
    these outcomes come from module logic, not from biology. So this is measured
    on the labels themselves before anything is shrunk toward anything.

    Returns mean within-family and across-family label correlation. A grouping
    worth using has within > across; if they are equal the families carry no
    information and shrinkage is just noise injection with extra steps.
    """
    Y = np.where(at_risk.astype(bool), y, np.nan).astype(float)
    n = Y.shape[1]
    C = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            m = ~np.isnan(Y[:, i]) & ~np.isnan(Y[:, j])
            if m.sum() < 20:
                continue
            a, b = Y[m, i], Y[m, j]
            if a.std() == 0 or b.std() == 0:
                continue
            C[i, j] = C[j, i] = float(np.corrcoef(a, b)[0, 1])
    same = family_matrix(codes)
    within = C[same & ~np.isnan(C)]
    across = C[~same & ~np.isnan(C)]
    np.fill_diagonal(C, np.nan)
    return {"within_mean": float(np.mean(within)) if len(within) else float("nan"),
            "across_mean": float(np.mean(across)) if len(across) else float("nan"),
            "n_within": int(len(within)), "n_across": int(len(across)),
            "gap": (float(np.mean(within) - np.mean(across))
                    if len(within) and len(across) else float("nan"))}


def shrink_to_family(P: np.ndarray, codes, n_pos: np.ndarray,
                     strength: float = 1.0) -> np.ndarray:
    """Shrink each label's LOGITS toward its family mean logit.

    Model-agnostic on purpose: it operates on predictions, so the same transform
    applies to LR, GBDT and the transformer, and to their ensemble.

    The weight is James-Stein in spirit -- a label with many positives keeps its
    own signal, a label with few borrows from its family::

        w_label = n_pos / (n_pos + strength * mean_n_pos_in_family)

    so `w -> 1` for well-estimated labels and `w -> 0` for the 5-positive ones.
    `strength = 0` is the identity, which is what makes this testable against
    doing nothing on the same held-out folds.

    ⚠ This is NOT monotone per label, so unlike Platt scaling it **can** change
    macro AUROC and macro AP in either direction. That is the point -- it is
    meant to move the ranking -- but it also means it must be validated
    out-of-fold rather than assumed safe. Calibration was free insurance; this
    is not.

    Singletons are returned untouched: a family of unrelated labels has no mean
    worth borrowing.
    """
    fam = family_of(codes)
    lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) /
                          (1 - np.clip(q, 1e-6, 1 - 1e-6)))
    L = lg(P).astype(float)
    out = L.copy()
    for f in set(fam.tolist()):
        idx = np.flatnonzero(fam == f)
        if f == "singleton" or len(idx) < 2:
            continue
        mean_logit = L[:, idx].mean(axis=1, keepdims=True)
        base = float(np.mean(n_pos[idx]))
        for j in idx:
            w = n_pos[j] / (n_pos[j] + strength * base) if base > 0 else 1.0
            out[:, j] = w * L[:, j] + (1.0 - w) * mean_logit[:, 0]
    return 1.0 / (1.0 + np.exp(-out))


# Swept rather than tuned: `strength` has no prior, and the honest form of the
# question is "does ANY setting help, and is the best one interior or at a
# boundary?" A win at the edge of the grid is not an optimum (the lesson the
# lr arms learned at 3.6e-3).
STRENGTHS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)


def evaluate(root: str = ".", oof_path: str = "artifacts/cv_oof_preds.npz",
             models=("lr", "gbdt", "transformer")) -> "object":
    """Sweep shrinkage strength on OUT-OF-FOLD predictions, per model.

    Out-of-fold is the right instrument and the only honest one available: it
    is the rotating train/test design (2,433+ patients scored once each), not
    the single screening fold, and shrinkage is exactly the kind of transform
    that would look good on the set that chose it.

    **`n_pos` comes from TRAINING rows, never from the rows being scored.** The
    shrinkage weight is a function of per-label positive counts, so taking them
    from the evaluation set would leak label information into the transform --
    weakly, since only counts are used, but there is no reason to accept even
    that when the training split gives the same quantity for free.
    """
    import pandas as pd

    from .data.examples import ExampleConfig, load_examples
    from .data.labels import at_risk_mask, load_labels
    from .evaluate import macro_ap, macro_auroc

    d = np.load(oof_path)
    lab = load_labels(root, ExampleConfig())
    y, ar = lab["y"].astype(int), at_risk_mask(lab)
    codes = np.asarray(lab["codes"])
    split = load_examples(root, ExampleConfig()).split.to_numpy()

    scored = d["scored"].astype(bool)
    tr = split == "train"
    n_pos = (y[tr] * ar[tr]).sum(axis=0).astype(float)

    idx = np.flatnonzero(scored)
    print(f"out-of-fold rows scored: {len(idx):,}")
    print(f"n_pos per label from TRAIN split: min {int(n_pos.min())} "
          f"median {int(np.median(n_pos))} max {int(n_pos.max())}\n")

    rows = []
    for mname in models:
        if mname not in d:
            print(f"  (skipping {mname}: not in {oof_path})")
            continue
        P = d[mname]
        for st in STRENGTHS:
            Q = shrink_to_family(P, codes, n_pos, strength=st)
            ap, _ = macro_ap(y[idx], Q[idx], mask=ar[idx])
            au, _ = macro_auroc(y[idx], Q[idx], mask=ar[idx])
            rows.append({"model": mname, "strength": st, "ap": ap, "auroc": au})
    df = pd.DataFrame(rows)

    print(f"{'model':>13}{'strength':>10}{'macro AP':>11}{'dAP':>10}"
          f"{'macro AUROC':>14}{'dAUROC':>10}")
    for mname, g in df.groupby("model", sort=False):
        base = g[g.strength == 0.0].iloc[0]
        for r in g.itertuples():
            print(f"{r.model:>13}{r.strength:>10.2f}{r.ap:>11.4f}"
                  f"{r.ap - base.ap:>+10.4f}{r.auroc:>14.4f}"
                  f"{r.auroc - base.auroc:>+10.4f}")
        best = g.loc[g.ap.idxmax()]
        edge = best.strength in (min(STRENGTHS), max(STRENGTHS))
        print(f"    best {mname}: strength {best.strength:g}, "
              f"dAP {best.ap - base.ap:+.4f}"
              + ("   \u26a0 AT THE EDGE of the grid -- not an optimum"
                 if edge and best.strength != 0.0 else "")
              + ("   (= no shrinkage)" if best.strength == 0.0 else ""))
        print()
    df.to_csv("outputs/hierarchy_shrinkage.csv", index=False)
    print("wrote outputs/hierarchy_shrinkage.csv")
    return df


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--oof", default="artifacts/cv_oof_preds.npz")
    ap_.add_argument("--models", nargs="+",
                     default=["lr", "gbdt", "transformer"])
    a = ap_.parse_args()
    evaluate(".", a.oof, tuple(a.models))


if __name__ == "__main__":
    main()
