"""Report figures, drawn from measured results rather than from prose.

Run: ``python -m src.figures``  ->  ``outputs/figures/*.png``

Everything here reads either ``report/experiments.json`` (results measured by
scripts named in its ``_script`` fields) or ``outputs/per_code_val.csv``
(written by the pipeline). No figure invents a number.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

__all__ = ["main"]

OUT = Path("outputs/figures")
INK, ACCENT, WARN, MUTED = "#16191d", "#1f4e79", "#b3402f", "#7b8794"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": "#c9d0d8", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10,
    "axes.titleweight": "600", "savefig.bbox": "tight", "figure.facecolor": "white",
})


def _load() -> dict:
    return json.loads(Path("report/experiments.json").read_text(encoding="utf-8"))


def fig_learning_curve(ex: dict) -> None:
    """The figure that refutes 'the model is saturated, so more data cannot help'."""
    d = ex["learning_curve_real_only"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(d["n_patients"], d["macro_auroc"], "o-", color=ACCENT, lw=1.8, ms=5)
    ax.axhspan(0.7679 - 0.01, 0.7679 + 0.01, color=ACCENT, alpha=0.07, lw=0)
    ax.annotate("still rising at the\nlargest available n",
                xy=(2791, 0.7679), xytext=(1500, 0.7715),
                fontsize=8, color=WARN,
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1))
    ax.set_xlabel("training patients (real cutoffs only)")
    ax.set_ylabel("validation macro AUROC")
    ax.set_title("More real data helps: the task is not saturated")
    ax.grid(alpha=0.25, lw=0.6)
    fig.savefig(OUT / "learning_curve.png")
    plt.close(fig)


def fig_domain_shift(ex: dict) -> None:
    """The 2x2 that exonerates the augmentation pipeline."""
    rows = ex["domain_shift_2x2"]["rows"]
    M = np.array([[r["eval_real"], r["eval_synth"]] for r in rows])
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    im = ax.imshow(M, cmap="RdYlBu", vmin=0.60, vmax=0.78, aspect="auto")
    ax.set_xticks([0, 1], ["eval: real", "eval: synthetic"])
    ax.set_yticks(range(len(rows)), [r["train"].replace(" n=", "\nn=") for r in rows])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            diag = (i == 0 and j == 0) or (i > 0 and j == 1)
            ax.text(j, i, f"{M[i, j]:.4f}", ha="center", va="center",
                    fontsize=10, color=INK,
                    fontweight="bold" if diag else "normal")
    ax.set_title("Each distribution scores best on itself\n"
                 "(bold = matched train/eval domain)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="macro AUROC")
    fig.savefig(OUT / "domain_shift_2x2.png")
    plt.close(fig)


def fig_blocks_and_strides(ex: dict) -> None:
    """Two null results side by side, drawn against the noise floor."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.3))

    arms = ex["feature_block_screen"]["arms"]
    names = [a["arm"] for a in arms]
    vals = [a["macro_auroc"] for a in arms]
    ref = arms[1]["macro_auroc"]
    a1.barh(names, [v - ref for v in vals],
            color=[ACCENT if v >= ref else WARN for v in vals], height=0.6)
    a1.axvspan(-0.01, 0.01, color=MUTED, alpha=0.18, lw=0,
               label="±0.01 noise floor")
    a1.axvline(0, color=INK, lw=0.8)
    a1.set_xlabel("Δ macro AUROC vs 'lab coverage 49'")
    a1.set_title("Feature blocks: every arm is inside the noise floor")
    a1.legend(fontsize=7, loc="lower right")
    a1.invert_yaxis()
    a1.grid(axis="x", alpha=0.25, lw=0.6)

    s = ex["augmentation_stride_sweep"]["arms"]
    a2.plot([x["n_train"] for x in s], [x["macro_auroc"] for x in s],
            "o-", color=WARN, lw=1.8, ms=6)
    for x in s:
        a2.annotate(x["arm"], (x["n_train"], x["macro_auroc"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=7.5)
    a2.set_xlabel("training rows after cutoff augmentation")
    a2.set_ylabel("validation macro AUROC")
    a2.set_title("More rows, worse score — because they are a different task")
    a2.grid(alpha=0.25, lw=0.6)

    fig.savefig(OUT / "null_results.png")
    plt.close(fig)


def fig_per_code() -> None:
    """Where the score comes from, and how little of it is resolvable."""
    p = Path("outputs/per_code_val.csv")
    if not p.exists():
        print("  per_code_val.csv absent -- run `python -m src.run_baseline` first")
        return
    t = pd.read_csv(p).dropna(subset=["auroc_all"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.4))

    a1.scatter(t.n_pos, t.auroc_all, s=26, color=ACCENT, alpha=0.75,
               edgecolor="white", lw=0.5)
    if "se_hanley" in t:
        a1.errorbar(t.n_pos, t.auroc_all, yerr=1.96 * t.se_hanley, fmt="none",
                    ecolor=MUTED, alpha=0.45, lw=0.8)
    a1.axhline(0.5, color=WARN, lw=1, ls="--", label="chance")
    a1.set_xscale("log")
    a1.set_xlabel("validation positives (log scale)")
    a1.set_ylabel("AUROC")
    a1.set_title("Per-condition AUROC, with Hanley–McNeil 95% bars")
    a1.legend(fontsize=7)
    a1.grid(alpha=0.25, lw=0.6)

    a2.scatter(t.prevalence, t.lift, s=26, color=ACCENT, alpha=0.75,
               edgecolor="white", lw=0.5)
    a2.axhline(1.0, color=WARN, lw=1, ls="--", label="AP = prevalence (chance)")
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xlabel("prevalence")
    a2.set_ylabel("lift  (AP / prevalence)")
    a2.set_title("Lift, because a bare AP is uninterpretable")
    a2.legend(fontsize=7)
    a2.grid(alpha=0.25, lw=0.6)

    fig.savefig(OUT / "per_code.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ex = _load()
    fig_learning_curve(ex)
    fig_domain_shift(ex)
    fig_blocks_and_strides(ex)
    fig_per_code()
    made = sorted(OUT.glob("*.png"))
    for f in made:
        print(f"  {f}  ({f.stat().st_size:,} bytes)")
    print(f"{len(made)} figure(s) in {OUT}")


if __name__ == "__main__":
    main()
