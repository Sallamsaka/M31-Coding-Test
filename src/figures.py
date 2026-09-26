"""Report figures, drawn from measured results rather than from prose.

Run: ``python -m src.figures``  ->  ``outputs/figures/*.png``

Every figure reads a file written by the pipeline or by an experiment script.
No figure invents a number.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.ticker                 # noqa: E402,F401  (MaxNLocator)
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


# The three seeds of the shipped transformer, as re-fitted with wandb on by
# `src.log_shipped_run` (which verified the refit reproduces the shipped matrix).
# Their entries are the only ones for these run ids that carry `val_loss`: the
# field was added for that refit, so filtering on it excludes the original fit.
SHIPPED_TX_RUNS = ("P4_seed300_5221b458", "P4_seed301_07316ad3", "P4_seed302_08149c73")


def fig_training_curves() -> None:
    """Training and validation curves of the shipped transformer, per seed.

    Read from ``outputs/metrics.jsonl`` -- the source of truth, and the same
    per-epoch rows wandb received -- so the figure is reproducible offline.
    """
    rows = [json.loads(line) for line in
            Path("outputs/metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    df = pd.DataFrame([r for r in rows
                       if r.get("run") in SHIPPED_TX_RUNS and "val_loss" in r])
    # The final fit and the wandb refit log under the same run id (same config,
    # same seed; the refit is verified bit-identical). Keep one row per epoch --
    # the latest -- so two runs can never be drawn as one zig-zagging line.
    if not df.empty:
        df = df.drop_duplicates(["run", "epoch"], keep="last")
    if df.empty:
        print("  training_curves: no logged refit in outputs/metrics.jsonl -- skipped")
        return
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    colours = (ACCENT, WARN, "#4a7a3a")
    for (run, g), c in zip(df.groupby("run", sort=True), colours):
        g = g.sort_values("epoch")
        seed = run.split("_")[1].replace("seed", "")
        # Replay the training loop's own rule, not argmax: an epoch counts as an
        # improvement only if it beats the best by more than min_delta = 0.002.
        # argmax would mark seed 300 at epoch 9; the loop selected epoch 6.
        best, best_ap = 0, -1.0
        for ep, ap_ in zip(g.epoch, g.macro_ap):
            if ap_ > best_ap + 0.002:
                best, best_ap = int(ep), ap_
        axes[0].plot(g.epoch, g.loss, "-", color=c, lw=1.5, label=f"train, seed {seed}")
        axes[0].plot(g.epoch, g.val_loss, "--", color=c, lw=1.5, label=f"val, seed {seed}")
        axes[1].plot(g.epoch, g.macro_auroc, "o-", color=c, lw=1.4, ms=3)
        axes[2].plot(g.epoch, g.macro_ap, "o-", color=c, lw=1.4, ms=3, label=f"seed {seed}")
        for ax in axes:
            ax.axvline(best, color=c, lw=0.8, alpha=0.35)
    axes[0].set_title("Masked BCE loss (solid train, dashed val)")
    axes[0].set_ylabel("loss")
    # Epoch 1's training loss (~0.38) is the average over a randomly initialised
    # model; left in view it flattens everything that matters into the bottom sixth.
    axes[0].set_ylim(0.08, 0.20)
    axes[1].set_title("Validation macro AUROC")
    axes[2].set_title("Validation macro AP")
    axes[2].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(alpha=0.25, lw=0.6)
    fig.suptitle("Shipped transformer, three seeds (wandb group shipped-transformer-age)."
                 "  Thin vertical line: epoch selected by early stopping.",
                 fontsize=9, fontweight="600", y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png")
    plt.close(fig)


ACUTE = ("fracture", "sprain", "laceration", "whiplash", "concussion", "sinusitis",
         "bronchitis", "pharyngitis", "sore throat")


def fig_per_condition_cv() -> None:
    """Per-condition AUROC of the submitted blend on the 5-fold test folds."""
    p = Path("outputs/per_condition_cv.csv")
    if not p.exists():
        print("  per_condition_cv.csv absent -- skipped")
        return
    t = pd.read_csv(p).sort_values("auroc")
    nm = t.name.str.lower()
    # "Pathological fracture due to osteoporosis" is bone fragility, not an injury.
    acute = nm.str.contains("|".join(ACUTE)) & ~nm.str.contains("pathological")
    fig, ax = plt.subplots(figsize=(7.2, 7.4))
    ax.barh(t.name.str.replace(r" \((disorder|finding|situation)\)", "", regex=True),
            t.auroc - 0.5, left=0.5, height=0.7,
            color=[WARN if a else ACCENT for a in acute])
    ax.axvline(0.5, color=INK, lw=0.8)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("test-fold AUROC (5-fold CV, 2,791 training patients)")
    ax.set_title("Per-condition AUROC; acute one-off events in red")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    fig.savefig(OUT / "per_condition_cv.png")
    plt.close(fig)


def fig_time_attention() -> None:
    """How strongly the trained transformer attends to one event by its age."""
    import math
    import torch
    from .models.gpt import GPTConfig, PatientTransformer
    ck = Path("artifacts/model_P4_seed300_5221b458.pt")
    if not ck.exists():
        print("  shipped seed-300 checkpoint absent -- skipped")
        return
    d = torch.load(ck, map_location="cpu", weights_only=False)
    m = PatientTransformer(GPTConfig(**d["gpt_config"]))
    m.load_state_dict(d["state_dict"]); m.eval()
    itos = json.loads(Path("artifacts/vocab.json").read_text())["itos"]
    stoi = {t: i for i, t in enumerate(itos)}
    ev, an, age0 = stoi["OBS_8480-6_Q5"], stoi["[ANCHOR]"], 60 * 365.25
    B = m.blocks[0]

    def score(dt, head, line_only):
        with torch.no_grad():
            def vec(tok, t):
                f = m.time(torch.tensor([[float(t)]]))
                if line_only:
                    f = f.clone(); f[..., 1:] = 0.0
                return m.mix(torch.cat([m.tok(torch.tensor([[tok]])), f,
                                        m.age(torch.tensor([[age0]]))], -1))
            qa = (B.ln1(vec(an, 0)) @ B.qkv.weight.T)[..., :64]
            ke = (B.ln1(vec(ev, dt)) @ B.qkv.weight.T)[..., 64:128]
            s = slice(32 * head, 32 * head + 32)
            return float((qa[..., s] * ke[..., s]).sum() / math.sqrt(32))

    days = np.unique(np.round(np.logspace(0, 4, 60)))
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for line_only, c, lab in ((False, ACCENT, "learned time encoding"),):
        s = np.array([score(x, 1, line_only) for x in days])
        ref = score(90, 1, line_only)
        ax.plot(days, np.exp(s - ref), color=c, lw=1.8, label=lab)
    ax.set_xscale("log")
    ax.axhline(1.0, color=INK, lw=0.6, ls=":")
    ax.set_xlabel("days between the event and the anchor (log scale)")
    ax.set_ylabel("attention relative to 90 days")
    ax.set_title("Head 2: how much one blood-pressure reading counts, by when it happened")

    ax.grid(alpha=0.25, lw=0.6)
    fig.savefig(OUT / "time_attention.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_training_curves()
    fig_per_condition_cv()
    fig_time_attention()
    made = sorted(OUT.glob("*.png"))
    for f in made:
        print(f"  {f}  ({f.stat().st_size:,} bytes)")
    print(f"{len(made)} figure(s) in {OUT}")


if __name__ == "__main__":
    main()
