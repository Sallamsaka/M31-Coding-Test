"""Fine-tune the patient transformer. CPU-only, length-bucketed, four arms.

Length-bucketed batching
------------------------
The median sequence is 198 tokens against a 512 block, so a fixed-width batch
spends roughly 60% of every forward pass attending to padding. Sorting by
length and batching neighbours makes a batch cost its own maximum instead of
the global one. Per layer the work is ``12 d^2 T`` for the projections and
``2 d T^2`` for attention, so cost tracks ``E[T]`` and ``E[T^2]`` rather than
``512`` and ``512^2`` -- about 2.5x on this length distribution.

The price is that batches are no longer randomly composed, which correlates
the examples inside a gradient step. The standard fix, applied here: shuffle
*within* each bucket every epoch and shuffle the order of the buckets, so
composition is stochastic across epochs while padding stays low.

Loss masking
------------
A patient already diagnosed with condition *c* cannot become an incident case
for *c*. Those pairs are excluded from the loss rather than their rows being
deleted: the row still carries 39 other labels and still trains the shared
trunk. This mirrors Steinfeldt et al. (Nat Commun 2025) across 1,741
endpoints -- all individuals train the shared representation, prevalent
individuals are dropped from the endpoint-specific term.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data.cohort import load_cohort
from .data.examples import ExampleConfig
from .data.labels import at_risk_mask, load_labels
from .data.sequences import MASK, SeqConfig, build_sequences, build_vocab
from .evaluate import macro_ap, macro_auroc
from .models.gpt import GPTConfig, PatientTransformer
from .utils import wandb_shim

__all__ = ["TrainConfig", "ARMS", "train", "pretrain"]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 6e-4
    weight_decay: float = 0.1
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    seed: int = 0
    threads: int = 8
    pretrain_epochs: int = 8
    eval_every: int = 1

    min_delta: float = 0.0
    """How much dev AP must IMPROVE to count as an improvement.

    Without a band, `apv > best` counts a 1e-6 wobble as progress, which resets
    the patience counter and means patience effectively never fires on a noisy
    plateau -- we keep paying for epochs that are chasing evaluation noise, and
    the checkpoint lands on whichever noise peak happened to be highest.

    The band should be set from the measured epoch-to-epoch noise in the metric,
    which is what `design --seeds` reports as `sigma_ap`. Until that is measured
    it stays 0.0, preserving the original behaviour exactly, and the experiment
    configs set it explicitly.
    """

    ema_decay: float = 0.8
    """Per-EPOCH decay for an exponential moving average of the weights.

    Measured motivation, not a convention: epoch-to-epoch macro AP wobbles with
    sigma ~ 0.0175, which is LARGER than the seed-to-seed sigma of 0.0080, and
    annealing the learning rate 1.6x changes that bounce by 0.91x -- so it is not
    step size, it is gradient noise (batch 32 is 1.32% of the corpus, an epoch is
    77 steps). Selecting the best of ~17 such epochs inflates the reported AP by
    sigma*sqrt(2 ln 17) ~ 0.042, which exceeds the largest model difference this
    project has ever resolved.

    An average over the trajectory does not bounce, so there is nothing to
    select. decay 0.8 gives an effective window of 1/(1-0.8) = 5 epochs, which
    takes the effective sigma to ~0.0078.

    This is the mechanism Schedule-Free uses, without the optimizer swap: SF
    evaluates gradients at an interpolation involving the average, so the average
    feeds back into the trajectory; this averages only for scoring. The
    benchmarks that put SF-AdamW behind AdamW+cosine at our scale measured the
    whole SF package, never the averaging in isolation.

    0.0 disables it.
    """

    resume: bool = True
    """Restore from `artifacts/ckpt_last_{arm}_seed{seed}.pt` if one exists.

    The checkpoint is written every eval epoch and deleted on clean completion,
    so one only exists if a run died. It carries optimiser moments, the step
    counter and the RNG states, so a resumed run continues the cosine schedule
    and the batch order rather than silently restarting them.
    """

    patience: int = 0
    """Stop after this many epochs with no improvement in validation AP.
    0 disables it and `epochs` becomes a hard budget.

    Budgeting by a fixed epoch count silently biases a hyperparameter search
    toward whatever converges fastest. Measured here: at a 20-epoch cap the
    1-layer configs peaked at epochs 19 and 20 -- still improving -- while
    2-layer/dropout-0.3 peaked at 14. Comparing those is comparing a finished
    run against an unfinished one. With patience, every config trains until
    it stops improving, so the comparison is between converged models.
    """

    # --- time-signal ablation (W7) -----------------------------------------
    use_time_encoding: bool = True
    """Per-token Time2Vec on/off. See GPTConfig for what off means."""
    use_dt_bias: bool = True
    """Pairwise Δt attention bias on/off.

    Both zero a signal rather than deleting a tensor, so every ablation arm has
    the same parameter count and the comparison measures the component instead
    of the capacity.
    """

    # --- capacity and regularisation, searchable ---------------------------
    n_layer: int = 4
    n_embd: int = 192
    n_head: int = 6
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1

    # --- feature fusion ----------------------------------------------------
    fusion: str = "none"
    """``none`` | ``readout`` (late fusion) | ``token`` (prepended, attended).

    Late fusion concatenates a projection of the engineered features to the
    pooled sequence summary, which is a jointly-trained ensemble. The token
    form prepends the projection as an extra position so attention can mix
    feature evidence with sequence evidence. They test different claims and
    the second is the one that would justify calling the result integrated
    rather than bolted on.
    """
    fusion_dim: int = 64
    """Deliberately small. A full 3,320 -> 192 projection is 637k parameters,
    a third of the model again, fitted on 2,791 examples. At 64 it is 212k and
    still the largest single block outside the trunk."""

    # --- evaluation target -------------------------------------------------
    dev_frac: float = 0.0
    """Fraction of the provided TRAIN patients held out for TUNING."""

    holdout_frac: float = 0.0
    """Fraction of the provided TRAIN patients kept as OUR OWN test set.

    Why this has to exist. The organisers' test split has no labels --
    `test/conditions.csv` contains zero rows at or after the anchor, so
    `y[test]` is all zeros by construction and no test metric is computable
    on our side. And the provided validation set has now been read dozens of
    times for model selection, which makes it a selection set carrying the
    winner's curse, not an unbiased estimate. Without a third split there is
    **no clean generalisation number anywhere in this project.**

    Why it comes out of TRAIN rather than from pooling everything: the brief
    says "do not train on validation or test", so the 365 provided
    validation patients cannot enter a fit set. The 2,791 training patients
    are ours to cut, and cutting them three ways costs fit data but buys the
    only honest estimate available.

    It is also what makes "which metric generalises better" answerable at
    all: choose a configuration by dev AP and by dev AUROC, then see which
    choice actually wins on data that informed neither.
    """

    eval_on: str = "auto"
    """Which split `train()` reports: ``auto`` picks holdout if present, else
    dev, else the provided validation set."""


# The four coherent arms. Readout follows the mask, and the pretraining
# objective follows the mask, so these are the only combinations that are
# internally consistent -- see docs and the plan's collapse argument.
ARMS = {
    "P1": dict(causal=True,  readout="last", pretrain="lm"),
    "P2": dict(causal=False, readout="mean", pretrain="mlm"),
    "P3": dict(causal=False, readout="mean", pretrain=None),
    "P4": dict(causal=True,  readout="last", pretrain=None),
}


def bucketed_batches(lengths: np.ndarray, idx: np.ndarray, batch_size: int,
                     rng: np.random.Generator, shuffle: bool = True
                     ) -> list[np.ndarray]:
    """Batches of similar-length rows, in random order, reshuffled per epoch."""
    order = idx[np.argsort(lengths[idx], kind="stable")]
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        for b in batches:
            rng.shuffle(b)
        rng.shuffle(batches)
    return batches


def _trim(tokens: np.ndarray, dt: np.ndarray, lengths: np.ndarray,
          rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cut the batch to its own longest sequence. This is the whole saving."""
    m = int(lengths[rows].max())
    return (torch.from_numpy(tokens[rows, :m].astype(np.int64)),
            torch.from_numpy(dt[rows, :m].astype(np.float32)),
            torch.from_numpy(lengths[rows].astype(np.int64)))


@torch.no_grad()
def predict(model: PatientTransformer, tokens, dt, lengths, rows,
            batch_size: int = 64, features: np.ndarray | None = None) -> np.ndarray:
    model.eval()
    out = []
    order = rows[np.argsort(lengths[rows], kind="stable")]
    for i in range(0, len(order), batch_size):
        b = order[i:i + batch_size]
        t, d, L = _trim(tokens, dt, lengths, b)
        fb = torch.from_numpy(features[b]) if features is not None else None
        out.append((b, torch.sigmoid(model(t, d, L, features=fb)).numpy()))
    P = np.zeros((len(rows), out[0][1].shape[1]), np.float32)
    pos = {r: k for k, r in enumerate(rows)}
    for b, p in out:
        for k, r in enumerate(b):
            P[pos[r]] = p[k]
    return P


def pretrain(model: PatientTransformer, pack, rows: np.ndarray, objective: str,
             cfg: TrainConfig, vocab, epochs: int = 8,
             mask_frac: float = 0.15, verbose: bool = True,
             max_steps: int | None = None) -> None:
    """Self-supervised warm-up on pre-cutoff tokens of TRAINING examples only.

    Two traps, both of which produce a model that scores well on the
    pretraining objective and has learned nothing.

    **Next-token must predict position i+1 from position i, never position i
    from itself.** ``_attn_bias`` always ORs the identity into the visibility
    mask, so a position can always see its own token -- necessary for the
    classification pass (otherwise a row could be entirely masked out and the
    softmax would be all -inf) but fatal for an unshifted LM objective, where
    the model would simply copy its input. Targets are therefore shifted.

    **Masked-token must score only the masked positions.** Scoring everything
    lets the unmasked majority dominate the loss with a copy task.

    Expected outcome: null. The one study of this exact protocol -- SSL on the
    same small training split with no external corpus -- measured +0.006 AUC,
    p=0.63. Our pretraining corpus is ~0.8M tokens; BabyLM's smallest track is
    10M words, and the only scaling law fitted to medical events (~1000:1
    tokens per parameter) would support a model of roughly 800 parameters.
    Running it is how that becomes a result instead of an assumption.
    """
    mask_id = vocab.stoi[MASK]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95),
                            weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed + 1000)
    model.train()
    t0 = time.time()
    for epoch in range(epochs):
        tot, nb = 0.0, 0
        for batch in bucketed_batches(pack.lengths, rows, cfg.batch_size, rng):
            tok, dt, _ = _trim(pack.tokens, pack.dt, pack.lengths, batch)
            real = tok != 0
            if objective == "lm":
                logits = model.lm_forward(tok, dt)
                # position i predicts token i+1
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    tok[:, 1:].reshape(-1),
                    ignore_index=0)
            else:
                sel = torch.from_numpy(
                    rng.random(tok.shape) < mask_frac) & real
                if not sel.any():
                    continue
                corrupted = tok.masked_fill(sel, mask_id)
                logits = model.mlm_forward(corrupted, dt)
                loss = F.cross_entropy(logits[sel], tok[sel])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += float(loss); nb += 1
            if max_steps is not None and nb >= max_steps:
                break
        if verbose:
            print(f"  pretrain[{objective}] epoch {epoch+1}/{epochs} "
                  f"loss {tot/max(nb,1):.4f}  [{(time.time()-t0)/60:.0f} min]",
                  flush=True)
        if max_steps is not None:
            break


def inner_split_pids(root: str | Path, dev_frac: float, holdout_frac: float,
                     seed: int = 12345) -> tuple[set[int], set[int]]:
    """``(dev_pids, hold_pids)`` carved out of TRAIN, as a pure function.

    Pulled out of ``train`` so it can be called **before** the vocabulary is
    built. It used to be computed from ``pack.pid``, which forced the ordering
    vocab -> sequences -> split, and that ordering is precisely bug D17: the
    vocabulary saw the dev patients because the split did not exist yet.

    The permutation is over the sorted unique train pids from the cohort, which
    is verified to be the identical array ``np.unique(pack.pid[tr])`` returned,
    so the partition is bit-identical to the one every prior run used. The seed
    is fixed at 12345 and deliberately not ``cfg.seed``: every config compared
    must see the same split, while training randomness varies.
    """
    coh = load_cohort(root)
    pids = np.sort(coh.loc[coh.split == "train", "pid"].unique())
    order = np.random.default_rng(seed).permutation(pids)
    n_dev = int(round(dev_frac * len(pids)))
    n_hold = int(round(holdout_frac * len(pids)))
    return (set(order[:n_dev].tolist()),
            set(order[n_dev:n_dev + n_hold].tolist()))


class WeightEMA:
    """Exponential moving average of a state_dict, for scoring.

    Initialised to the FIRST state it sees, which is already unbiased -- so
    there is deliberately no Adam-style 1/(1-d^t) correction. Applying both
    conventions inflates: feeding constant weights returned 5.0x them at the
    first step, then 2.8x, 2.0x. That bug shipped in the closure version and was
    caught only because this logic became testable.

    Non-float buffers (integer counters and the like) are copied, not averaged.
    """

    def __init__(self, decay: float = 0.8):
        self.decay = float(decay)
        self.state: dict | None = None
        self.seen = 0

    def update(self, sd: dict) -> dict:
        d = self.decay
        if self.state is None:
            # Only FLOAT tensors become float. Blanket .float() here promoted
            # integer buffers, which then satisfied is_floating_point() on the
            # next call and got averaged -- silently corrupting any counter or
            # index buffer the model carries.
            self.state = {k: (v.detach().clone().float()
                              if v.is_floating_point() else v.detach().clone())
                          for k, v in sd.items()}
        else:
            for k, v in sd.items():
                if self.state[k].is_floating_point():
                    self.state[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
                else:
                    self.state[k] = v.detach().clone()
        self.seen += 1
        return dict(self.state)


def _ckpt_path(root, arm: str, seed: int, fingerprint: str = "") -> Path:
    """Keyed by CONFIGURATION, not just arm and seed.

    The basin sweep runs five learning rates as arm=P4, seed=0, so under the old
    name they shared one checkpoint file. The fingerprint check meant a wrong
    resume was refused rather than silently accepted -- but the refusing run then
    OVERWROTE the file, so a genuinely crashed run could never resume once a
    sibling configuration had started. Including the fingerprint gives each
    configuration its own slot.
    """
    tag = f"_{fingerprint[:8]}" if fingerprint else ""
    return Path(root) / "artifacts" / f"ckpt_last_{arm}_seed{seed}{tag}.pt"


def _config_fingerprint(cfg: "TrainConfig", arm: str, vocab_size: int) -> str:
    """What a resume must match. Resuming into a different architecture would
    load the wrong tensors; resuming into a different schedule would continue a
    cosine curve computed for a different horizon. Both fail silently, so the
    fingerprint is checked rather than trusted."""
    import hashlib
    keys = ("n_layer", "n_embd", "n_head", "attn_dropout", "resid_dropout",
            "fusion", "fusion_dim", "lr", "weight_decay", "epochs",
            "batch_size", "warmup_frac", "seed", "dev_frac", "holdout_frac",
            "use_time_encoding", "use_dt_bias")
    blob = "|".join(f"{k}={getattr(cfg, k)}" for k in keys)
    blob += f"|arm={arm}|vocab={vocab_size}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _save_resume(path: Path, *, model, opt, epoch: int, step: int, best: dict,
                 rng, fingerprint: str) -> None:
    """Write the FULL training state, atomically.

    Atomically because the entire point is surviving a run that died: a plain
    torch.save interrupted mid-write leaves a truncated file that cannot be
    loaded, which is precisely the situation the checkpoint exists for. Write to
    a temp file, then os.replace, which is atomic on both POSIX and Windows.
    """
    import os
    path.parent.mkdir(exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "epoch": epoch, "step": step,
        "best": {k: v for k, v in best.items()},
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": rng.bit_generator.state,
        "fingerprint": fingerprint,
        "gpt_config": asdict(model.cfg),
    }
    # The temp name carries the PID. Two processes running the SAME experiment
    # would otherwise write the same `.tmp` and corrupt each other's payload
    # before either rename -- which is exactly what happened when an orphaned
    # chain re-ran `--seeds 5` alongside the live one (§D21).
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(payload, tmp)

    # os.replace is atomic on POSIX but fails on Windows with WinError 32 if
    # ANYTHING holds a handle on the destination -- a virus scanner reading the
    # freshly-written file, or another process writing the same path. Retry,
    # then give up QUIETLY.
    #
    # A checkpoint is insurance. Insurance failing must never destroy the thing
    # it insures, and the first version of this raised straight through and
    # killed a 36-minute training run over a file lock.
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.5 * (attempt + 1))
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    print(f"  [warn] could not update {path.name}; training continues without "
          f"a resume point this epoch", flush=True)


def _try_resume(path: Path, *, model, opt, rng, fingerprint: str, verbose: bool):
    """Returns (start_epoch, step, best) -- (0, 0, fresh) if no usable state."""
    fresh = (0, 0, {"macro_ap": -1.0})
    if not path.exists():
        return fresh
    try:
        ck = torch.load(path, weights_only=False)
    except Exception as e:                      # truncated or unreadable
        if verbose:
            print(f"  resume: checkpoint unreadable ({e}); starting fresh",
                  flush=True)
        return fresh
    if ck.get("fingerprint") != fingerprint:
        if verbose:
            print("  resume: checkpoint is for a DIFFERENT configuration; "
                  "ignoring it and starting fresh", flush=True)
        return fresh
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    torch.set_rng_state(ck["torch_rng"])
    rng.bit_generator.state = ck["numpy_rng"]
    if verbose:
        print(f"  RESUMED from epoch {ck['epoch']} (step {ck['step']:,}), "
              f"best AP so far {ck['best'].get('macro_ap', -1):.4f}", flush=True)
    return ck["epoch"], ck["step"], ck["best"]


def train(arm: str = "P4", root: str | Path = ".", cfg: TrainConfig | None = None,
          ex_cfg: ExampleConfig | None = None, seq_cfg: SeqConfig | None = None,
          max_steps: int | None = None, verbose: bool = True) -> dict:
    cfg = cfg or TrainConfig()
    seq_cfg = seq_cfg or SeqConfig()
    spec = ARMS[arm]
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(cfg.threads)

    # D17: the split must exist before the vocabulary, or the dev patients help
    # decide which codes clear `min_patients_per_code` and where the decile
    # edges fall -- in the vocabulary they are then scored against. Measured on
    # this cohort: 36 of 1,105 tokens are admitted by dev patients alone.
    dev_pids, hold_pids = set(), set()
    if cfg.dev_frac > 0 or cfg.holdout_frac > 0:
        dev_pids, hold_pids = inner_split_pids(root, cfg.dev_frac, cfg.holdout_frac)
    fit_pids = None
    if dev_pids or hold_pids:
        coh = load_cohort(root)
        all_train = set(coh.loc[coh.split == "train", "pid"].tolist())
        fit_pids = all_train - dev_pids - hold_pids

    vocab = build_vocab(root, seq_cfg, fit_pids=fit_pids)
    pack = build_sequences(root, vocab, seq_cfg, ex_cfg)
    lab = load_labels(root, ex_cfg)
    y = lab["y"].astype(np.float32)
    ar = at_risk_mask(lab).astype(np.float32)
    assert (lab["eid"] == pack.eid).all(), "labels and sequences disagree on order"

    # Training rows EXCLUDE the locked test set. `pack.split == "train"` is
    # 2,791 patients and includes the locked 358.
    from .cv import trainable_pids
    _pool = trainable_pids(str(root))
    tr = np.flatnonzero((pack.split == "train")
                        & np.isin(pack.pid, list(_pool)))
    va = np.flatnonzero(pack.split == "val")
    assert pack.is_real[va].all(), "validation must never be augmented"

    hold = np.array([], int)
    if cfg.dev_frac > 0 or cfg.holdout_frac > 0:
        # Grouped by PATIENT, from ONE fixed permutation, so every config
        # ever compared sees identical splits. A patient's augmented cutoffs
        # must not straddle a boundary or the score is partly memorisation.
        # dev_pids / hold_pids were computed above, before the vocabulary.
        assert set(np.unique(pack.pid[tr])) == (
            set(np.unique(pack.pid[tr])) | dev_pids | hold_pids),             "split pids are not a subset of the pack's train pids"
        in_dev = np.isin(pack.pid[tr], list(dev_pids))
        in_hold = np.isin(pack.pid[tr], list(hold_pids))
        dev, hold, tr = tr[in_dev], tr[in_hold], tr[~(in_dev | in_hold)]
        for a, b in ((tr, dev), (tr, hold), (dev, hold)):
            assert not (set(pack.pid[a]) & set(pack.pid[b])), "split overlap"

        target = cfg.eval_on
        if target == "auto":
            target = "holdout" if len(hold) else ("dev" if len(dev) else "val")
        va = {"holdout": hold, "dev": dev, "val": va}[target]
        if verbose:
            print(f"  fit {len(tr):,} | dev {len(dev):,} | our-test {len(hold):,} "
                  f"rows -> reporting '{target}'. Provided val NOT read.",
                  flush=True)

    feats = None
    if cfg.fusion != "none":
        from .data.features import build_features
        # NOT `F` -- that is torch.nn.functional in this module, and shadowing
        # it makes the loss line fail with an attribute error from a data class.
        # D18: this call used to pass no fit mask, so all nine statistics in
        # features.py were fitted on the full train split -- including the dev
        # patients being scored. Dormant only while fusion == "none", and stage
        # C of the search switched it on. Measured: the inner-train matrix is
        # 3,266 columns against 3,320, so 54 columns existed only because dev
        # patients contributed them.
        #
        # `fit_rows` is aligned to the pack ordering, which the assert below
        # confirms build_features shares.
        fit_rows = None
        if dev_pids or hold_pids:
            fit_rows = (pack.split == "train") & ~np.isin(
                pack.pid, list(dev_pids | hold_pids))
        fmat = build_features(root, ex_cfg=ex_cfg, fit_mask=fit_rows)
        assert (fmat.eid == pack.eid).all(), "features and sequences disagree on order"
        X = np.log1p(np.clip(np.nan_to_num(fmat.X, nan=0.0), 0, None))
        scale = np.abs(X[tr]).max(0)                 # train-only, like the LR path
        feats = (X / np.where(scale > 0, scale, 1.0)).astype(np.float32)

    model = PatientTransformer(GPTConfig(
        vocab_size=len(vocab), block_size=seq_cfg.block_size,
        n_outputs=y.shape[1], causal=spec["causal"], readout=spec["readout"],
        n_layer=cfg.n_layer, n_embd=cfg.n_embd, n_head=cfg.n_head,
        attn_dropout=cfg.attn_dropout, resid_dropout=cfg.resid_dropout,
        use_time_encoding=cfg.use_time_encoding, use_dt_bias=cfg.use_dt_bias,
        fusion=cfg.fusion, fusion_dim=cfg.fusion_dim,
        n_features=(feats.shape[1] if feats is not None else 0)))

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95))

    rng = np.random.default_rng(cfg.seed)
    steps_per_epoch = math.ceil(len(tr) / cfg.batch_size)
    total = steps_per_epoch * cfg.epochs if max_steps is None else max_steps
    warm = max(1, int(cfg.warmup_frac * total))

    if spec["pretrain"] is not None:
        # A smoke run must exercise the pretraining path, not sit in it: with
        # `max_steps` set we are checking the code runs, not training anything.
        pretrain(model, pack, tr, spec["pretrain"], cfg, vocab,
                 epochs=cfg.pretrain_epochs, verbose=verbose,
                 max_steps=(5 if max_steps is not None else None))

    # The run id must distinguish CONFIGURATIONS, not just arm and seed. All
    # five learning-rate points of the basin sweep run as arm=P4, seed=0, so
    # under the old name they logged as one run and 220 epochs of five
    # different learning rates appeared as a single trajectory -- which makes
    # any per-run variance or convergence analysis meaningless. The fingerprint
    # already covers architecture, schedule, seed, arm and vocabulary.
    fingerprint = _config_fingerprint(cfg, arm, len(vocab))

    run = wandb_shim.init(f"{arm}_seed{cfg.seed}_{fingerprint[:8]}",
                          {"arm": arm, **spec, **asdict(cfg),
                           "fingerprint": fingerprint,
                           "n_train": len(tr), "vocab": len(vocab)})

    Y = torch.from_numpy(y)
    A = torch.from_numpy(ar)

    # EMA of the weights, updated once per evaluated epoch. float() copies so
    # the average is never a view onto live parameters.
    ema = WeightEMA(cfg.ema_decay)

    ckpt = _ckpt_path(root, arm, cfg.seed, fingerprint)
    start_epoch, step, best = 0, 0, {"macro_ap": -1.0}
    if cfg.resume and max_steps is None:
        start_epoch, step, best = _try_resume(
            ckpt, model=model, opt=opt, rng=rng, fingerprint=fingerprint,
            verbose=verbose)
    t0 = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        tot, nb = 0.0, 0
        # Trajectory telemetry, accumulated over the epoch. `clip_grad_norm_`
        # already computes the total norm and we were discarding its return
        # value, so the single most diagnostic number in the loop was being
        # thrown away every step.
        gn_sum, gn_max, n_clipped = 0.0, 0.0, 0
        upd_norm = float("nan")
        for rows in bucketed_batches(pack.lengths, tr, cfg.batch_size, rng):
            lr = cfg.lr * (step / warm if step < warm else
                           0.5 * (1 + math.cos(math.pi * (step - warm) /
                                               max(1, total - warm))))
            for g in opt.param_groups:
                g["lr"] = lr

            t, d, L = _trim(pack.tokens, pack.dt, pack.lengths, rows)
            fb = torch.from_numpy(feats[rows]) if feats is not None else None
            logits = model(t, d, L, features=fb)
            r = torch.from_numpy(rows.astype(np.int64))
            # Prevalent pairs contribute no gradient, but the row still trains
            # the trunk through its other 39 labels.
            loss = (F.binary_cross_entropy_with_logits(
                logits, Y[r], reduction="none") * A[r]).sum() / A[r].sum().clamp(min=1)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            # The RETURN value is the pre-clip total norm. Keeping it answers a
            # question the config cannot: is grad_clip=1.0 a live constraint or
            # dead configuration? If the norm is routinely above the threshold
            # we are rescaling most updates, which silently changes the
            # effective learning rate and makes the lr factor mean something
            # other than what the design thinks it means.
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                         cfg.grad_clip))
            gn_sum += gnorm
            gn_max = max(gn_max, gnorm)
            n_clipped += int(gnorm > cfg.grad_clip)
            # Once per epoch, measure the REAL update norm by snapshotting around
            # one step. lr*||g||/||w|| is the SGD update and is wrong for Adam,
            # whose step is lr*m/(sqrt(v)+eps) -- about lr per COORDINATE, so
            # ||dw|| ~ lr*sqrt(n), not lr*||g||. The SGD form understated the
            # ratio by ~6,000x and made a healthy 7.5e-3 look like 1.2e-6.
            _snap = ([q.detach().clone() for q in model.parameters()]
                     if nb == 0 else None)
            opt.step()
            if _snap is not None:
                upd_norm = float(sum(float((q.detach() - o).norm()) ** 2
                                     for q, o in zip(model.parameters(), _snap)) ** 0.5)
            tot += float(loss); nb += 1; step += 1

            if step == 20 and verbose:
                rate = (time.time() - t0) / step
                print(f"  projected finish: {rate*total/60:.0f} min "
                      f"({rate*1000:.0f} ms/step, {total:,} steps)", flush=True)
            if max_steps is not None and step >= max_steps:
                break

        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            P = predict(model, pack.tokens, pack.dt, pack.lengths, va,
                        features=feats)
            Pm = np.where(ar[va].astype(bool), P, 1e-6)
            au, _ = macro_auroc(y[va].astype(int), Pm)
            apv, _ = macro_ap(y[va].astype(int), Pm)
            # Score the averaged weights on the SAME eval set, then restore.
            # One extra forward pass per epoch; selection is untouched.
            ema_au = ema_ap = float("nan")
            if cfg.ema_decay > 0:
                live = {k: v.detach().clone() for k, v in model.state_dict().items()}
                try:
                    avg = ema.update(model.state_dict())
                    model.load_state_dict({k: v.to(live[k].dtype)
                                           for k, v in avg.items()})
                    Pe = predict(model, pack.tokens, pack.dt, pack.lengths, va,
                                 features=feats)
                    Pem = np.where(ar[va].astype(bool), Pe, 1e-6)
                    ema_au = macro_auroc(y[va].astype(int), Pem)[0]
                    ema_ap = macro_ap(y[va].astype(int), Pem)[0]
                finally:
                    model.load_state_dict(live)

            if verbose:
                print(f"  epoch {epoch+1:>3}  loss {tot/max(nb,1):.4f}  "
                      f"AUROC {au:.4f}  AP {apv:.4f}  "
                      f"| EMA AUROC {ema_au:.4f} AP {ema_ap:.4f}  "
                      f"[{(time.time()-t0)/60:.0f} min]", flush=True)
            # Parameter norm: with weight decay the weights settle toward a
            # scale, and the update/param ratio is the standard check that the
            # step size is sane (~1e-3 is the usual healthy band). Neither is
            # recoverable after the fact, so both are logged per epoch.
            pnorm = float(sum(float(q.detach().norm()) ** 2
                              for q in model.parameters()) ** 0.5)
            run.log({"loss": tot / max(nb, 1), "macro_auroc": au,
                     "macro_ap": apv, "lr": lr,
                     "epoch": epoch + 1,
                     "grad_norm_mean": gn_sum / max(nb, 1),
                     "grad_norm_max": gn_max,
                     "clip_frac": n_clipped / max(nb, 1),
                     "param_norm": pnorm,
                     "update_norm": upd_norm,
                     "update_param_ratio": upd_norm / max(pnorm, 1e-12),
                     "ema_macro_auroc": ema_au, "ema_macro_ap": ema_ap,
                     "minutes": (time.time() - t0) / 60}, step=step)
            # A band, not a bare ">": without it a 1e-6 wobble resets
            # patience and the checkpoint chases evaluation noise.
            improved = apv > best["macro_ap"] + cfg.min_delta
            if improved:
                best = {"epoch": epoch + 1, "macro_auroc": au, "macro_ap": apv,
                        "preds": P.copy()}
                # The reported score is a MAX over epochs taken on the very set
                # that selects the epoch, so it carries a winner's curse of the
                # same shape as the one G1 measures on validation. It is not
                # common-mode either: a config with a noisier training curve
                # (high dropout, high lr -- both design factors) gets a larger
                # max-selection boost, so the bias lands on the contrasts.
                #
                # When a holdout exists and is NOT the selection set, score it
                # at the selected epoch. It informed neither the fit nor the
                # epoch choice, so `holdout_macro_ap` is clean and the gap to
                # `macro_ap` is the epoch-selection inflation, MEASURED rather
                # than argued about.
                if len(hold) and not np.array_equal(hold, va):
                    Ph = predict(model, pack.tokens, pack.dt, pack.lengths,
                                 hold, features=feats)
                    Phm = np.where(ar[hold].astype(bool), Ph, 1e-6)
                    best["holdout_macro_auroc"] = macro_auroc(
                        y[hold].astype(int), Phm)[0]
                    best["holdout_macro_ap"] = macro_ap(
                        y[hold].astype(int), Phm)[0]
                # Checkpoint at the best epoch, not the last. Without this the
                # reported score belongs to a model that no longer exists by
                # the end of training, and nothing can be published or
                # re-scored without a full retrain.
                Path("artifacts").mkdir(exist_ok=True)
                if max_steps is None:
                    torch.save({"state_dict": model.state_dict(),
                                "gpt_config": asdict(model.cfg),
                                "arm": arm, "spec": spec, "seed": cfg.seed,
                                "epoch": epoch + 1, "vocab_size": len(vocab),
                                "val_macro_auroc": au, "val_macro_ap": apv},
                               f"artifacts/model_{arm}_seed{cfg.seed}.pt")

            # Resume state EVERY eval epoch, improved or not -- an interrupted
            # run must restart from where it stopped, not from its best epoch.
            if max_steps is None:
                _save_resume(ckpt, model=model, opt=opt, epoch=epoch + 1,
                             step=step, best=best, rng=rng,
                             fingerprint=fingerprint)
        if max_steps is not None and step >= max_steps:
            break
        if cfg.patience > 0 and (epoch + 1) - best.get("epoch", 0) >= cfg.patience:
            if verbose:
                print(f"  early stop at epoch {epoch+1}: no improvement for "
                      f"{cfg.patience} epochs (best was {best['epoch']})",
                      flush=True)
            break

    run.summary(best_macro_ap=best["macro_ap"], best_macro_auroc=best["macro_auroc"])
    run.finish()
    # Clean completion: drop the resume state so it cannot be mistaken for a
    # crashed run later, and so 20 design runs do not leave 20 x ~25 MB behind.
    if max_steps is None:
        ckpt.unlink(missing_ok=True)

    return {"arm": arm, "seed": cfg.seed, **{k: v for k, v in best.items()
                                             if k != "preds"},
            "preds": best.get("preds"), "epochs_run": epoch + 1,
            "minutes": (time.time() - t0) / 60}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="P4", choices=list(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    ex_cfg = (ExampleConfig() if args.stride == 0
              else ExampleConfig(augment=True, stride_years=args.stride))
    seq_cfg = SeqConfig(block_size=args.block)
    out = []
    for s in args.seeds:
        cfg = TrainConfig(seed=s, epochs=2 if args.smoke else args.epochs)
        print(f"\n=== {args.arm} seed {s} ===", flush=True)
        out.append(train(args.arm, ".", cfg, ex_cfg, seq_cfg,
                         max_steps=30 if args.smoke else None))
    # A smoke run trains for 30 steps. Persisting its predictions would let
    # `compare.py` rank a 30-step model against a real one and, worse, would
    # look exactly like a real result on disk.
    if args.smoke:
        print("smoke run -- predictions not persisted")
    else:
        Path("artifacts").mkdir(exist_ok=True)
        for r in out:
            if r.get("preds") is not None:
                np.save(f"artifacts/val_preds_{args.arm}_seed{r['seed']}.npy",
                        r["preds"])
    aps = [r["macro_ap"] for r in out]
    aus = [r["macro_auroc"] for r in out]
    print(f"\n{args.arm}: AUROC {np.mean(aus):.4f} +- {np.std(aus):.4f}   "
          f"AP {np.mean(aps):.4f} +- {np.std(aps):.4f}   over {len(out)} seeds")


if __name__ == "__main__":
    main()
