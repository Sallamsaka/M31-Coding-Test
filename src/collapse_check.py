"""Does the transformer actually use its sequence when handed tabular features?

The fusion literature names a failure mode for exactly our regime -- a strong,
fast-converging tabular stream next to a weak, slow-converging sequence stream.
Joint training then learns the easy modality and starves the hard one
("modality laziness"; Wang et al. CVPR 2020; UME, arXiv 2305.01233). In that
state the transformer is a very expensive constant, the fusion numbers are
measuring logistic regression wearing a hat, and every architecture question
downstream is moot.

The diagnostic is an ablation at *inference*, on a model that is already
trained, so it costs one forward pass per condition:

* **intact** -- real sequence, real features. The reference.
* **features zeroed** -- if the score barely moves, the tabular branch is not
  contributing and fusion is decorative.
* **sequence shuffled** -- each patient keeps their own features and labels but
  is handed another patient's event history. **If the score barely moves, the
  transformer has collapsed.** This is the one that matters.

Shuffling rather than zeroing the sequence is deliberate: zeroing produces an
out-of-distribution all-pad input, which a model can detect and which would
confound "unused" with "confused". A permuted real sequence is in-distribution
and carries no information about the patient it is attached to.

Interpretation, with the resolution floor in mind (sigma ~ 0.0104 on 365
patients, wider on a 558-patient dev split):

* shuffle-drop within noise of zero -> collapsed. Stop and fix the training
  (auxiliary sequence-only loss, modality dropout) before tuning anything.
* shuffle-drop large, zero-drop small -> the transformer carries the signal and
  the tabular branch is redundant.
* both large -> genuine fusion; proceed.

**Pre-registered reading, fixed before the result was seen.** This trains for 12
epochs, and the search ledger's runs needed 14-20 to peak. An undertrained model
also shows a small shuffle-drop -- the sequence branch has not learned yet -- and
that is a *different* diagnosis from collapse with the opposite action (train
longer vs. change the training objective). The two are separated by the intact
dev AUROC, which must be read first:

* intact dev AUROC ~0.74-0.76 (the known non-fused P4 range) -> the model is
  trained, and a small shuffle-drop means **collapse**.
* intact dev AUROC materially below that -> **undertrained**, the verdict below
  is void, and the rerun is at more epochs, not a training-objective change.

The VERDICT line printed at the end does not know about this distinction, so it
must not be quoted without the intact AUROC beside it.

Run: ``python -m src.collapse_check``
"""

from __future__ import annotations

import numpy as np
import torch

from .data.examples import ExampleConfig
from .data.labels import at_risk_mask, load_labels
from .data.sequences import SeqConfig, build_sequences, build_vocab
from .evaluate import macro_ap, macro_auroc
from .models.gpt import GPTConfig, PatientTransformer
from .train_finetune import TrainConfig, inner_split_pids, predict, train


def _features(root, ex_cfg, fit_rows, tr_rows, n_rows):
    """Reproduces train()'s feature preprocessing exactly."""
    from .data.features import build_features
    fmat = build_features(root, ex_cfg=ex_cfg, fit_mask=fit_rows)
    X = np.log1p(np.clip(np.nan_to_num(fmat.X, nan=0.0), 0, None))
    scale = np.abs(X[tr_rows]).max(0)
    return (X / np.where(scale > 0, scale, 1.0)).astype(np.float32), fmat


def main(root: str = ".", arm: str = "P4", seed: int = 0) -> None:
    seq_cfg = SeqConfig()
    ex_cfg = ExampleConfig()
    cfg = TrainConfig(seed=seed, dev_frac=0.2, fusion="readout", epochs=12)

    print("training a fused model (fusion=readout, dev_frac=0.2) ...", flush=True)
    res = train(arm, root, cfg, ex_cfg, seq_cfg, verbose=True)
    print(f"  trained: best epoch {res['epoch']}, dev AUROC {res['macro_auroc']:.4f}, "
          f"AP {res['macro_ap']:.4f}, {res['minutes']:.1f} min\n", flush=True)

    ckpt = torch.load(f"{root}/artifacts/model_{arm}_seed{seed}.pt", weights_only=False)
    model = PatientTransformer(GPTConfig(**ckpt["gpt_config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dev_pids, hold_pids = inner_split_pids(root, cfg.dev_frac, cfg.holdout_frac)
    # The vocabulary MUST match the one train() fitted, or the token embedding
    # is the wrong width and the checkpoint will not load. train() narrows it to
    # inner-train when dev_frac > 0 (D17), so this must do the same.
    from .data.cohort import load_cohort
    coh = load_cohort(root)
    all_train = set(coh.loc[coh.split == "train", "pid"].tolist())
    fit_pids = all_train - dev_pids - hold_pids
    vocab = build_vocab(root, seq_cfg, fit_pids=fit_pids)
    assert len(vocab.itos) == ckpt["vocab_size"], (
        f"vocab {len(vocab.itos)} != checkpoint {ckpt['vocab_size']}")
    pack = build_sequences(root, vocab, seq_cfg, ex_cfg)
    lab = load_labels(root, ex_cfg)
    y = lab["y"].astype(int)
    ar = at_risk_mask(lab)

    tr_all = np.flatnonzero(pack.split == "train")
    in_dev = np.isin(pack.pid[tr_all], list(dev_pids))
    dev, tr = tr_all[in_dev], tr_all[~in_dev]
    fit_rows = np.zeros(len(pack.split), bool)
    fit_rows[tr] = True
    feats, _ = _features(root, ex_cfg, fit_rows, tr, len(pack.split))

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(dev))
    conditions = {
        "intact": (pack.tokens, pack.dt, pack.lengths, feats),
        "features zeroed": (pack.tokens, pack.dt, pack.lengths, np.zeros_like(feats)),
    }

    print(f"{'condition':<22}{'macro AUROC':>13}{'drop':>9}{'macro AP':>11}{'drop':>9}")
    base_au = base_ap = None
    out = {}
    for name, (tok, dt, ln, ft) in conditions.items():
        P = predict(model, tok, dt, ln, dev, features=ft)
        Pm = np.where(ar[dev].astype(bool), P[:, :y.shape[1]], 1e-6)
        au = macro_auroc(y[dev], Pm)[0]
        ap = macro_ap(y[dev], Pm)[0]
        if base_au is None:
            base_au, base_ap = au, ap
        out[name] = (au, ap)
        print(f"{name:<22}{au:>13.4f}{base_au-au:>9.4f}{ap:>11.4f}{base_ap-ap:>9.4f}")

    # Sequence shuffled: each dev patient keeps their own features and labels but
    # receives another dev patient's event history.
    shuffled = dev[perm]
    tok2, dt2, ln2 = pack.tokens.copy(), pack.dt.copy(), pack.lengths.copy()
    tok2[dev], dt2[dev], ln2[dev] = pack.tokens[shuffled], pack.dt[shuffled], pack.lengths[shuffled]
    P = predict(model, tok2, dt2, ln2, dev, features=feats)
    Pm = np.where(ar[dev].astype(bool), P[:, :y.shape[1]], 1e-6)
    au, ap = macro_auroc(y[dev], Pm)[0], macro_ap(y[dev], Pm)[0]
    out["sequence shuffled"] = (au, ap)
    print(f"{'sequence shuffled':<22}{au:>13.4f}{base_au-au:>9.4f}{ap:>11.4f}{base_ap-ap:>9.4f}")

    print()
    d_seq = base_au - out["sequence shuffled"][0]
    d_tab = base_au - out["features zeroed"][0]
    sigma = 0.0104 * np.sqrt(365 / len(dev))     # scaled to the dev split size
    print(f"  dev n={len(dev)}, sigma(macro AUROC) ~ {sigma:.4f}")
    print(f"  sequence contributes {d_seq:+.4f} AUROC  ({d_seq/sigma:.1f} sigma)")
    print(f"  tabular  contributes {d_tab:+.4f} AUROC  ({d_tab/sigma:.1f} sigma)")
    verdict = ("COLLAPSED -- the sequence is not being used" if d_seq < 2 * sigma
               else "sequence is carrying signal")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
