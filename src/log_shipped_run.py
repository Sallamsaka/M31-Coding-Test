"""Re-fit the SHIPPED transformer with wandb on, for the report's curves.

Run:  ``$env:WANDB=1; python -u -m src.log_shipped_run``   (~15 min, CPU)

The brief requires wandb training and validation curves, and every existing
wandb run is an offline experiment from before the shipped model existed. This
refits exactly the recipe behind ``model_transformer.joblib`` -- it calls
``cross_validate._fit_transformer``, the same function ``run_baseline`` calls,
rather than a copy of it -- so the logged run *is* the submitted model, and
then proves it: the refit's prediction matrix is compared against the stored
one, and the verdict is printed.

The refit writes the same checkpoint paths the shipped fit did. Those are
backed up first and restored afterwards, so the weights on disk (and on the
Hub) stay the ones that produced predictions.csv even if the refit differs in
the last bit.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import joblib
import numpy as np


def main() -> None:
    from .utils.wandb_shim import _has_netrc

    if os.environ.get("WANDB", "").lower() not in {"1", "true", "yes"}:
        raise SystemExit("set WANDB=1 first -- this run exists only to be logged")
    if not (os.environ.get("WANDB_API_KEY") or _has_netrc()):
        # Offline mode would "succeed" and produce another offline-run-* with
        # no URL, which is the exact state this module exists to fix.
        raise SystemExit("no wandb credential: run `wandb login` first")
    os.environ.setdefault("WANDB_RUN_GROUP", "shipped-transformer")

    from .cross_validate import _fit_transformer
    from .data.examples import ExampleConfig
    from .data.features import build_features
    from .data.labels import at_risk_mask, load_labels
    from .hf_upload import _shipped_transformer_ckpts
    from .train_baseline import _fit_rows, apply_at_risk_mask

    root = Path(".")
    stored = joblib.load("artifacts/model_transformer.joblib")["preds"]
    ckpts = _shipped_transformer_ckpts(root)
    backup = root / "artifacts" / "shipped_backup"
    backup.mkdir(exist_ok=True)
    for p in ckpts:
        shutil.copy2(p, backup / p.name)
    print(f"backed up {len(ckpts)} shipped checkpoints to {backup}", flush=True)

    try:
        ex_cfg = ExampleConfig()
        F = build_features(root, ex_cfg=ex_cfg)
        lab = load_labels(root, ex_cfg)
        t = time.time()
        P = apply_at_risk_mask(
            _fit_transformer(root, _fit_rows(F, None), ex_cfg,
                             lab["y"].shape, verbose=True),
            at_risk_mask(lab))
        print(f"refit {time.time() - t:.0f}s", flush=True)
    finally:
        for p in ckpts:
            shutil.copy2(backup / p.name, p)
        print("restored the shipped checkpoints", flush=True)

    d = np.abs(P.astype(np.float64) - stored.astype(np.float64))
    print(f"refit vs shipped prediction matrix: max |diff| {d.max():.2e}, "
          f"mean {d.mean():.2e}")
    print("VERDICT:", "the logged run reproduces the shipped model"
          if d.max() < 1e-4 else
          "the refit DIFFERS from the shipped model -- the curves describe "
          "the recipe, not these exact weights; say so in the report")


if __name__ == "__main__":
    main()
