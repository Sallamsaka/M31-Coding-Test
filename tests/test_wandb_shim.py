"""The tracking layer must never be able to take a run down with it.

This is not hypothetical. A 40-minute gradient-boosting fit was lost to
`ConnectionAbortedError` raised from wandb's own service teardown -- after the
work was finished, and from an atexit handler no wrapper can catch. Hence:
the backend is opt-in, and everything is written locally regardless.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils import wandb_shim


@pytest.fixture
def local(tmp_path):
    return tmp_path / "metrics.jsonl"


def test_backend_is_off_unless_asked_for(monkeypatch, local):
    monkeypatch.delenv("WANDB", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    run = wandb_shim.Run("t", {}, backend=None, local=local)
    assert run.offline


def test_metrics_are_written_locally_with_no_backend(local):
    run = wandb_shim.Run("t", {"a": 1}, backend=None, local=local)
    run.log({"macro_auroc": 0.77}, step=3)
    run.summary(selected="lr")
    run.finish()

    rows = [json.loads(l) for l in local.read_text().splitlines()]
    kinds = [r.get("event") for r in rows]
    assert "init" in kinds and "summary" in kinds and "finish" in kinds
    scored = [r for r in rows if "macro_auroc" in r]
    assert scored and scored[0]["macro_auroc"] == 0.77 and scored[0]["step"] == 3


def test_a_throwing_backend_cannot_break_the_run(local):
    """The local record must survive, and the backend must be dropped rather
    than retried on every subsequent call."""
    class Exploding:
        def __init__(self):
            self.calls = 0

        def log(self, metrics, step=None):
            self.calls += 1
            raise ConnectionResetError("connection lost")

        def finish(self):
            raise RuntimeError("teardown exploded")

    boom = Exploding()
    run = wandb_shim.Run("t", {}, backend=boom, local=local)
    run.log({"macro_auroc": 0.5})
    run.log({"macro_auroc": 0.6})          # backend already disabled
    run.finish()                            # must not raise

    assert boom.calls == 1, "backend was retried after failing"
    rows = [json.loads(l) for l in local.read_text().splitlines()]
    assert sum("macro_auroc" in r for r in rows) == 2, "local record lost a metric"
    assert any(r.get("event") == "wandb_log_failed" for r in rows), \
        "the failure was swallowed without a trace"


def test_unwritable_summary_values_do_not_raise(local):
    """`default=str` in the JSON dump: a config holding a numpy dtype or a
    Path should not be able to kill a training run at the logging call."""
    import numpy as np
    run = wandb_shim.Run("t", {"p": Path("x"), "d": np.float32}, backend=None,
                         local=local)
    run.log({"macro_auroc": np.float32(0.5), "arr": np.arange(3)})
    run.finish()
    assert local.exists() and local.read_text().strip()
