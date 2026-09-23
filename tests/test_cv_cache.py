"""The CV cache key must change when anything that changes the cached value does.

D34 is the bug these guard: the per-fold cache key captured fold membership and
the model list but **not the model configs**, so editing `TRANSFORMER_CFG` or
`GBDTConfig` would have reloaded the OLD predictions under an unchanged key and
reported them as the new configuration's -- a run that succeeds, passes every
test, and answers a different question than the one asked.

D34 recorded, as a known residual, that `_model_sig` lived inside `run_cv` as a
closure and therefore could not be tested. It is module level now, so it can be.

These are behavioural: they assert the signature MOVES when a config moves,
not that some field is present in some string.
"""
import json
import os

import pytest

from src.cross_validate import MODELS, _gbdt_cfg, _model_sig
from src.train_baseline import GBDTConfig


def test_every_model_has_a_distinct_signature():
    """Two models sharing a signature would share a cache file."""
    sigs = {m: _model_sig(m) for m in MODELS}
    tuned = [m for m in MODELS if m != "prevalence"]
    assert len({sigs[m] for m in tuned}) == len(tuned), (
        f"signature collision between models: {sigs}")


def test_gbdt_signature_moves_when_the_gbdt_config_moves(monkeypatch):
    """The exact D34 scenario, as a behaviour."""
    monkeypatch.delenv("CV_GBDT", raising=False)
    before = _model_sig("gbdt")

    monkeypatch.setenv("CV_GBDT", json.dumps({"min_samples_leaf": 25}))
    after = _model_sig("gbdt")

    assert before != after, (
        "changing min_samples_leaf left the gbdt cache key unchanged -- an "
        "overridden run would reload the default config's fits and report them "
        "as the override's")


def test_changing_gbdt_does_not_disturb_other_models(monkeypatch):
    """A fold's `lr` predictions do not depend on the GBDT config, so its cache
    must not be invalidated by one -- that was the coupling D34 also removed."""
    monkeypatch.delenv("CV_GBDT", raising=False)
    before = {m: _model_sig(m) for m in MODELS if m != "gbdt"}

    monkeypatch.setenv("CV_GBDT", json.dumps({"max_iter": 77}))
    after = {m: _model_sig(m) for m in MODELS if m != "gbdt"}

    assert before == after, "a GBDT-only change moved another model's cache key"


def test_transformer_signature_moves_with_its_config(monkeypatch):
    import src.cross_validate as cv

    before = _model_sig("transformer")
    monkeypatch.setitem(cv.TRANSFORMER_CFG, "fusion_dim", 256)
    assert _model_sig("transformer") != before, (
        "changing fusion_dim left the transformer cache key unchanged")


def test_transformer_signature_moves_with_its_seeds(monkeypatch):
    """Seed-averaging is part of what produces the matrix, so the seeds are
    part of what identifies it."""
    import src.cross_validate as cv

    before = _model_sig("transformer")
    monkeypatch.setattr(cv, "TRANSFORMER_SEEDS", (300, 301, 302, 303))
    assert _model_sig("transformer") != before


@pytest.mark.parametrize("raw,expected", [("", 5), ("   ", 5)])
def test_absent_override_yields_the_declared_default(monkeypatch, raw, expected):
    """An empty or whitespace env var must not silently become an override --
    otherwise a stray export changes the shipped model."""
    monkeypatch.setenv("CV_GBDT", raw)
    assert _gbdt_cfg().min_samples_leaf == expected
    assert _gbdt_cfg() == GBDTConfig()


def test_override_applies_only_the_named_field(monkeypatch):
    monkeypatch.setenv("CV_GBDT", json.dumps({"min_samples_leaf": 25}))
    cfg, base = _gbdt_cfg(), GBDTConfig()
    assert cfg.min_samples_leaf == 25
    assert cfg.learning_rate == base.learning_rate
    assert cfg.max_iter == base.max_iter
    assert cfg.max_leaf_nodes == base.max_leaf_nodes


def test_signature_is_stable_across_calls():
    """A key that is not reproducible is not a key."""
    assert _model_sig("gbdt") == _model_sig("gbdt")
    assert _model_sig("transformer") == _model_sig("transformer")
