"""Resuming a killed run must restore the OPTIMISER and the RNG, not just weights.

A checkpoint holding only `state_dict` looks like it works -- the model loads,
training continues, nothing errors -- and is silently wrong: Adam's moments
restart from zero, the cosine schedule restarts from step 0, and the batch order
repeats. The run finishes and is not the run you think it is, which is this
project's most common bug shape.

These tests are on the save/load helpers with a tiny model, so they cost
milliseconds rather than a training run.
"""

from __future__ import annotations

import torch

from src.models.gpt import GPTConfig, PatientTransformer
from src.train_finetune import (TrainConfig, _ckpt_path, _config_fingerprint,
                                _save_resume, _try_resume)
import numpy as np


def _tiny():
    torch.manual_seed(0)
    m = PatientTransformer(GPTConfig(vocab_size=40, n_layer=1, n_embd=32,
                                     n_head=2, d_time=8, block_size=16))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    return m, opt


def _step(m, opt, n=3):
    """Take real steps so the optimiser accumulates non-trivial moments."""
    for _ in range(n):
        tok = torch.randint(1, 40, (2, 8))
        dt = torch.arange(8, 0, -1).float().repeat(2, 1)
        lengths = torch.full((2,), 8)
        loss = m(tok, dt, lengths).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()


def test_resume_restores_optimiser_moments_not_just_weights(tmp_path):
    m, opt = _tiny()
    _step(m, opt)
    rng = np.random.default_rng(7)
    rng.integers(0, 10, 5)                      # advance it
    fp = "fingerprint-A"
    path = tmp_path / "ck.pt"
    _save_resume(path, model=m, opt=opt, epoch=4, step=123, best={"macro_ap": 0.2},
                 rng=rng, fingerprint=fp)

    m2, opt2 = _tiny()
    # A fresh optimiser has NO state at all -- this is what a weights-only
    # checkpoint would silently leave us with.
    assert len(opt2.state) == 0
    ep, st, best = _try_resume(path, model=m2, opt=opt2, rng=np.random.default_rng(0),
                               fingerprint=fp, verbose=False)
    assert (ep, st) == (4, 123)
    assert best["macro_ap"] == 0.2
    assert len(opt2.state) == len(opt.state) > 0, "optimiser moments not restored"

    ref = opt.state_dict()["state"]
    got = opt2.state_dict()["state"]
    for k in ref:
        assert torch.allclose(ref[k]["exp_avg"], got[k]["exp_avg"])
        assert torch.allclose(ref[k]["exp_avg_sq"], got[k]["exp_avg_sq"])


def test_resume_restores_the_numpy_generator_stream(tmp_path):
    """Batch order comes from this generator; a reset stream repeats epochs."""
    m, opt = _tiny()
    rng = np.random.default_rng(11)
    rng.integers(0, 100, 17)
    expected = rng.integers(0, 100, 5)

    rng2 = np.random.default_rng(11)
    rng2.integers(0, 100, 17)
    path = tmp_path / "ck.pt"
    _save_resume(path, model=m, opt=opt, epoch=1, step=1, best={}, rng=rng2,
                 fingerprint="f")

    target = np.random.default_rng(999)         # wrong stream on purpose
    _try_resume(path, model=m, opt=opt, rng=target, fingerprint="f", verbose=False)
    assert np.array_equal(target.integers(0, 100, 5), expected)


def test_a_checkpoint_from_a_different_config_is_refused(tmp_path):
    """Loading another architecture's tensors is the failure this guards."""
    m, opt = _tiny()
    path = tmp_path / "ck.pt"
    _save_resume(path, model=m, opt=opt, epoch=9, step=99, best={"macro_ap": 0.5},
                 rng=np.random.default_rng(0), fingerprint="config-A")

    m2, opt2 = _tiny()
    ep, st, best = _try_resume(path, model=m2, opt=opt2,
                               rng=np.random.default_rng(0),
                               fingerprint="config-B", verbose=False)
    assert (ep, st) == (0, 0) and best["macro_ap"] == -1.0
    assert len(opt2.state) == 0, "refused resume must not touch the optimiser"


def test_a_truncated_checkpoint_starts_fresh_instead_of_crashing(tmp_path):
    """The checkpoint exists for runs that died -- including ones that died
    mid-write. An unreadable file must not take the retry down with it."""
    path = tmp_path / "ck.pt"
    path.write_bytes(b"not a torch file")
    m, opt = _tiny()
    ep, st, best = _try_resume(path, model=m, opt=opt,
                               rng=np.random.default_rng(0),
                               fingerprint="f", verbose=False)
    assert (ep, st) == (0, 0) and best["macro_ap"] == -1.0


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path):
    m, opt = _tiny()
    path = tmp_path / "ck.pt"
    _save_resume(path, model=m, opt=opt, epoch=1, step=1, best={},
                 rng=np.random.default_rng(0), fingerprint="f")
    assert path.exists()
    assert not path.with_suffix(".tmp").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_missing_checkpoint_is_a_fresh_start(tmp_path):
    m, opt = _tiny()
    ep, st, best = _try_resume(tmp_path / "nope.pt", model=m, opt=opt,
                               rng=np.random.default_rng(0),
                               fingerprint="f", verbose=False)
    assert (ep, st) == (0, 0) and best["macro_ap"] == -1.0


def test_fingerprint_separates_the_things_a_resume_must_not_cross():
    BASE = dict(seed=0, n_layer=2, lr=6e-4)
    base = TrainConfig(**BASE)
    fp = _config_fingerprint(base, "P4", 1105)
    assert fp == _config_fingerprint(TrainConfig(**BASE), "P4", 1105)
    for kw in ({"n_layer": 4}, {"lr": 1e-3}, {"seed": 1}, {"epochs": 99},
               {"use_dt_bias": False}, {"fusion": "readout"}):
        assert _config_fingerprint(TrainConfig(**{**BASE, **kw}),
                                   "P4", 1105) != fp, kw
    assert _config_fingerprint(base, "P3", 1105) != fp          # arm
    assert _config_fingerprint(base, "P4", 900) != fp           # vocabulary


def test_min_delta_default_preserves_the_original_behaviour():
    assert TrainConfig().min_delta == 0.0


def test_ckpt_path_is_keyed_by_arm_and_seed():
    a = _ckpt_path(".", "P4", 0)
    assert a != _ckpt_path(".", "P3", 0) and a != _ckpt_path(".", "P4", 1)


# --------------------------------------------------------------------------
# WeightEMA. These exist because the first version was a closure inside
# train(), therefore untestable, and shipped a bias-correction bug that
# returned 5x the weights at the first epoch.
# --------------------------------------------------------------------------

def test_ema_of_a_constant_is_that_constant():
    """The invariant that caught the bug. Averaging an unchanging quantity must
    return it unchanged at EVERY step, including the first."""
    from src.train_finetune import WeightEMA
    e = WeightEMA(0.8)
    w = {"a": torch.tensor([1.0, -2.0])}
    for _ in range(6):
        out = e.update(w)
        assert torch.allclose(out["a"], w["a"]), out["a"]


def test_ema_step_response_matches_the_closed_form():
    """After a 0 -> 1 step the average must approach 1 as 1 - d^t exactly."""
    from src.train_finetune import WeightEMA
    d = 0.8
    e = WeightEMA(d)
    e.update({"a": torch.tensor([0.0])})
    for t in range(1, 6):
        got = e.update({"a": torch.tensor([1.0])})["a"].item()
        assert abs(got - (1 - d ** t)) < 1e-6, (t, got, 1 - d ** t)


def test_ema_is_not_a_view_onto_the_live_weights():
    """If the average aliased the parameters it would track them exactly and
    the whole point -- damping the bounce -- would silently not happen."""
    from src.train_finetune import WeightEMA
    e = WeightEMA(0.5)
    w = {"a": torch.tensor([0.0])}
    e.update(w)
    w["a"].add_(100.0)                      # mutate the "live" tensor in place
    assert e.state["a"].item() == 0.0, "EMA aliased the source tensor"


def test_ema_copies_non_float_buffers_instead_of_averaging_them():
    """Averaging an integer counter is meaningless and would corrupt a buffer."""
    from src.train_finetune import WeightEMA
    e = WeightEMA(0.8)
    e.update({"n": torch.tensor([3], dtype=torch.long)})
    out = e.update({"n": torch.tensor([9], dtype=torch.long)})
    assert out["n"].item() == 9 and out["n"].dtype == torch.long


def test_ema_decay_controls_the_averaging_window():
    """Higher decay = longer memory = slower response to a change."""
    from src.train_finetune import WeightEMA
    resp = {}
    for d in (0.5, 0.9):
        e = WeightEMA(d)
        e.update({"a": torch.tensor([0.0])})
        for _ in range(3):
            v = e.update({"a": torch.tensor([1.0])})["a"].item()
        resp[d] = v
    assert resp[0.5] > resp[0.9], resp
