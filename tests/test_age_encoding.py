"""Age-at-event encoding (§E56.3): off is the old model exactly, on is live.

Behavioural, per the project rule: assert that changing a patient's age moves
the output, not that a flag is set.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.gpt import GPTConfig, PatientTransformer

VOCAB = 1105
ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "artifacts" / "model_P4_seed300_90b58acf.pt"


def _batch(B=4, T=20):
    g = torch.Generator().manual_seed(0)
    tokens = torch.randint(3, VOCAB, (B, T), generator=g)
    dt = torch.sort(torch.rand(B, T, generator=g) * 5000, dim=1, descending=True).values
    tokens[:, -1], dt[:, -1] = 2, 0.0                     # [ANCHOR] last, dt = 0
    return tokens, dt, torch.full((B,), T)


def _model(age: bool):
    torch.manual_seed(0)
    m = PatientTransformer(GPTConfig(vocab_size=VOCAB, n_layer=1, n_embd=64,
                                     n_head=2, use_age_encoding=age))
    return m.eval()


def test_age_off_builds_the_old_model():
    """No new parameters, and `mix` keeps its 64 x 128 shape, so every
    existing checkpoint still loads strictly."""
    m = _model(False)
    assert m.age is None
    assert tuple(m.mix.weight.shape) == (64, 128)
    assert not any(k.startswith("age.") for k in m.state_dict())


def test_age_on_widens_mix_and_adds_one_time2vec():
    m = _model(True)
    assert tuple(m.mix.weight.shape) == (64, 192)
    assert sum(p.numel() for n, p in m.named_parameters() if n.startswith("age.")) == 128


def test_age_moves_the_output_only_when_on():
    tokens, dt, L = _batch()
    young, old = torch.full((4,), 30 * 365.25), torch.full((4,), 70 * 365.25)
    with torch.no_grad():
        on = _model(True)
        assert (on(tokens, dt, L, age0=young) - on(tokens, dt, L, age0=old)).abs().max() > 1e-4
        off = _model(False)
        assert torch.equal(off(tokens, dt, L, age0=young), off(tokens, dt, L, age0=old))


def test_age_on_refuses_a_missing_age():
    tokens, dt, L = _batch()
    with pytest.raises(AssertionError, match="age0"):
        _model(True)(tokens, dt, L)


def test_age_is_per_event_not_per_patient():
    """Shifting the SAME patient's events in time, with the anchor fixed, must
    change the age of each event and so the output -- the encoding is of age
    at the event, not a constant stamped on the whole sequence."""
    tokens, dt, L = _batch()
    a0 = torch.full((4,), 50 * 365.25)
    m = _model(True)
    m.cfg.use_time_encoding = False          # isolate the age path from dt's own encoding
    with torch.no_grad():
        m.time.w0.zero_(); m.time.a0.zero_()
        base = m(tokens, dt, L, age0=a0)
        dt2 = dt.clone(); dt2[:, :-1] = dt2[:, :-1] * 0.5
        assert (base - m(tokens, dt2, L, age0=a0)).abs().max() > 1e-4


@pytest.mark.skipif(not SHIPPED.exists(), reason="shipped checkpoint not present")
def test_shipped_checkpoint_still_loads_strictly():
    d = torch.load(SHIPPED, map_location="cpu", weights_only=False)
    m = PatientTransformer(GPTConfig(**d["gpt_config"]))
    m.load_state_dict(d["state_dict"], strict=True)
    assert m.age is None
