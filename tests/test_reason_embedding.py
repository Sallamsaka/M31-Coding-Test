"""Reason embedding (sweep arm B3): off is the old model exactly, on is live.

Behavioural: assert that changing an event's reason moves the output, and that
"no reason" contributes nothing -- not that a flag is set.
"""
from __future__ import annotations

import pytest
import torch

from src.models.gpt import GPTConfig, PatientTransformer

VOCAB, N_REASONS = 1105, 110


def _batch(B=4, T=20):
    g = torch.Generator().manual_seed(0)
    tokens = torch.randint(3, VOCAB, (B, T), generator=g)
    dt = torch.sort(torch.rand(B, T, generator=g) * 5000, dim=1, descending=True).values
    tokens[:, -1], dt[:, -1] = 2, 0.0
    reasons = torch.zeros(B, T, dtype=torch.long)
    reasons[:, 3:8] = torch.randint(2, N_REASONS, (B, 5), generator=g)
    return tokens, dt, torch.full((B,), T), reasons


def _model(on: bool):
    torch.manual_seed(0)
    return PatientTransformer(GPTConfig(
        vocab_size=VOCAB, n_layer=1, n_embd=64, n_head=2,
        use_reason_embedding=on, n_reasons=N_REASONS if on else 0)).eval()


def test_reason_off_builds_the_old_model():
    m = _model(False)
    assert m.reason is None
    assert tuple(m.mix.weight.shape) == (64, 128)
    assert not any(k.startswith("reason.") for k in m.state_dict())


def test_no_reason_row_is_zero_and_widens_mix():
    m = _model(True)
    assert tuple(m.mix.weight.shape) == (64, 192)
    assert torch.count_nonzero(m.reason.weight[0]) == 0


def test_reason_moves_the_output_only_when_on():
    tokens, dt, L, reasons = _batch()
    other = reasons.clone()
    other[:, 3:8] = (other[:, 3:8] + 1) % N_REASONS
    other[:, 3:8] = other[:, 3:8].clamp(min=2)
    with torch.no_grad():
        on = _model(True)
        assert (on(tokens, dt, L, reasons=reasons)
                - on(tokens, dt, L, reasons=other)).abs().max() > 1e-4
        off = _model(False)
        assert torch.equal(off(tokens, dt, L, reasons=reasons),
                           off(tokens, dt, L, reasons=other))


def test_all_no_reason_equals_a_zero_reason_input():
    """With every position at row 0, the reason channel is exactly zero, so
    the output equals the same model with that slice of `mix` zeroed out."""
    tokens, dt, L, _ = _batch()
    m = _model(True)
    with torch.no_grad():
        a = m(tokens, dt, L, reasons=torch.zeros_like(tokens))
        m.mix.weight[:, 128:] = 0.0
        b = m(tokens, dt, L, reasons=torch.randint(2, N_REASONS, tokens.shape))
    assert torch.allclose(a, b, atol=1e-6)


def test_reason_on_refuses_missing_reasons():
    tokens, dt, L, _ = _batch()
    with pytest.raises(AssertionError, match="reasons"):
        _model(True)(tokens, dt, L)
