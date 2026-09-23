"""Properties of the transformer that must hold before any run is trusted.

Synthetic batches, not the real sequence pack: these assert architectural
invariants, and a 60-second fixture build would buy nothing. The tie structure
is realistic -- 97.7% of real events share a timestamp with another.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.models.gpt import GPTConfig, PatientTransformer

VOCAB = 1105          # the real vocabulary size built by sequences.build_vocab


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = PatientTransformer(GPTConfig(vocab_size=VOCAB))
    m.eval()
    return m


@pytest.fixture(scope="module")
def batch():
    """8 patients, heavy ties, [ANCHOR] at dt=0 in the final position."""
    rng = np.random.default_rng(0)
    B, T = 8, 24
    tokens = torch.zeros(B, T, dtype=torch.long)
    dt = torch.zeros(B, T)
    lengths = torch.zeros(B, dtype=torch.long)
    pool = np.array([2000., 900., 900., 900., 365., 365., 90., 90., 90., 30.,
                     30., 7., 7., 7., 3., 1., 1., 1., 1., 1., 1., 1., 1.])
    for b in range(B):
        L = int(rng.integers(12, T))
        vals = np.sort(rng.choice(pool, L - 1, replace=True))[::-1].copy()
        dt[b, :L - 1] = torch.tensor(vals, dtype=torch.float32)
        dt[b, L - 1] = 0.0
        tokens[b, :L] = torch.from_numpy(rng.integers(1, VOCAB, L))
        lengths[b] = L
    return tokens, dt, lengths


def test_parameter_count_is_about_two_million(model):
    n = model.n_params()
    assert 1.5e6 < n < 2.5e6, f"{n:,} params -- Delphi-2M is 2.2M at this task"


def test_forward_and_lm_forward_are_finite(model, batch):
    tokens, dt, lengths = batch
    with torch.no_grad():
        out, lm = model(tokens, dt, lengths), model.lm_forward(tokens, dt)
    # Asserted against the config, not a literal. These pinned 41, which no
    # trained model has ever had: train() builds every model with
    # n_outputs=y.shape[1] and the label array has 40 columns, so the 41st
    # head was dead code and these tests were guarding a shape that only
    # existed in the default. Reading it from the config means the test
    # follows a deliberate change and still catches an accidental one.
    assert out.shape == (len(tokens), model.cfg.n_outputs)
    assert lm.shape == (*tokens.shape, VOCAB)
    # The strict mask hides everything at the current instant, so the first
    # position sees only itself. Without the diagonal added back its softmax
    # row would be all -inf and every gradient would be NaN.
    assert torch.isfinite(out).all() and torch.isfinite(lm).all()


def test_bucket_zero_means_exactly_simultaneous(model):
    """Bucket 0 must be reachable only by Δt == 0, and the top edge must
    exceed the largest Δt in the data (measured 37,284 days) so the longest
    histories do not all collapse together."""
    e = model.bucket_edges
    assert torch.bucketize(torch.tensor([0.0]), e).item() == 0
    # 55 seconds is the smallest non-zero gap measured in this dataset.
    assert torch.bucketize(torch.tensor([55.0 / 86400.0]), e).item() == 1
    assert e[-1].item() >= 37_284.0


def test_permutation_of_simultaneous_events_does_not_change_predictions(model, batch):
    """C4. The mask is built from ``dt``, not from the index, so it is
    block-constant within a timestamp and reordering inside a block is
    invisible to attention.

    The tolerance is not slack: float32 addition is not associative, so
    reordering changes the summation order inside the attention weighted sum.
    ``eps * sqrt(T)`` at T=24 is ~6e-7. An assertion of exact equality would
    fail for a correct model, and a tolerance of 1e-3 would pass for a broken
    one that used index positions.
    """
    tokens, dt, lengths = batch
    rng = np.random.default_rng(1)
    permuted = tokens.clone()
    n_groups = 0
    for b in range(len(tokens)):
        L = int(lengths[b])
        i = 0
        while i < L - 1:                        # body only; [ANCHOR] stays last
            j = i
            while j + 1 < L - 1 and dt[b, j + 1] == dt[b, i]:
                j += 1
            if j > i:
                order = torch.from_numpy(rng.permutation(j - i + 1) + i)
                permuted[b, i:j + 1] = tokens[b, order]
                n_groups += 1
            i = j + 1

    assert n_groups >= 8, f"only {n_groups} tie-groups -- fixture is not exercising ties"
    assert not torch.equal(permuted, tokens), "permutation was a no-op"

    with torch.no_grad():
        a, b_ = model(tokens, dt, lengths), model(permuted, dt, lengths)
    delta = (a - b_).abs().max().item()
    assert delta < 1e-5, f"max|delta| = {delta:.2e} -- order is leaking into the output"


def test_padding_cannot_influence_the_readout(model, batch):
    """Garbage in the padded tail must not move a single logit."""
    tokens, dt, lengths = batch
    poisoned, poisoned_dt = tokens.clone(), dt.clone()
    for b in range(len(tokens)):
        L = int(lengths[b])
        poisoned_dt[b, L:] = 99999.0            # pad keeps id 0; only dt is junk
    with torch.no_grad():
        a, b_ = model(tokens, dt, lengths), model(poisoned, poisoned_dt, lengths)
    assert (a - b_).abs().max().item() < 1e-5


# --------------------------------------------------- the four coherent arms --
@pytest.mark.parametrize("causal,readout", [
    (True, "last"),     # P1/P4  Delphi-2M, ETHOS
    (False, "mean"),    # P2/P3  BEHRT, CEHR-BERT, CORE-BEHRT
])
def test_both_coherent_arms_run_and_stay_finite(batch, causal, readout):
    tokens, dt, lengths = batch
    torch.manual_seed(0)
    m = PatientTransformer(GPTConfig(vocab_size=VOCAB, causal=causal,
                                     readout=readout)).eval()
    with torch.no_grad():
        out = m(tokens, dt, lengths)
    # Asserted against the config, not a literal. These pinned 41, which no
    # trained model has ever had: train() builds every model with
    # n_outputs=y.shape[1] and the label array has 40 columns, so the 41st
    # head was dead code and these tests were guarding a shape that only
    # existed in the default. Reading it from the config means the test
    # follows a deliberate change and still catches an accidental one.
    assert out.shape == (len(tokens), m.cfg.n_outputs)
    assert torch.isfinite(out).all()


def test_bidirectional_actually_sees_the_future(batch):
    """Otherwise the 'bidirectional' arm is silently a second causal arm.

    Changing a token that lies AFTER position 0 must move position 0's
    representation under a bidirectional mask, and must not under a causal one.
    """
    tokens, dt, lengths = batch
    tokens = tokens.clone()
    altered = tokens.clone()
    altered[0, int(lengths[0]) - 2] = (altered[0, int(lengths[0]) - 2] % 500) + 300

    for causal, expect_change in ((False, True), (True, False)):
        torch.manual_seed(0)
        m = PatientTransformer(GPTConfig(vocab_size=VOCAB, causal=causal)).eval()
        with torch.no_grad():
            a = m.encode(tokens, dt)[0, 0]
            b = m.encode(altered, dt)[0, 0]
        moved = (a - b).abs().max().item() > 1e-6
        assert moved == expect_change, (
            f"causal={causal}: position 0 {'moved' if moved else 'did not move'} "
            f"when a later token changed")


def test_lm_pretraining_is_causal_even_on_a_bidirectional_model(batch):
    """Next-token prediction under a bidirectional mask is trivially solved --
    the target is in the input. `lm_forward` must force the causal mask."""
    tokens, dt, _ = batch
    torch.manual_seed(0)
    m = PatientTransformer(GPTConfig(vocab_size=VOCAB, causal=False)).eval()
    with torch.no_grad():
        a = m.lm_forward(tokens, dt)
        b = m.mlm_forward(tokens, dt)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert (a - b).abs().max().item() > 1e-4, \
        "lm_forward and mlm_forward produced the same thing -- the mask is not switching"


# ------------------------------------------------- pretraining objectives --
def test_lm_pretraining_cannot_see_the_token_it_predicts(batch):
    """The trap that makes next-token pretraining worthless.

    ``_attn_bias`` always ORs the identity into the visibility mask, so every
    position can see its own token. That is required for the classification
    pass -- without it a row could be fully masked and the softmax would be
    all -inf -- but it means an UNSHIFTED LM objective is a copy task.

    The training loop shifts targets so position i predicts token i+1. This
    asserts the property that makes the shift sound: position i's output must
    not move when token i+1 changes.
    """
    tokens, dt, lengths = batch
    torch.manual_seed(0)
    m = PatientTransformer(GPTConfig(vocab_size=VOCAB)).eval()

    # Pick a position whose successor is at a STRICTLY later time, so the
    # strict mask genuinely separates them.
    b = 0
    i = next(k for k in range(int(lengths[b]) - 1) if dt[b, k] > dt[b, k + 1])
    altered = tokens.clone()
    altered[b, i + 1] = (int(altered[b, i + 1]) % 500) + 300
    assert altered[b, i + 1] != tokens[b, i + 1], "fixture did not change the token"

    with torch.no_grad():
        a = m.lm_forward(tokens, dt)[b, i]
        c = m.lm_forward(altered, dt)[b, i]
    assert (a - c).abs().max().item() < 1e-6, (
        "position i's next-token logits moved when token i+1 changed -- the "
        "LM objective is leaking its own target")


def test_mlm_scores_only_the_masked_positions():
    """Scoring every position lets the unmasked majority dominate with a copy
    task, and the pretraining loss falls while nothing is learned."""
    import torch.nn.functional as Fn
    torch.manual_seed(0)
    m = PatientTransformer(GPTConfig(vocab_size=VOCAB, causal=False)).eval()
    tok = torch.randint(1, VOCAB - 1, (4, 12))
    dt = torch.arange(12, 0, -1).float().repeat(4, 1)
    sel = torch.zeros_like(tok, dtype=torch.bool)
    sel[:, 3] = True

    corrupted = tok.masked_fill(sel, VOCAB - 1)
    with torch.no_grad():
        logits = m.mlm_forward(corrupted, dt)
    scored = Fn.cross_entropy(logits[sel], tok[sel])
    assert scored.numel() == 1 and torch.isfinite(scored)
    assert logits[sel].shape == (4, VOCAB)
    # At init the loss should be about ln(V): the model knows nothing yet. A
    # value far below that would mean the answer is reachable from the input.
    assert abs(float(scored) - math.log(VOCAB)) < 1.5, float(scored)


def test_batched_prediction_restores_input_row_order():
    """Length-bucketed batching reorders rows; `predict` must undo it.

    If it does not, every patient is scored against a different patient's
    outcomes. The metrics would still look plausible -- slightly worse, not
    obviously broken -- and no other test in this suite would fail. Checked
    against a one-row-at-a-time reference, where reordering is impossible.
    """
    from src.train_finetune import predict

    rng = np.random.default_rng(0)
    n, T = 40, 16
    tok = np.zeros((n, T), np.int32)
    dt = np.zeros((n, T), np.float32)
    lengths = np.zeros(n, np.int32)
    for i in range(n):
        L = int(rng.integers(4, T))
        tok[i, :L] = rng.integers(1, VOCAB, L)
        dt[i, :L] = np.sort(rng.random(L))[::-1] * 100
        lengths[i] = L

    torch.manual_seed(0)
    m = PatientTransformer(GPTConfig(vocab_size=VOCAB, n_outputs=40)).eval()
    rows = np.arange(5, 35)

    batched = predict(m, tok, dt, lengths, rows, batch_size=7)
    one_by_one = np.vstack([predict(m, tok, dt, lengths, np.array([r]), batch_size=1)
                            for r in rows])
    delta = np.abs(batched - one_by_one).max()
    assert delta < 1e-5, f"row order lost in batching: max|delta| = {delta:.2e}"


# ------------------------------------------------------- feature fusion ----
@pytest.mark.parametrize("mode", ["readout", "token"])
def test_feature_fusion_actually_reaches_the_output(mode):
    """A fused input that changes nothing is worse than no fusion at all.

    The `token` mode failed this when first written. The feature vector was
    prepended at dt = 0, but dt counts DOWN to the anchor, so dt = 0 is the
    most recent instant and the causal rule (attend to j iff dt[j] >= dt[i])
    made it invisible to every other position. The model ran, trained, and
    reported plausible numbers while the features contributed exactly
    nothing -- measured output delta 0.0000.
    """
    torch.manual_seed(0)
    n_feat = 128
    m = PatientTransformer(GPTConfig(vocab_size=VOCAB, n_outputs=40,
                                     fusion=mode, n_features=n_feat)).eval()
    tok = torch.randint(1, VOCAB, (4, 12))
    dt = torch.arange(12, 0, -1).float().repeat(4, 1)
    lengths = torch.full((4,), 12)
    fa, fb = torch.randn(4, n_feat), torch.randn(4, n_feat)

    with torch.no_grad():
        delta = (m(tok, dt, lengths, features=fa)
                 - m(tok, dt, lengths, features=fb)).abs().max().item()
    assert delta > 1e-4, f"{mode} fusion is a no-op: delta {delta:.2e}"


@pytest.mark.parametrize("mode", ["readout", "token"])
def test_modality_dropout_at_one_is_identical_to_having_no_features(mode):
    """p = 1.0 in TRAINING must reproduce the features-zeroed forward exactly.

    Behavioural, not a flag check: it does not assert that `modality_dropout`
    is set, it asserts that setting it produces the output the model gives when
    the tabular branch is genuinely absent. That is the property the arm is
    supposed to train against, and it is the same condition the collapse check
    evaluates at.

    The non-vacuity guard matters more than the equality here. If features did
    not reach the output at all -- which is exactly what `token` fusion did
    before D5, at a measured delta of 0.0000 -- then zeroing them would trivially
    change nothing and this test would pass while testing nothing.
    """
    torch.manual_seed(0)
    n_feat = 128
    # attn/resid dropout MUST be silenced to isolate the variable. They
    # default to 0.1 and are active in train mode, so two identical forwards
    # differ by ~0.24 and every assertion below would be measuring them instead
    # of modality dropout. Config (1) runs at 0.0 anyway, so this is also the
    # realistic setting rather than a convenience.
    m = PatientTransformer(GPTConfig(vocab_size=VOCAB, n_outputs=40,
                                     fusion=mode, n_features=n_feat,
                                     attn_dropout=0.0, resid_dropout=0.0,
                                     modality_dropout=1.0))
    tok = torch.randint(1, VOCAB, (4, 12))
    dt = torch.arange(12, 0, -1).float().repeat(4, 1)
    lengths = torch.full((4,), 12)
    f = torch.randn(4, n_feat)

    m.train()
    with torch.no_grad():
        dropped = m(tok, dt, lengths, features=f)
        zeroed = m(tok, dt, lengths, features=torch.zeros_like(f))
    assert torch.allclose(dropped, zeroed, atol=1e-6), (
        "modality_dropout=1.0 did not reproduce the features-zeroed forward")

    # Non-vacuity: the features must genuinely matter, or the above is empty.
    m.eval()
    with torch.no_grad():
        delta = (m(tok, dt, lengths, features=f)
                 - m(tok, dt, lengths, features=torch.zeros_like(f))
                 ).abs().max().item()
    assert delta > 1e-4, f"vacuous: features do not reach the output ({delta:.2e})"


def test_modality_dropout_is_training_only():
    """In eval the branch must be intact however high p is set.

    Otherwise every reported metric would be computed on a randomly crippled
    model, and the damage would look like noise rather than like a bug.
    """
    torch.manual_seed(0)
    n_feat = 64
    m = PatientTransformer(GPTConfig(vocab_size=VOCAB, n_outputs=40,
                                     fusion="readout", n_features=n_feat,
                                     attn_dropout=0.0, resid_dropout=0.0,
                                     modality_dropout=1.0)).eval()
    ref = PatientTransformer(GPTConfig(vocab_size=VOCAB, n_outputs=40,
                                       fusion="readout", n_features=n_feat,
                                       attn_dropout=0.0, resid_dropout=0.0,
                                       modality_dropout=0.0)).eval()
    ref.load_state_dict(m.state_dict())
    tok = torch.randint(1, VOCAB, (3, 9))
    dt = torch.arange(9, 0, -1).float().repeat(3, 1)
    lengths = torch.full((3,), 9)
    f = torch.randn(3, n_feat)
    with torch.no_grad():
        assert torch.allclose(m(tok, dt, lengths, features=f),
                              ref(tok, dt, lengths, features=f), atol=1e-6), (
            "modality_dropout leaked into evaluation")


def test_modality_dropout_default_is_inert():
    """The default must reproduce the pre-existing model bit-for-bit.

    Every number measured before this switch existed was produced at p = 0, so
    if the default were not inert the new arm would silently re-baseline the
    whole project -- the failure mode G4 warns about.
    """
    torch.manual_seed(0)
    n_feat = 64
    # attn/resid dropout MUST be silenced to isolate the variable. They
    # default to 0.1 and are active in train mode, so two identical forwards
    # differ by ~0.24 and every assertion below would be measuring them instead
    # of modality dropout. Config (1) runs at 0.0 anyway, so this is also the
    # realistic setting rather than a convenience.
    cfg = dict(vocab_size=VOCAB, n_outputs=40, fusion="readout",
               n_features=n_feat, attn_dropout=0.0, resid_dropout=0.0)
    m = PatientTransformer(GPTConfig(**cfg))
    assert m.cfg.modality_dropout == 0.0
    tok = torch.randint(1, VOCAB, (3, 9))
    dt = torch.arange(9, 0, -1).float().repeat(3, 1)
    lengths = torch.full((3,), 9)
    f = torch.randn(3, n_feat)
    m.train()
    with torch.no_grad():
        a = m(tok, dt, lengths, features=f)
        b = m(tok, dt, lengths, features=f)
    assert torch.equal(a, b), "p=0 is not deterministic in training mode"


def test_modality_dropout_is_per_example_not_per_unit():
    """At an intermediate p some rows must be dropped and others kept.

    This is the distinction from `resid_dropout`, which thins units inside the
    projection and leaves every example's tabular branch informative. If the
    mask were per-unit the arm would not be testing what it claims to test.
    """
    torch.manual_seed(0)
    n_feat = 32
    # attn/resid dropout MUST be silenced to isolate the variable. They
    # default to 0.1 and are active in train mode, so two identical forwards
    # differ by ~0.24 and every assertion below would be measuring them instead
    # of modality dropout. Config (1) runs at 0.0 anyway, so this is also the
    # realistic setting rather than a convenience.
    m = PatientTransformer(GPTConfig(vocab_size=VOCAB, n_outputs=40,
                                     fusion="readout", n_features=n_feat,
                                     attn_dropout=0.0, resid_dropout=0.0,
                                     modality_dropout=0.5))
    m.train()
    tok = torch.randint(1, VOCAB, (64, 6))
    dt = torch.arange(6, 0, -1).float().repeat(64, 1)
    lengths = torch.full((64,), 6)
    f = torch.randn(64, n_feat)
    with torch.no_grad():
        out = m(tok, dt, lengths, features=f)
        zero = m(tok, dt, lengths, features=torch.zeros_like(f))
    # Rows whose features were dropped match the all-zero forward exactly.
    matched = torch.isclose(out, zero, atol=1e-6).all(dim=1)
    assert 0 < int(matched.sum()) < 64, (
        f"expected a mix of dropped and kept rows, got {int(matched.sum())}/64")


def test_fused_feature_token_is_visible_to_the_whole_sequence():
    """Under `token` fusion the summary must reach other positions, not just
    sit there. That is the whole difference from late fusion."""
    torch.manual_seed(0)
    n_feat = 128
    m = PatientTransformer(GPTConfig(vocab_size=VOCAB, n_outputs=40,
                                     fusion="token", n_features=n_feat)).eval()
    tok = torch.randint(1, VOCAB, (4, 12))
    dt = torch.arange(12, 0, -1).float().repeat(4, 1)
    fa, fb = torch.randn(4, n_feat), torch.randn(4, n_feat)

    def encoded(f):
        vec = m.feat_to_tok(m.feat_proj(f)).unsqueeze(1)
        oldest = dt.max(dim=1, keepdim=True).values
        return m.encode(torch.cat([torch.ones(4, 1, dtype=tok.dtype), tok], 1),
                        torch.cat([oldest, dt], 1), inject=(0, vec))

    with torch.no_grad():
        ha, hb = encoded(fa), encoded(fb)
    assert (ha[:, 0] - hb[:, 0]).abs().max().item() > 1e-4, "injection failed"
    assert (ha[:, -1] - hb[:, -1]).abs().max().item() > 1e-5, \
        "the feature token never reaches the readout position through attention"


# --------------------------------------------------------------------------
# Ablation switches for the design-claim experiment (W7).
#
# `docs/01-why-time.md` argues formally that without a positional signal a
# transformer sees only a multiset. That argument is the intellectual centre of
# the submission and nothing measured it, because until now the model had no
# switch to turn the time signals off. These tests assert the switches do what
# they claim BEHAVIOURALLY -- that removing a signal removes the model's
# sensitivity to it -- rather than asserting that a flag is set.
# --------------------------------------------------------------------------

def _ablated(**kw):
    torch.manual_seed(0)
    m = PatientTransformer(GPTConfig(vocab_size=VOCAB, **kw))
    m.eval()
    return m


def _fixed_batch(B=4, T=16, seed=3):
    """Every row full-length, so a permutation needs no pad bookkeeping."""
    rng = np.random.default_rng(seed)
    tokens = torch.from_numpy(rng.integers(1, VOCAB, (B, T))).long()
    base = np.sort(rng.choice([2000., 900., 365., 90., 30., 7., 1.],
                              (B, T - 1), replace=True), axis=1)[:, ::-1].copy()
    dt = torch.zeros(B, T)
    dt[:, :T - 1] = torch.from_numpy(base).float()
    lengths = torch.full((B,), T, dtype=torch.long)
    return tokens, dt, lengths


def test_ablating_both_time_signals_makes_dt_magnitude_irrelevant():
    """With Time2Vec and the Δt bias off, scaling every gap must change nothing.

    Doubling dt preserves order and every tie, so the visibility mask is
    identical; only the *magnitudes* differ. A model with no time signals must
    be blind to that.
    """
    tokens, dt, lengths = _fixed_batch()
    m = _ablated(use_time_encoding=False, use_dt_bias=False)
    with torch.no_grad():
        a, b = m(tokens, dt, lengths), m(tokens, dt * 2.0, lengths)
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max().item()


def test_the_dt_scaling_probe_is_not_vacuous():
    """Positive control: the DEFAULT model must be sensitive to that scaling.

    Without this, the test above would pass against a probe that never
    perturbed anything.
    """
    tokens, dt, lengths = _fixed_batch()
    m = _ablated()
    with torch.no_grad():
        a, b = m(tokens, dt, lengths), m(tokens, dt * 2.0, lengths)
    assert not torch.allclose(a, b, atol=1e-6), "probe moved nothing"


def test_dt_bias_off_alone_still_leaves_the_model_time_aware():
    """Each switch removes its own signal, not the other's."""
    tokens, dt, lengths = _fixed_batch()
    m = _ablated(use_dt_bias=False)
    with torch.no_grad():
        a, b = m(tokens, dt, lengths), m(tokens, dt * 2.0, lengths)
    assert not torch.allclose(a, b, atol=1e-6), "Time2Vec should still see dt"


def test_ablating_time_does_NOT_by_itself_produce_a_bag_of_codes():
    """The subtlety that makes the 4-arm framing of this ablation wrong.

    The visibility mask is derived from `dt`, so a causal model still knows
    which event came first even with both explicit time signals removed.
    Permuting positions must therefore still move the output. A "no time" arm
    is *not* the multiset baseline that `01-why-time.md` reasons about.
    """
    tokens, dt, lengths = _fixed_batch()
    m = _ablated(use_time_encoding=False, use_dt_bias=False)
    perm = torch.randperm(tokens.shape[1], generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        a = m(tokens, dt, lengths)
        b = m(tokens[:, perm], dt[:, perm], lengths)
    assert not torch.allclose(a, b, atol=1e-6), (
        "order should still reach the model through the visibility mask")


def test_the_true_multiset_baseline_is_permutation_invariant():
    """`01-why-time.md`'s claim, made executable.

    Drop both time signals AND the direction constraint AND the position-picking
    readout, and what is left provably cannot distinguish orderings: every
    position sees every other with zero bias, and mean pooling is symmetric.
    This is the arm the time ablation must compare against.
    """
    tokens, dt, lengths = _fixed_batch()
    m = _ablated(use_time_encoding=False, use_dt_bias=False,
                 causal=False, readout="mean")
    perm = torch.randperm(tokens.shape[1], generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        a = m(tokens, dt, lengths)
        b = m(tokens[:, perm], dt[:, perm], lengths)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max().item()


def test_ablation_switches_do_not_change_the_parameter_count():
    """Zeroing a signal, not deleting the tensor, so capacity is held fixed.

    An ablation that also shrinks the model measures capacity, not the
    component -- the most common way this experiment is got wrong.
    """
    full = _ablated().n_params()
    for kw in ({"use_time_encoding": False}, {"use_dt_bias": False},
               {"use_time_encoding": False, "use_dt_bias": False}):
        assert _ablated(**kw).n_params() == full, kw
