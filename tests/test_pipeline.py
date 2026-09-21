"""Tests T12-T17: vocabulary, tokenizer determinism, and two falsification runs.

The last two are the ones that would catch a whole-pipeline error rather than a
local one. A pipeline can pass every unit test and still have X misaligned with
y; the only thing that finds that is destroying the signal on purpose and
checking the score collapses.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.data.cohort import load_cohort
from src.data.examples import ExampleConfig
from src.data.features import build_features
from src.data.labels import at_risk_mask, load_labels
from src.data.sequences import SeqConfig, build_sequences, build_vocab
from src.evaluate import macro_auroc
from src.train_baseline import LRConfig, fit_lr

ROOT = Path(__file__).resolve().parents[1]
SMALL = SeqConfig(block_size=128)


@pytest.fixture(scope="module")
def vocab():
    return build_vocab(ROOT, SMALL)


# ---------------------------------------------------------------- T12 ------
def test_vocabulary_is_fitted_on_training_patients_only(vocab):
    """Fitting the vocabulary on val or test is the quietest leak available.

    EHRSHOT's reference implementation makes exactly this mistake. The
    non-vacuity half matters as much as the assertion: if test produced no
    unknown tokens at all, the vocabulary would have seen everything and the
    restriction would be doing nothing.
    """
    assert vocab.meta["split"] == "train"
    assert vocab.meta["n_patients"] == 2791

    cohort = load_cohort(ROOT)
    pack = build_sequences(ROOT, vocab, SMALL)
    unk = {i for t, i in vocab.stoi.items() if t.startswith("[UNK_")}
    test_rows = pack.split == "test"
    n_unk = int(np.isin(pack.tokens[test_rows], list(unk)).sum())
    assert n_unk > 0, ("non-vacuity: test produced zero unknown tokens, so the "
                       "train-only restriction is not binding")


# ---------------------------------------------------------------- T13 ------
def test_quantile_edges_are_strictly_increasing(vocab):
    """Near-binary labs produce duplicate quantiles, and `searchsorted` then
    assigns every value to one bucket -- a silently dead feature block."""
    assert vocab.edges, "non-vacuity: no decile edges were fitted"
    for tok, e in vocab.edges.items():
        assert len(e) >= 2, tok
        assert all(b > a for a, b in zip(e, e[1:])), f"{tok}: {e}"


# ---------------------------------------------------------------- T14 ------
def test_no_patient_appears_in_two_splits():
    cohort = load_cohort(ROOT)
    by = {s: set(g.pid) for s, g in cohort.groupby("split")}
    assert by["train"] & by["val"] == set()
    assert by["train"] & by["test"] == set()
    assert by["val"] & by["test"] == set()
    assert sum(map(len, by.values())) == 3514


# ---------------------------------------------------------------- T16 ------
def test_tokenization_is_deterministic(vocab):
    """Two builds must be byte-identical, or no result is reproducible."""
    a = build_sequences(ROOT, vocab, SMALL)
    b = build_sequences(ROOT, vocab, SMALL)
    assert np.array_equal(a.tokens, b.tokens)
    assert np.array_equal(a.dt, b.dt)
    assert np.array_equal(a.lengths, b.lengths)
    h = hashlib.sha256(a.tokens.tobytes()).hexdigest()[:16]
    assert h == hashlib.sha256(b.tokens.tobytes()).hexdigest()[:16]


def test_sequences_and_labels_share_a_row_order(vocab):
    """Off-by-one here silently trains every patient against someone else's
    outcomes, and every other test would still pass."""
    pack = build_sequences(ROOT, vocab, SMALL)
    lab = load_labels(ROOT)
    assert np.array_equal(pack.eid, lab["eid"])
    assert np.array_equal(pack.pid, lab["pid"])
    F = build_features(ROOT)
    assert np.array_equal(F.eid, lab["eid"])


# ---------------------------------------------------------------- T17 ------
@pytest.mark.slow
def test_shuffled_labels_score_chance():
    """Destroy the signal on purpose; the score must collapse to chance.

    This is the only test here that can catch a whole-pipeline misalignment --
    X paired with the wrong y. If a model still scores well after the label
    rows are permuted, it is reading something it should not be able to read.

    Shuffling ROWS (not columns) preserves each label's prevalence and the
    correlation structure between labels, so the only thing destroyed is the
    patient-to-outcome correspondence.
    """
    cfg = ExampleConfig()
    F = build_features(ROOT, ex_cfg=cfg)
    lab = load_labels(ROOT, cfg)
    y = lab["y"].astype(int)
    ar = at_risk_mask(lab)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    P = fit_lr(F, y[perm], ar[perm], LRConfig(C=0.03), verbose=False)

    val = F.split == "val"
    au, n = macro_auroc(y[perm][val], P[val])
    assert n >= 35, "non-vacuity: too few scorable labels"
    assert 0.42 < au < 0.58, (
        f"shuffled-label macro AUROC {au:.4f} is not chance -- the pipeline is "
        "reading signal that survives destroying the patient-outcome pairing")
