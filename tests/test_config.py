"""`configs/default.yaml` is documentation, and this keeps it honest.

No code reads the YAML -- the dataclasses hold the real defaults. That is a
deliberate choice (one source of truth, no parallel config plumbing to keep in
sync at runtime) but it has an obvious failure mode: the file drifts, and then
it actively misleads. It already had, listing a fusion threshold of 500
against an actual 50, and an SVD block that was dropped on evidence.

So the YAML is pinned to the dataclasses here. If a default changes, this test
fails and the documentation gets updated in the same commit.
"""
from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from src.data.examples import ExampleConfig
from src.data.features import FeatureConfig
from src.data.sequences import SeqConfig
from src.models.gpt import GPTConfig
from src.train_baseline import GBDTConfig, LRConfig
from src.train_finetune import ARMS, TrainConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())


def test_vocab_and_sequence_match(cfg):
    s = SeqConfig()
    assert cfg["vocab"]["min_patients_per_code"] == s.min_patients_per_code
    assert cfg["vocab"]["n_value_bins"] == s.n_value_bins
    assert cfg["vocab"]["fuse_value_min_events"] == s.fuse_min_events_per_bin
    assert cfg["sequence"]["block_size"] == s.block_size


def test_feature_defaults_match(cfg):
    f, c = FeatureConfig(), cfg["features"]
    assert c["lab_min_coverage"] == f.lab_min_coverage
    assert c["n_time_since_common"] == f.n_time_since_common
    for flag in ("reason", "slope", "time_since", "cost", "age_residual"):
        assert c[f"enable_{flag}"] == getattr(f, f"enable_{flag}"), flag
    # windows: YAML writes `null` where the code uses inf
    want = [None if not (w == w and w != float("inf")) else int(w) for w in f.windows]
    assert c["windows_days"] == want


def test_example_defaults_match(cfg):
    e, c = ExampleConfig(), cfg["examples"]
    assert c["augment"] == e.augment, (
        "augmentation default disagrees -- it is OFF on measured evidence")
    assert c["stride_years"] == e.stride_years
    assert c["min_events"] == e.min_events


def test_baseline_defaults_match(cfg):
    g, c = GBDTConfig(), cfg["baseline"]["gbdt"]
    assert c["learning_rate"] == g.learning_rate
    assert c["max_iter"] == g.max_iter
    assert c["max_leaf_nodes"] == g.max_leaf_nodes
    assert c["min_samples_leaf"] == g.min_samples_leaf
    assert c["l2_regularization"] == g.l2_regularization
    assert c["early_stopping"] == g.early_stopping
    assert cfg["baseline"]["lr_C"] == LRConfig().C


def test_transformer_defaults_and_arms_match(cfg):
    g, t, c = GPTConfig(vocab_size=1), TrainConfig(), cfg["transformer"]
    assert c["n_layer"] == g.n_layer
    assert c["n_embd"] == g.n_embd
    assert c["n_head"] == g.n_head
    assert c["d_time"] == g.d_time
    assert c["n_dt_buckets"] == g.n_dt_buckets
    assert c["max_dt_days"] == g.max_dt_days
    assert c["batch_size"] == t.batch_size
    assert c["lr"] == t.lr
    assert c["grad_clip"] == t.grad_clip
    assert c["attn_dropout"] == g.attn_dropout

    assert set(c["arms"]) == set(ARMS)
    for name, spec in ARMS.items():
        y = c["arms"][name]
        assert y["causal"] == spec["causal"], name
        assert y["readout"] == spec["readout"], name
        assert y["pretrain"] == spec["pretrain"], name


def test_forbidden_columns_match_the_code(cfg):
    from src.data.cohort import FORBIDDEN_PATIENT_COLS
    for col in cfg["forbidden"]["patient_columns"]:
        assert col in FORBIDDEN_PATIENT_COLS, col
