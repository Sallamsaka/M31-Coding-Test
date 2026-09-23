"""Behavioural tests for the clinical grouping and family shrinkage.

Behavioural, not flag-based: these assert that shrinkage MOVES the right
predictions in the right direction and leaves the right ones alone, rather than
that some `strength` attribute was set.
"""
import numpy as np
import pandas as pd
import pytest

from src.hierarchy import (FAMILIES, family_matrix, family_of,
                           shrink_to_family)


@pytest.fixture(scope="module")
def codes():
    return pd.read_csv("outputs/per_code_val.csv").code.to_numpy()


def test_every_target_has_exactly_one_family(codes):
    """A label in two families would be shrunk twice, toward different means."""
    flat = [c for cs in FAMILIES.values() for c in cs]
    assert len(flat) == len(set(flat)), "a code appears in two families"
    assert len(flat) == 40, f"families cover {len(flat)} labels, expected 40"
    assert set(flat) == {int(c) for c in codes}, "families do not match targets"
    family_of(codes)            # raises if any target is unassigned


def test_family_matrix_excludes_self(codes):
    same = family_matrix(codes)
    assert not same.diagonal().any(), "a label is its own family member"
    assert (same == same.T).all(), "family membership must be symmetric"


def test_strength_zero_is_exactly_identity(codes):
    """The null arm of the experiment must be a true no-op, or the comparison
    measures the implementation rather than the idea."""
    rng = np.random.default_rng(0)
    P = rng.uniform(0.01, 0.99, size=(50, 40))
    n_pos = rng.integers(5, 100, size=40)
    out = shrink_to_family(P, codes, n_pos, strength=0.0)
    assert np.allclose(out, P, atol=1e-9), "strength=0 changed the predictions"


def test_rare_labels_move_more_than_common_ones(codes):
    """The whole point: a 5-positive label should borrow, a 99-positive one
    should not. Asserted as a behaviour, on labels in the SAME family so the
    comparison is not confounded by which mean they shrink toward."""
    fam = family_of(codes)
    resp = np.flatnonzero(fam == "respiratory_infection")
    assert len(resp) >= 2

    rng = np.random.default_rng(1)
    P = rng.uniform(0.05, 0.95, size=(200, 40))
    n_pos = np.full(40, 50)
    rare, common = resp[0], resp[1]
    n_pos[rare], n_pos[common] = 5, 500

    out = shrink_to_family(P, codes, n_pos, strength=1.0)
    moved_rare = np.abs(out[:, rare] - P[:, rare]).mean()
    moved_common = np.abs(out[:, common] - P[:, common]).mean()
    assert moved_rare > moved_common * 3, (
        f"rare label moved {moved_rare:.4f}, common moved {moved_common:.4f} "
        "-- shrinkage is not weighting by evidence")


def test_singletons_are_never_shrunk(codes):
    """Grouping unlike labels together to avoid a leftover bucket is how
    shrinkage does harm, so the bucket must be inert."""
    fam = family_of(codes)
    single = np.flatnonzero(fam == "singleton")
    assert len(single) > 0

    rng = np.random.default_rng(2)
    P = rng.uniform(0.05, 0.95, size=(100, 40))
    n_pos = np.full(40, 8)          # maximal shrinkage pressure everywhere
    out = shrink_to_family(P, codes, n_pos, strength=5.0)
    assert np.allclose(out[:, single], P[:, single], atol=1e-9), (
        "singleton labels were shrunk toward a family mean that does not exist")


def test_shrinkage_pulls_toward_the_family_mean_not_the_global_mean(codes):
    """A label must borrow from its OWN family. If it drifts toward the global
    mean the grouping is doing nothing and this is just ridge on the logits."""
    fam = family_of(codes)
    msk = np.flatnonzero(fam == "msk_trauma")
    other = np.flatnonzero(fam == "respiratory_infection")

    P = np.full((1, 40), 0.5)
    P[0, msk] = 0.9                 # family runs high
    P[0, other] = 0.1               # a different family runs low
    n_pos = np.full(40, 10)

    out = shrink_to_family(P, codes, n_pos, strength=3.0)
    # a heavily shrunk msk label stays near its family's high value
    assert out[0, msk].min() > 0.6, (
        "an msk label was pulled below its family's level -- it is borrowing "
        "from outside its family")
    assert out[0, other].max() < 0.4, "a respiratory label drifted upward"
