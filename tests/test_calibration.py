"""Unit tests for calibration. No model required."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from guardrails.calibration import (
    Recommendation,
    load,
    pick_by_cost,
    pick_by_f1,
    pick_by_fpr_budget,
    save,
)


@pytest.fixture
def separable_scores() -> tuple[np.ndarray, np.ndarray]:
    """Near-perfect scores where negatives cluster near 0 and positives near 1."""
    rng = np.random.default_rng(0)
    neg = rng.uniform(0.0, 0.3, size=100)
    pos = rng.uniform(0.7, 1.0, size=100)
    probs = np.concatenate([neg, pos])
    labels = np.concatenate([np.zeros(100, dtype=int), np.ones(100, dtype=int)])
    return probs, labels


@pytest.fixture
def noisy_scores() -> tuple[np.ndarray, np.ndarray]:
    """Overlapping distributions so cost weighting has a real effect."""
    rng = np.random.default_rng(1)
    neg = rng.uniform(0.0, 0.7, size=200)
    pos = rng.uniform(0.3, 1.0, size=200)
    probs = np.concatenate([neg, pos])
    labels = np.concatenate([np.zeros(200, dtype=int), np.ones(200, dtype=int)])
    return probs, labels


def test_cost_equal_weights_on_separable_data_is_perfect(separable_scores):
    probs, labels = separable_scores
    rec = pick_by_cost(probs, labels, cost_fp=1.0, cost_fn=1.0)
    assert rec.precision == 1.0
    assert rec.recall == 1.0
    assert 0.3 < rec.threshold < 0.7


def test_f1_on_separable_data_is_perfect(separable_scores):
    probs, labels = separable_scores
    rec = pick_by_f1(probs, labels)
    assert rec.f1 == 1.0
    assert rec.mode == "f1"


def test_f1_beats_symmetric_cost_on_imbalanced_data():
    """On 90/10 imbalanced data, F1 should find a better operating point than naive 1:1 cost."""
    rng = np.random.default_rng(7)
    neg = rng.uniform(0.0, 0.6, size=900)  # 90% negatives, noisy
    pos = rng.uniform(0.4, 1.0, size=100)  # 10% positives, noisy
    probs = np.concatenate([neg, pos])
    labels = np.concatenate([np.zeros(900, dtype=int), np.ones(100, dtype=int)])

    f1_rec = pick_by_f1(probs, labels)
    cost_rec = pick_by_cost(probs, labels, cost_fp=1.0, cost_fn=1.0)
    assert f1_rec.f1 >= cost_rec.f1


def test_f1_mode_saves_as_f1(separable_scores):
    probs, labels = separable_scores
    rec = pick_by_f1(probs, labels)
    assert rec.mode == "f1"
    assert rec.criterion == "max F1"


def test_cost_fn_heavy_lowers_threshold(noisy_scores):
    probs, labels = noisy_scores
    symmetric = pick_by_cost(probs, labels, cost_fp=1.0, cost_fn=1.0)
    fn_heavy = pick_by_cost(probs, labels, cost_fp=1.0, cost_fn=100.0)
    assert fn_heavy.threshold <= symmetric.threshold
    assert fn_heavy.recall >= symmetric.recall


def test_cost_fp_heavy_raises_threshold(noisy_scores):
    probs, labels = noisy_scores
    symmetric = pick_by_cost(probs, labels, cost_fp=1.0, cost_fn=1.0)
    fp_heavy = pick_by_cost(probs, labels, cost_fp=100.0, cost_fn=1.0)
    assert fp_heavy.threshold >= symmetric.threshold
    assert fp_heavy.fpr <= symmetric.fpr + 1e-9


def test_fpr_budget_respects_budget(noisy_scores):
    probs, labels = noisy_scores
    for budget in (0.01, 0.05, 0.1):
        rec = pick_by_fpr_budget(probs, labels, max_fpr=budget, min_threshold=0.0)
        assert rec.fpr <= budget + 1e-9


def test_fpr_budget_picks_lowest_threshold_under_budget(separable_scores):
    probs, labels = separable_scores
    rec = pick_by_fpr_budget(probs, labels, max_fpr=0.0, min_threshold=0.0)
    # FPR=0 is achievable with separable data; resulting recall should still be perfect
    assert rec.fpr == 0.0
    assert rec.recall == 1.0


def test_fpr_budget_respects_min_threshold_floor(noisy_scores):
    """The floor must prevent near-zero thresholds even if val FPR allows them."""
    probs, labels = noisy_scores
    rec = pick_by_fpr_budget(probs, labels, max_fpr=0.99, min_threshold=0.5)
    # Without floor the algorithm would pick the lowest threshold; with floor=0.5 it can't
    assert rec.threshold >= 0.5
    assert "floor hit" in rec.criterion


def test_fpr_budget_floor_absent_when_not_hit(noisy_scores):
    probs, labels = noisy_scores
    rec = pick_by_fpr_budget(probs, labels, max_fpr=0.01, min_threshold=0.0)
    assert "floor hit" not in rec.criterion


def test_fpr_budget_rejects_invalid_min_threshold(noisy_scores):
    probs, labels = noisy_scores
    with pytest.raises(ValueError):
        pick_by_fpr_budget(probs, labels, max_fpr=0.05, min_threshold=1.5)


def test_cost_rejects_negative_weights(noisy_scores):
    probs, labels = noisy_scores
    with pytest.raises(ValueError):
        pick_by_cost(probs, labels, cost_fp=-1.0, cost_fn=1.0)


def test_fpr_budget_rejects_out_of_range(noisy_scores):
    probs, labels = noisy_scores
    with pytest.raises(ValueError):
        pick_by_fpr_budget(probs, labels, max_fpr=1.5)


def test_save_load_roundtrip(tmp_path, separable_scores):
    probs, labels = separable_scores
    rec = pick_by_cost(probs, labels)
    path = tmp_path / "threshold.json"
    save(rec, path)
    loaded = load(path)
    assert loaded == rec


def test_load_missing_file_returns_none(tmp_path):
    assert load(tmp_path / "nope.json") is None


def test_recommendation_is_frozen():
    rec = Recommendation(
        threshold=0.5,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        fpr=0.0,
        tpr=1.0,
        accuracy=1.0,
        mode="cost",
        criterion="",
        n=1,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.threshold = 0.1  # type: ignore[misc]  # ty: ignore[invalid-assignment]
