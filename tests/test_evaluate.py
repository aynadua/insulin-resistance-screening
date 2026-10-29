"""Evaluation metrics, checked on cases with known answers."""

import numpy as np
import pytest

from irisk.evaluate import (
    calibration,
    net_benefit,
    score,
    sensitivity_at_specificity,
    treat_all_net_benefit,
    weighted_auroc,
)


def test_perfect_ranking_scores_auroc_one():
    y = np.array([0, 0, 1, 1])
    assert weighted_auroc(y, np.array([0.1, 0.2, 0.8, 0.9]), np.ones(4)) == 1.0


def test_reversed_ranking_scores_zero():
    y = np.array([0, 0, 1, 1])
    assert weighted_auroc(y, np.array([0.9, 0.8, 0.2, 0.1]), np.ones(4)) == 0.0


def test_weights_change_auroc():
    """If weighting made no difference, the survey design would not matter."""
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.8, 0.2])
    flat = weighted_auroc(y, scores, np.ones(4))
    skewed = weighted_auroc(y, scores, np.array([100.0, 100.0, 1.0, 1.0]))
    assert flat != pytest.approx(skewed)


def test_sensitivity_at_specificity_respects_its_constraint():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    scores = y + rng.normal(0, 0.6, 500)
    sensitivity, _ = sensitivity_at_specificity(y, scores, np.ones(500), 0.90)
    assert 0.0 <= sensitivity <= 1.0


def test_a_useless_model_scores_half():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 4000)
    noise = rng.normal(size=4000)
    assert weighted_auroc(y, noise, np.ones(4000)) == pytest.approx(0.5, abs=0.04)


def test_calibration_is_perfect_when_predictions_are_true_rates():
    rng = np.random.default_rng(2)
    probabilities = rng.uniform(0.05, 0.95, 20000)
    y = (rng.uniform(size=20000) < probabilities).astype(int)
    calibrated = calibration(y, probabilities, np.ones(20000))
    assert calibrated["max_abs_gap"] < 0.05


def test_calibration_detects_overconfidence():
    """A model that always says 90% when the truth is 50% must be flagged."""
    y = np.array([1, 0] * 500)
    probabilities = np.full(1000, 0.9)
    calibrated = calibration(y, probabilities, np.ones(1000))
    assert calibrated["max_abs_gap"] > 0.3


def test_net_benefit_of_a_perfect_model_equals_prevalence():
    """With no false positives, net benefit is just the true positive rate."""
    y = np.array([1, 1, 0, 0])
    probabilities = np.array([0.99, 0.99, 0.01, 0.01])
    assert net_benefit(y, probabilities, np.ones(4), 0.5) == pytest.approx(0.5)


def test_treat_all_net_benefit_goes_negative_above_prevalence():
    """Treating everyone stops being worthwhile once the threshold exceeds prevalence."""
    y = np.array([1] + [0] * 9)  # 10% prevalence
    weights = np.ones(10)
    assert treat_all_net_benefit(y, weights, 0.05) > 0
    assert treat_all_net_benefit(y, weights, 0.50) < 0


def test_net_benefit_rejects_degenerate_thresholds():
    y = np.array([1, 0])
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="threshold probability"):
            net_benefit(y, np.array([0.5, 0.5]), np.ones(2), bad)


def test_scorecard_reports_nan_brier_without_probabilities():
    """A clinical index has an ordering but no probability scale."""
    y = np.array([0, 1, 0, 1])
    card = score("index", y, np.array([1.0, 4.0, 2.0, 5.0]), np.ones(4))
    assert np.isnan(card.brier)
    assert not np.isnan(card.auroc)
