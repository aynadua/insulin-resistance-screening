"""Evaluation for a screening model, which is not the same as for a classifier.

Accuracy is close to useless here. With insulin resistance at roughly 40%
prevalence, a model that answers "no" to everyone scores 60%, and a model
that answers "yes" to everyone scores 40%. Neither helps a clinician.

What a screening test is judged on:

Discrimination, by AUROC, which is the probability the model ranks a randomly
chosen resistant person above a randomly chosen sensitive one. It is
threshold-free and prevalence-independent, which makes it the right summary
and also means it cannot tell you whether the test is usable.

Operating points. A real test runs at a chosen threshold, so sensitivity at a
fixed specificity (and the converse) is what determines whether it can be
deployed. A screening test that misses half the cases is not redeemed by a
good AUROC.

Calibration. If the model says 30%, about 30% of those people should be
resistant. An uncalibrated model can discriminate perfectly and still be
unusable for a decision, because the number it emits does not mean what it
appears to mean.

Net benefit, via decision curve analysis. This is the one that answers the
question a clinician actually has: is acting on this model better than
treating everybody, or nobody? It weighs true positives against false
positives at the threshold probability where a reasonable person would switch
from watchful waiting to intervening.

Every metric takes survey weights, because the estimate being made is about
the United States population and not about the people who happened to be
sampled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def weighted_auroc(y_true: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    """Survey-weighted area under the ROC curve."""
    return float(roc_auc_score(y_true, scores, sample_weight=weights))


def sensitivity_at_specificity(
    y_true: np.ndarray, scores: np.ndarray, weights: np.ndarray, specificity: float = 0.90
) -> tuple[float, float]:
    """Sensitivity achievable at a given specificity, and the threshold for it."""
    fpr, tpr, thresholds = roc_curve(y_true, scores, sample_weight=weights)
    target_fpr = 1.0 - specificity

    usable = fpr <= target_fpr
    if not usable.any():
        return float("nan"), float("nan")
    index = int(np.argmax(tpr * usable))
    return float(tpr[index]), float(thresholds[index])


def specificity_at_sensitivity(
    y_true: np.ndarray, scores: np.ndarray, weights: np.ndarray, sensitivity: float = 0.90
) -> tuple[float, float]:
    """Specificity achievable at a given sensitivity, and the threshold for it."""
    fpr, tpr, thresholds = roc_curve(y_true, scores, sample_weight=weights)

    usable = tpr >= sensitivity
    if not usable.any():
        return float("nan"), float("nan")
    index = int(np.argmin(np.where(usable, fpr, np.inf)))
    return float(1.0 - fpr[index]), float(thresholds[index])


def calibration(
    y_true: np.ndarray, probabilities: np.ndarray, weights: np.ndarray, n_bins: int = 10
) -> dict[str, Any]:
    """Brier score plus a weighted reliability curve.

    The curve is what to look at. A Brier score mixes discrimination and
    calibration together, so a model can improve it by getting better at
    ranking while still being systematically over-confident.
    """
    brier = float(brier_score_loss(y_true, probabilities, sample_weight=weights))

    edges = np.quantile(probabilities, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bin_index = np.digitize(probabilities, edges[1:-1])

    predicted, observed, share = [], [], []
    total_weight = weights.sum()
    for b in range(n_bins):
        mask = bin_index == b
        if not mask.any():
            continue
        bin_weight = weights[mask].sum()
        predicted.append(float(np.average(probabilities[mask], weights=weights[mask])))
        observed.append(float(np.average(y_true[mask], weights=weights[mask])))
        share.append(float(bin_weight / total_weight))

    return {
        "brier": brier,
        "predicted": predicted,
        "observed": observed,
        "weight_share": share,
        "max_abs_gap": float(np.max(np.abs(np.array(predicted) - np.array(observed))))
        if predicted else float("nan"),
    }


def net_benefit(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> float:
    """Net benefit at one threshold probability, per decision curve analysis.

        NB = TP/N - (FP/N) * (p / (1 - p))

    The odds term is the exchange rate between harms: a clinician willing to
    act at a 20% probability is implicitly saying one missed case costs as
    much as four unnecessary interventions. Vickers & Elkin, 2006.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold probability must lie strictly in (0, 1)")

    predicted_positive = probabilities >= threshold
    total = weights.sum()
    if total <= 0:
        return float("nan")

    true_positive = weights[(predicted_positive) & (y_true == 1)].sum()
    false_positive = weights[(predicted_positive) & (y_true == 0)].sum()
    odds = threshold / (1.0 - threshold)
    return float(true_positive / total - (false_positive / total) * odds)


def treat_all_net_benefit(
    y_true: np.ndarray, weights: np.ndarray, threshold: float
) -> float:
    """Net benefit of intervening on everyone, the baseline any model must beat."""
    total = weights.sum()
    prevalence = float(weights[y_true == 1].sum() / total)
    odds = threshold / (1.0 - threshold)
    return prevalence - (1.0 - prevalence) * odds


@dataclass
class Scorecard:
    """Every metric for one model."""

    name: str
    auroc: float
    sensitivity_at_90_spec: float
    specificity_at_90_sens: float
    brier: float
    calibration_gap: float
    n: int

    def row(self) -> str:
        return (
            f"| {self.name} | {self.auroc:.3f} | {self.sensitivity_at_90_spec:.3f} | "
            f"{self.specificity_at_90_sens:.3f} | {self.brier:.4f} | "
            f"{self.calibration_gap:.3f} |"
        )


def score(
    name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> Scorecard:
    """Score one model. ``probabilities`` is needed for calibration and Brier.

    A clinical index such as TyG has no probability scale, only an ordering,
    so calibration is reported as NaN for those rather than invented by
    rescaling the index into a fake probability.
    """
    sensitivity, _ = sensitivity_at_specificity(y_true, scores, weights)
    specificity, _ = specificity_at_sensitivity(y_true, scores, weights)

    if probabilities is None:
        brier, gap = float("nan"), float("nan")
    else:
        calibrated = calibration(y_true, probabilities, weights)
        brier, gap = calibrated["brier"], calibrated["max_abs_gap"]

    return Scorecard(
        name=name,
        auroc=weighted_auroc(y_true, scores, weights),
        sensitivity_at_90_spec=sensitivity,
        specificity_at_90_sens=specificity,
        brier=brier,
        calibration_gap=gap,
        n=int(len(y_true)),
    )
