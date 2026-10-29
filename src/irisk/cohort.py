"""Turning pooled NHANES records into an analysis cohort.

Every exclusion is a clinical judgement, so each one is named, counted and
reported rather than buried in a chain of boolean masks. The resulting
:class:`CohortReport` is the equivalent of the participant flow diagram a
clinical paper is expected to publish.

Why each exclusion exists:

Adults only. HOMA-IR thresholds in children differ substantially and track
pubertal stage, so a single cut-off applied across ages misclassifies
adolescents.

Not pregnant. Pregnancy induces physiological insulin resistance by design,
to divert glucose to the fetus. Those women are not pathologically resistant
and including them conflates two different things.

No diagnosed diabetes, and not taking insulin. Both matter and for different
reasons. Someone already diagnosed does not need screening, which is what
this model is for. Someone injecting insulin has a fasting insulin
measurement that reflects their last injection rather than their pancreas, so
HOMA-IR is not merely noisy for them, it is meaningless.

Complete predictors. Requiring the full set keeps the comparison against the
clinical indices honest, since those indices cannot be computed on a partial
lipid panel either.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from irisk.indices import homa_ir

MIN_AGE = 20

# Predictors a model may use. Fasting insulin is deliberately absent: the
# point is to estimate insulin resistance without an insulin assay, and a
# model given insulin to predict an insulin-derived label has learned
# arithmetic rather than physiology.
PREDICTORS: tuple[str, ...] = (
    "age", "sex", "bmi", "waist_cm", "height_cm", "weight_kg",
    "triglycerides_mg_dl", "hdl_mg_dl", "glucose_mg_dl", "hba1c_pct",
    "systolic_bp", "diastolic_bp", "alt_u_l",
)

REQUIRED = ("glucose_mg_dl", "insulin_uU_mL", "triglycerides_mg_dl",
            "hdl_mg_dl", "bmi", "waist_cm", "weight")


@dataclass
class CohortReport:
    """How many participants each exclusion removed."""

    steps: list[tuple[str, int]] = field(default_factory=list)

    def record(self, label: str, remaining: int) -> None:
        self.steps.append((label, remaining))

    def describe(self) -> str:
        lines = []
        previous: int | None = None
        for label, remaining in self.steps:
            dropped = "" if previous is None else f"  (-{previous - remaining})"
            lines.append(f"  {remaining:6d}  {label}{dropped}")
            previous = remaining
        return "\n".join(lines)

    @property
    def final(self) -> int:
        return self.steps[-1][1] if self.steps else 0


def build_cohort(
    frame: pd.DataFrame, min_age: int = MIN_AGE
) -> tuple[pd.DataFrame, CohortReport]:
    """Apply the exclusions and attach the HOMA-IR label."""
    report = CohortReport()
    data = frame.copy()
    report.record("pooled NHANES participants", len(data))

    data = data[data["glucose_mg_dl"].notna() & data["insulin_uU_mL"].notna()]
    report.record("fasting subsample with glucose and insulin", len(data))

    data = data[data["age"] >= min_age]
    report.record(f"adults {min_age}+", len(data))

    data = data[data["pregnant"] != 1]
    report.record("not pregnant", len(data))

    data = data[data["DIQ010"] != 1]
    report.record("no diagnosed diabetes", len(data))

    data = data[data["DIQ050"] != 1]
    report.record("not taking insulin", len(data))

    data = data.dropna(subset=[c for c in REQUIRED if c in data.columns])
    report.record("complete on required measurements", len(data))

    data = data[data["weight"] > 0]
    report.record("positive survey weight", len(data))

    data = data.copy()
    data["homa_ir"] = homa_ir(data["glucose_mg_dl"], data["insulin_uU_mL"])
    return data, report


def weighted_prevalence(labels: np.ndarray, weights: np.ndarray) -> float:
    """Survey-weighted proportion, which is the population estimate.

    The unweighted proportion describes the people who happened to be sampled.
    NHANES deliberately oversamples some groups, so the two differ, and only
    the weighted figure refers to the United States population.
    """
    labels = np.asarray(labels, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    return float((labels * weights).sum() / total) if total > 0 else float("nan")


def weighted_quantile(
    values: np.ndarray, quantile: float, weights: np.ndarray
) -> float:
    """Survey-weighted quantile, used for population reference ranges."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values, weights = values[order], weights[order]

    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return float(np.interp(quantile, cumulative, values))
