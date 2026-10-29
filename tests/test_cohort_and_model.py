"""Cohort construction, and the leakage guard that matters most here."""

import numpy as np
import pandas as pd
import pytest

from irisk.cohort import (
    PREDICTORS,
    build_cohort,
    weighted_prevalence,
    weighted_quantile,
)
from irisk.model import clinical_index_scores, feature_matrix


def _frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "SEQN": np.arange(n),
        "age": rng.integers(18, 80, n).astype(float),
        "sex": rng.integers(1, 3, n).astype(float),
        "pregnant": np.where(rng.random(n) < 0.03, 1.0, 2.0),
        "DIQ010": np.where(rng.random(n) < 0.1, 1.0, 2.0),
        "DIQ050": np.where(rng.random(n) < 0.02, 1.0, 2.0),
        "glucose_mg_dl": rng.normal(100, 12, n),
        "insulin_uU_mL": rng.lognormal(2.2, 0.5, n),
        "triglycerides_mg_dl": rng.lognormal(4.7, 0.5, n),
        "hdl_mg_dl": rng.normal(53, 14, n),
        "bmi": rng.normal(28, 6, n),
        "waist_cm": rng.normal(98, 16, n),
        "height_cm": rng.normal(168, 10, n),
        "weight_kg": rng.normal(80, 18, n),
        "hba1c_pct": rng.normal(5.5, 0.5, n),
        "systolic_bp": rng.normal(122, 15, n),
        "diastolic_bp": rng.normal(72, 10, n),
        "alt_u_l": rng.lognormal(3.0, 0.4, n),
        "weight": rng.uniform(1000, 40000, n),
        "cycle": "2017-2018",
    })


def test_insulin_is_never_a_predictor():
    """The defect that would invalidate the whole project.

    The label is derived from insulin. A model given insulin as an input is
    inverting a formula, would score near-perfectly, and would be useless for
    the clinical purpose, which is estimating resistance for the patients who
    will never have an insulin assay ordered.
    """
    assert "insulin_uU_mL" not in PREDICTORS
    assert "homa_ir" not in PREDICTORS

    cohort, _ = build_cohort(_frame())
    matrix = feature_matrix(cohort)
    assert "insulin_uU_mL" not in matrix.columns
    assert not any("insulin" in c.lower() for c in matrix.columns)
    assert not any("homa" in c.lower() for c in matrix.columns)


def test_cohort_applies_every_exclusion():
    cohort, report = build_cohort(_frame())
    labels = [label for label, _ in report.steps]
    assert any("pregnant" in label for label in labels)
    assert any("diabetes" in label for label in labels)
    assert any("insulin" in label for label in labels)
    assert report.final == len(cohort)


def test_cohort_counts_never_increase():
    _, report = build_cohort(_frame())
    counts = [n for _, n in report.steps]
    assert counts == sorted(counts, reverse=True)


def test_excluded_groups_are_actually_gone():
    cohort, _ = build_cohort(_frame())
    assert (cohort["age"] >= 20).all()
    assert (cohort["pregnant"] != 1).all()
    assert (cohort["DIQ010"] != 1).all()
    assert (cohort["DIQ050"] != 1).all()
    assert (cohort["weight"] > 0).all()


def test_homa_ir_is_attached():
    cohort, _ = build_cohort(_frame())
    expected = cohort["insulin_uU_mL"] * cohort["glucose_mg_dl"] / 405.0
    assert np.allclose(cohort["homa_ir"], expected)


def test_weighted_prevalence_differs_from_unweighted():
    """If they matched, the weights would not be doing anything."""
    labels = np.array([1, 1, 0, 0])
    weights = np.array([1.0, 1.0, 9.0, 9.0])
    assert weighted_prevalence(labels, weights) == pytest.approx(0.1)
    assert labels.mean() == pytest.approx(0.5)


def test_weighted_quantile_follows_the_weights():
    values = np.array([1.0, 2.0, 3.0, 100.0])
    light = np.array([1.0, 1.0, 1.0, 0.001])
    heavy = np.array([0.001, 0.001, 0.001, 1.0])
    assert weighted_quantile(values, 0.5, light) < 5.0
    assert weighted_quantile(values, 0.5, heavy) > 50.0


def test_clinical_indices_need_no_insulin():
    """Every baseline must be computable from a routine panel."""
    cohort, _ = build_cohort(_frame())
    without_insulin = cohort.drop(columns=["insulin_uU_mL"])
    scores = clinical_index_scores(without_insulin)
    assert set(scores) >= {"TyG index", "TG/HDL ratio", "BMI alone", "Waist alone"}
    for values in scores.values():
        assert np.isfinite(np.asarray(values, dtype=float)).any()
