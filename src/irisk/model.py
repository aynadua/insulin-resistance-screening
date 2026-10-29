"""The screening model, and the clinical indices it has to beat.

Gradient boosting on routinely available measurements: age, sex,
anthropometrics, a lipid panel, fasting glucose, HbA1c, blood pressure and
ALT. Fasting insulin is deliberately excluded. The label is derived from
insulin, so a model allowed to see it would be inverting a formula rather
than learning anything, and the clinical point is to estimate insulin
resistance for the large majority of patients who will never have an insulin
assay ordered.

Validation is by cycle, not at random. Training on 2013-2016 and testing on
2017-2018 asks whether the model still works on a survey wave collected
later, under a different sampling frame, with a population whose obesity
distribution has shifted. A random split over pooled cycles would answer an
easier question and report a better number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from irisk.cohort import PREDICTORS
from irisk.indices import tg_hdl_ratio, tyg_index


def clinical_index_scores(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Scores from the established surrogates, as baselines.

    Each is a published index in its own right, not a weakened stand-in. BMI
    and waist circumference are included because they are free, and a model
    that cannot beat a tape measure is not worth deploying.
    """
    return {
        "TyG index": tyg_index(frame["glucose_mg_dl"], frame["triglycerides_mg_dl"]),
        "TG/HDL ratio": tg_hdl_ratio(frame["triglycerides_mg_dl"], frame["hdl_mg_dl"]),
        "BMI alone": frame["bmi"].to_numpy(dtype=float),
        "Waist alone": frame["waist_cm"].to_numpy(dtype=float),
        "Fasting glucose alone": frame["glucose_mg_dl"].to_numpy(dtype=float),
    }


def build_estimator(seed: int = 7, **overrides: Any) -> HistGradientBoostingClassifier:
    """Boosted trees, sized for a few thousand rows and a dozen features."""
    params: dict[str, Any] = {
        "max_iter": 400,
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 40,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 30,
        "random_state": seed,
    }
    params.update(overrides)
    return HistGradientBoostingClassifier(**params)


@dataclass
class FittedModel:
    """A fitted estimator plus the columns it expects."""

    estimator: HistGradientBoostingClassifier
    features: list[str]
    threshold: float
    trained_on: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(frame[self.features])[:, 1]

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> FittedModel:
        import joblib

        return joblib.load(path)


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Predictors available to the model, with insulin absent by construction."""
    columns = [c for c in PREDICTORS if c in frame.columns]
    assert "insulin_uU_mL" not in columns, "insulin must not be a predictor"
    return frame[columns]


def fit(
    frame: pd.DataFrame, threshold: float, seed: int = 7, use_weights: bool = True
) -> FittedModel:
    """Fit on a cohort, with the label thresholded from HOMA-IR."""
    x = feature_matrix(frame)
    y = (frame["homa_ir"] >= threshold).astype(int).to_numpy()
    weights = frame["weight"].to_numpy() if use_weights else None

    estimator = build_estimator(seed)
    estimator.fit(x, y, sample_weight=weights)

    return FittedModel(
        estimator=estimator,
        features=list(x.columns),
        threshold=threshold,
        trained_on={
            "n": int(len(frame)),
            "cycles": sorted(frame["cycle"].unique().tolist()),
            "prevalence": float(np.average(y, weights=frame["weight"])),
            "n_iterations": int(estimator.n_iter_),
        },
    )


def cross_validated_scores(
    frame: pd.DataFrame, threshold: float, n_splits: int = 5, seed: int = 7
) -> np.ndarray:
    """Out-of-fold probabilities, so every score comes from an unseen row."""
    x = feature_matrix(frame)
    y = (frame["homa_ir"] >= threshold).astype(int).to_numpy()
    weights = frame["weight"].to_numpy()

    out_of_fold = np.zeros(len(frame))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_index, test_index in splitter.split(x, y):
        estimator = build_estimator(seed)
        estimator.fit(x.iloc[train_index], y[train_index], sample_weight=weights[train_index])
        out_of_fold[test_index] = estimator.predict_proba(x.iloc[test_index])[:, 1]

    return out_of_fold


def by_cycle_split(
    frame: pd.DataFrame, test_cycle: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on every other cycle, test on one held-out survey wave."""
    if test_cycle not in set(frame["cycle"]):
        raise KeyError(f"cycle {test_cycle!r} not present")
    return frame[frame["cycle"] != test_cycle], frame[frame["cycle"] == test_cycle]
