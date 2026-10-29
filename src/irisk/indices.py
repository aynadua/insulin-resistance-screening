"""Clinical surrogate indices for insulin resistance.

Insulin resistance is properly measured with a hyperinsulinaemic-euglycaemic
clamp: infuse insulin, infuse glucose to hold blood sugar steady, and read off
how much glucose it took. It is the reference standard and it is also a
multi-hour procedure with two intravenous lines, so it is used in research and
essentially never in clinic.

Everything below is a surrogate: an arithmetic combination of a fasting blood
draw that correlates with the clamp well enough to be useful at population
scale. They are the baselines any model in this repo has to beat, and they are
not weak. HOMA-IR has been validated against the clamp in thousands of
subjects across decades.

UNITS ARE THE MAIN HAZARD HERE. The same index has different constants
depending on whether glucose is reported in mg/dL (United States) or mmol/L
(most of the rest of the world), and the two differ by a factor of 18.
Substituting one for the other does not raise an error, it silently moves
HOMA-IR by 18x. Every function below takes an explicit unit and converts, and
there is no positional path that lets a caller skip the question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

GlucoseUnit = Literal["mg/dL", "mmol/L"]

# 1 mmol/L of glucose is 18.0156 mg/dL (molar mass 180.156 g/mol).
MG_DL_PER_MMOL_L = 18.0156

ArrayLike = np.ndarray | float


def glucose_to_mg_dl(glucose: ArrayLike, unit: GlucoseUnit) -> np.ndarray:
    """Convert fasting glucose to mg/dL."""
    values = np.asarray(glucose, dtype=float)
    if unit == "mg/dL":
        return values
    if unit == "mmol/L":
        return values * MG_DL_PER_MMOL_L
    raise ValueError(f"unknown glucose unit {unit!r}; use 'mg/dL' or 'mmol/L'")


def homa_ir(
    glucose: ArrayLike, insulin_uU_mL: ArrayLike, unit: GlucoseUnit = "mg/dL"
) -> np.ndarray:
    """Homeostatic model assessment of insulin resistance.

    HOMA-IR = fasting insulin (uU/mL) * fasting glucose (mg/dL) / 405

    The 405 is the mg/dL form of the constant; in mmol/L the same index uses
    22.5, and 405 / 18.0156 = 22.48, which is where that number comes from.

    Higher means more resistant. Matthews et al., Diabetologia, 1985.
    """
    glucose_mg_dl = glucose_to_mg_dl(glucose, unit)
    insulin = np.asarray(insulin_uU_mL, dtype=float)
    return insulin * glucose_mg_dl / 405.0


def quicki(
    glucose: ArrayLike, insulin_uU_mL: ArrayLike, unit: GlucoseUnit = "mg/dL"
) -> np.ndarray:
    """Quantitative insulin sensitivity check index.

    QUICKI = 1 / (log10(insulin uU/mL) + log10(glucose mg/dL))

    A sensitivity index, so unlike HOMA-IR it runs the other way: higher means
    more sensitive, less resistant. Taking logs first makes it better behaved
    than HOMA-IR in the obese and diabetic range, where the insulin
    distribution has a long right tail.

    Katz et al., J Clin Endocrinol Metab, 2000.
    """
    glucose_mg_dl = glucose_to_mg_dl(glucose, unit)
    insulin = np.asarray(insulin_uU_mL, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = np.log10(insulin) + np.log10(glucose_mg_dl)
        return np.where(denominator > 0, 1.0 / denominator, np.nan)


def tyg_index(
    glucose: ArrayLike, triglycerides_mg_dL: ArrayLike, unit: GlucoseUnit = "mg/dL"
) -> np.ndarray:
    """Triglyceride-glucose index.

    TyG = ln(fasting triglycerides (mg/dL) * fasting glucose (mg/dL) / 2)

    The interesting one, because it needs no insulin assay at all. Fasting
    insulin is expensive, is not standardised between laboratories, and is
    simply not ordered in routine practice, whereas a lipid panel and a
    fasting glucose are. TyG is therefore the baseline to beat for any model
    whose point is to avoid measuring insulin.

    Simental-Mendia et al., Metab Syndr Relat Disord, 2008.
    """
    glucose_mg_dl = glucose_to_mg_dl(glucose, unit)
    triglycerides = np.asarray(triglycerides_mg_dL, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        product = triglycerides * glucose_mg_dl / 2.0
        return np.where(product > 0, np.log(product), np.nan)


def tg_hdl_ratio(
    triglycerides_mg_dL: ArrayLike, hdl_mg_dL: ArrayLike
) -> np.ndarray:
    """Triglyceride to HDL cholesterol ratio, the cheapest surrogate of all.

    Both numbers appear on any standard lipid panel. Its weakness is that the
    threshold separating resistant from sensitive differs substantially by
    ancestry, notably being lower in Black populations, so a single cut-off
    applied to a mixed cohort misclassifies by group.
    """
    triglycerides = np.asarray(triglycerides_mg_dL, dtype=float)
    hdl = np.asarray(hdl_mg_dL, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(hdl > 0, triglycerides / hdl, np.nan)


@dataclass(frozen=True)
class Threshold:
    """A published cut-off, with the population it was derived in.

    The population matters more than the number. HOMA-IR cut-offs in the
    literature range from about 2.0 to 3.8 depending on the cohort's ancestry,
    age and adiposity, because the index has no absolute scale: it is
    calibrated against whatever distribution it was fitted in. Quoting "the"
    HOMA-IR threshold without saying whose is how a model trained in one
    population silently misclassifies another.
    """

    value: float
    population: str
    source: str


# Not an exhaustive list, and deliberately plural to make the spread visible.
HOMA_IR_THRESHOLDS: tuple[Threshold, ...] = (
    Threshold(2.5, "general adult, widely used convention", "common clinical practice"),
    Threshold(2.73, "US adults, NHANES III non-diabetic", "Esteghamati et al., 2010"),
    Threshold(3.8, "European adults, upper quartile", "Ascaso et al., Diabetes Care 2003"),
)


def classify(values: ArrayLike, threshold: float) -> np.ndarray:
    """Label as insulin resistant at or above a threshold.

    NaN in gives False out rather than an error: a missing lab result is not
    evidence of resistance, and the caller is expected to track missingness
    separately rather than letting it become a negative prediction by accident.
    """
    array = np.asarray(values, dtype=float)
    return np.where(np.isnan(array), False, array >= threshold)
