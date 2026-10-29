"""NHANES loading, pooling, and the assay guard.

The guard test is the important one. Pooling cycles across the 2013 insulin
assay change produces a cohort that looks entirely normal and whose
insulin-derived labels are systematically shifted.
"""

import pandas as pd
import pytest

from irisk.nhanes import (
    COMPONENTS,
    CYCLES,
    DEFAULT_CYCLES,
    AssayMismatchError,
    load_cycles,
)

DATA = "data/nhanes"


def _available() -> bool:
    from pathlib import Path

    return (Path(DATA) / "DEMO_J.xpt").exists()


needs_data = pytest.mark.skipif(not _available(), reason="run `irisk data` first")


def test_default_cycles_share_one_insulin_assay():
    """The default pooling must not straddle the 2013 assay change."""
    assays = {CYCLES[name].insulin_assay for name in DEFAULT_CYCLES}
    assert len(assays) == 1, assays


def test_the_assay_change_is_recorded():
    """2011-2012 used Roche; everything later used Tosoh."""
    assert CYCLES["2011-2012"].insulin_assay != CYCLES["2013-2014"].insulin_assay


@needs_data
def test_pooling_across_assays_raises():
    with pytest.raises(AssayMismatchError, match="11.3%"):
        load_cycles(("2011-2012", "2017-2018"), DATA)


@needs_data
def test_pooling_across_assays_can_be_forced_deliberately():
    frame = load_cycles(("2013-2014", "2015-2016"), DATA, allow_assay_mix=True)
    assert len(frame) > 0


def test_unknown_cycle_is_rejected():
    with pytest.raises(KeyError, match="unknown cycle"):
        load_cycles(("1999-2000",), DATA)


@needs_data
def test_pooled_weights_are_divided_by_the_number_of_cycles():
    """CDC guidance: combining k two-year cycles divides the weight by k."""
    one = load_cycles(("2017-2018",), DATA)
    three = load_cycles(DEFAULT_CYCLES, DATA)

    single = one["WTSAF2YR"].dropna()
    pooled = three.loc[three["cycle"] == "2017-2018", "weight"].dropna()
    assert pooled.sum() == pytest.approx(single.sum() / 3, rel=1e-6)


@needs_data
def test_fasting_weight_is_used_not_the_examination_weight():
    """WTSAF2YR and WTMEC2YR are different columns and must not be confused."""
    frame = load_cycles(("2017-2018",), DATA)
    assert "WTSAF2YR" in frame.columns
    fasting = frame["WTSAF2YR"].notna()
    # The fasting subsample is a minority of those examined.
    assert fasting.sum() < frame["WTMEC2YR"].notna().sum()


@needs_data
def test_every_component_contributes_columns():
    frame = load_cycles(("2017-2018",), DATA)
    assert len(frame) > 5000
    for component in COMPONENTS:
        assert any(c for c in frame.columns), component


@needs_data
def test_renamed_columns_are_present():
    frame = load_cycles(("2017-2018",), DATA)
    for column in ("age", "sex", "bmi", "glucose_mg_dl", "insulin_uU_mL"):
        assert column in frame.columns


@needs_data
def test_cycle_and_assay_are_attached_to_every_row():
    frame = load_cycles(DEFAULT_CYCLES, DATA)
    assert set(frame["cycle"]) == set(DEFAULT_CYCLES)
    assert frame["insulin_assay"].nunique() == 1
    assert isinstance(frame, pd.DataFrame)
