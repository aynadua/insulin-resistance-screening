"""Clinical indices, checked against published formulas and unit identities."""

import numpy as np
import pytest

from irisk.indices import (
    HOMA_IR_THRESHOLDS,
    MG_DL_PER_MMOL_L,
    classify,
    glucose_to_mg_dl,
    homa_ir,
    quicki,
    tg_hdl_ratio,
    tyg_index,
)


def test_homa_ir_matches_the_published_formula():
    # 405 is the mg/dL constant from Matthews 1985.
    assert homa_ir(90.0, 6.0) == pytest.approx(90.0 * 6.0 / 405.0)


def test_homa_ir_agrees_across_units():
    """The same patient must score the same whichever unit is used."""
    mg_dl = 90.0
    mmol_l = mg_dl / MG_DL_PER_MMOL_L
    assert homa_ir(mg_dl, 6.0, "mg/dL") == pytest.approx(
        homa_ir(mmol_l, 6.0, "mmol/L")
    )


def test_the_mmol_constant_follows_from_the_mg_dl_one():
    """405 / 18.0156 = 22.5, the constant the mmol/L form of HOMA-IR uses."""
    assert 405.0 / MG_DL_PER_MMOL_L == pytest.approx(22.5, abs=0.03)


def test_unknown_glucose_unit_is_rejected():
    with pytest.raises(ValueError, match="unknown glucose unit"):
        glucose_to_mg_dl(90.0, "mg/L")


def test_quicki_runs_opposite_to_homa_ir():
    """QUICKI is a sensitivity index, so a more resistant patient scores lower."""
    sensitive = (85.0, 4.0)
    resistant = (115.0, 25.0)
    assert homa_ir(*resistant) > homa_ir(*sensitive)
    assert quicki(*resistant) < quicki(*sensitive)


def test_tyg_matches_its_definition():
    glucose, triglycerides = 100.0, 150.0
    assert tyg_index(glucose, triglycerides) == pytest.approx(
        np.log(triglycerides * glucose / 2.0)
    )


def test_tyg_needs_no_insulin():
    """The whole point of TyG: it is computable from a lipid panel alone."""
    import inspect

    assert "insulin" not in inspect.signature(tyg_index).parameters


def test_tg_hdl_handles_a_zero_denominator():
    assert np.isnan(tg_hdl_ratio(150.0, 0.0))


def test_indices_are_vectorised():
    glucose = np.array([90.0, 110.0, 130.0])
    insulin = np.array([5.0, 12.0, 30.0])
    values = homa_ir(glucose, insulin)
    assert values.shape == (3,)
    assert np.all(np.diff(values) > 0)


def test_classify_treats_missing_as_not_resistant():
    """A missing lab result is not evidence of disease."""
    assert not classify(np.nan, 2.5)
    assert classify(3.0, 2.5)
    assert not classify(2.0, 2.5)


def test_thresholds_carry_their_population():
    """A HOMA-IR cut-off without its population is not a usable number."""
    assert len(HOMA_IR_THRESHOLDS) >= 2
    for threshold in HOMA_IR_THRESHOLDS:
        assert threshold.population and threshold.source
    spread = max(t.value for t in HOMA_IR_THRESHOLDS) - min(
        t.value for t in HOMA_IR_THRESHOLDS
    )
    assert spread > 1.0, "published thresholds really do disagree by this much"
