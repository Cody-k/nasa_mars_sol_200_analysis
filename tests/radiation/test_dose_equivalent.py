"""Tests for dose equivalent calculations"""

import pytest
from src.radiation import DoseEquivalentCalculator, LETSpectrumAnalyzer
from pathlib import Path


@pytest.fixture
def calculator():
    """Create dose calculator"""
    return DoseEquivalentCalculator()


@pytest.fixture
def let_analyzer():
    """Create LET analyzer"""
    rad_file = (
        Path(__file__).parent.parent.parent
        / "data"
        / "raw"
        / "RAD_RDR_2013_058_02_42_0200_V00.TXT"
    )
    return LETSpectrumAnalyzer(rad_file)


def test_quality_factor_low_let(calculator):
    """Low LET should have quality factor of 1"""
    q = calculator.quality_factor(0.5)
    assert q == 1.0


def test_quality_factor_medium_let(calculator):
    """Medium LET should increase quality factor"""
    q_low = calculator.quality_factor(10)
    q_high = calculator.quality_factor(50)
    assert q_high > q_low


def test_quality_factor_high_let(calculator):
    """High LET should use sqrt formula"""
    q = calculator.quality_factor(150)
    expected = 300.0 / np.sqrt(150)
    assert abs(q - expected) < 0.01


def test_dose_calculation(calculator, let_analyzer):
    """Dose equivalent should be calculated correctly"""
    dose = calculator.calculate_dose_equivalent(let_analyzer, "A1")

    assert dose.absorbed_dose_cgy_per_sol > 0
    assert dose.dose_equivalent_msv_per_sol > 0
    assert dose.annual_dose_equivalent_msv > 0
    assert dose.average_quality_factor >= 1.0
    assert 150 <= dose.annual_dose_equivalent_msv <= 350


def test_earth_comparison(calculator):
    """Earth comparison should return reasonable ratios"""
    comparison = calculator.compare_with_earth(mars_annual_msv=200.0)

    assert comparison["mars_annual_msv"] == 200.0
    assert comparison["mars_vs_earth_sea_level"] > 10
    assert comparison["mars_vs_iss"] > 0


import numpy as np
