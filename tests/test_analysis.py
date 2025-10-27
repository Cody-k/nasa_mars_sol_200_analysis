"""Tests for analysis module"""

from pathlib import Path
import pytest
from src.analysis import Sol200Analysis


@pytest.fixture
def analysis():
    """Create analysis instance with test data"""
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    return Sol200Analysis(data_dir)


def test_detector_statistics(analysis):
    """Detector parameters should return valid measurements"""
    stats = analysis.detector_statistics()

    assert "detector_b_thickness_um" in stats
    assert "detector_b_mass_g_cm2" in stats
    assert "detector_e_mass_mg_cm2" in stats
    assert stats["detector_b_thickness_um"] > 0
    assert stats["detector_b_mass_g_cm2"] > 0


def test_observation_summary(analysis):
    """Observation summary should return count and timing"""
    summary = analysis.observation_summary()

    assert "observation_count" in summary
    assert isinstance(summary["observation_count"], int)
    assert summary["observation_count"] >= 0


def test_metadata_loading(analysis):
    """Metadata should load with required fields"""
    meta = analysis.get_metadata()

    required_fields = ["START_SOL", "DETECTOR_B_THICKNESS", "DETECTOR_E_MASS"]
    for field in required_fields:
        assert field in meta
