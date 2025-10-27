"""Tests for RAD data parser"""

from pathlib import Path
import pytest
from src.rad_parser import RADDataParser


@pytest.fixture
def rad_file():
    """Path to RAD data file"""
    return Path(__file__).parent.parent / "data" / "raw" / "RAD_RDR_2013_058_02_42_0200_V00.TXT"


def test_parser_initialization(rad_file):
    """Parser should initialize with data file"""
    parser = RADDataParser(rad_file)
    assert parser.data_file.exists()


def test_header_parsing(rad_file):
    """Header should contain file metadata"""
    parser = RADDataParser(rad_file)
    header = parser.parse_header()

    assert "SOFTWARE_VERSION" in header
    assert "START_SOL" in header
    assert header["START_SOL"] == "0200"


def test_observation_parsing(rad_file):
    """Observations should parse into records"""
    parser = RADDataParser(rad_file)
    observations = list(parser.parse_observations())

    assert len(observations) > 0
    first_obs = observations[0]
    assert "obs_id" in first_obs
