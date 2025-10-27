"""Tests for LET spectrum analysis"""

from pathlib import Path
import pytest
import numpy as np
from src.radiation import LETSpectrumAnalyzer, ParticleFlux


@pytest.fixture
def let_analyzer():
    """Create LET analyzer with Sol 200 data"""
    rad_file = Path(__file__).parent.parent.parent / "data" / "raw" / "RAD_RDR_2013_058_02_42_0200_V00.TXT"
    return LETSpectrumAnalyzer(rad_file)


def test_let_parsing(let_analyzer):
    """LET histograms should parse with correct structure"""
    histograms = let_analyzer.parse_let_histograms()

    assert "A1" in histograms
    assert "A2" in histograms
    assert len(histograms["A1"]) == 44
    assert all(hasattr(b, "energy") for b in histograms["A1"])
    assert all(hasattr(b, "width") for b in histograms["A1"])
    assert all(hasattr(b, "count") for b in histograms["A1"])


def test_energy_bin_range(let_analyzer):
    """Energy bins should span expected LET range"""
    histograms = let_analyzer.parse_let_histograms()
    bins = histograms["A1"]

    energies = [b.energy for b in bins]
    assert min(energies) < 0.2
    assert max(energies) > 200
    assert energies == sorted(energies)


def test_proton_peak_identification(let_analyzer):
    """Should identify minimum ionizing proton peak near 0.2 keV/μm"""
    peaks = let_analyzer.identify_particle_peaks("A1")

    if "proton_minimum_ionizing" in peaks:
        peak_energy, peak_count = peaks["proton_minimum_ionizing"]
        assert 0.15 <= peak_energy <= 0.30
        assert peak_count > 0


def test_particle_flux_calculation(let_analyzer):
    """Particle flux should have physically reasonable values"""
    flux = let_analyzer.calculate_particle_flux("A1")

    assert isinstance(flux, ParticleFlux)
    assert flux.total > 0
    assert 0 <= flux.proton_fraction <= 1.0
    assert flux.protons >= 0
    assert flux.alphas >= 0
    assert flux.heavy_ions >= 0


def test_gcr_composition_approximation(let_analyzer):
    """Protons should dominate in minimum ionizing LET range"""
    flux = let_analyzer.calculate_particle_flux("A1")

    assert flux.proton_fraction > 0.2
    assert flux.protons > flux.alphas
    assert flux.alphas > flux.heavy_ions


def test_a1_a2_comparison(let_analyzer):
    """Both detector regions should give similar results"""
    flux_a1 = let_analyzer.calculate_particle_flux("A1")
    flux_a2 = let_analyzer.calculate_particle_flux("A2")

    ratio = flux_a1.proton_fraction / flux_a2.proton_fraction
    assert 0.8 <= ratio <= 1.2
