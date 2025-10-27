"""LET spectrum analysis | Particle species identification from energy deposition"""

from dataclasses import dataclass
from pathlib import Path
import polars as pl
import numpy as np
from scipy import signal, stats
from typing import Optional


@dataclass
class LETBin:
    """LET histogram bin data"""

    energy: float
    width: float
    count: float


@dataclass
class ParticleFlux:
    """Particle flux estimate by species"""

    protons: float
    alphas: float
    heavy_ions: float
    total: float
    proton_fraction: float


class LETSpectrumAnalyzer:
    """Analyze Linear Energy Transfer spectra to identify particle species"""

    PROTON_PEAK_RANGE = (0.15, 0.30)
    ALPHA_RANGE = (1.5, 15.0)
    HEAVY_ION_THRESHOLD = 15.0

    def __init__(self, rad_file: Path):
        self.rad_file = Path(rad_file)
        self.let_data: Optional[dict[str, list[LETBin]]] = None

    def parse_let_histograms(self) -> dict[str, list[LETBin]]:
        """Extract and aggregate LET histogram data for both detector regions"""
        with self.rad_file.open() as f:
            content = f.read()

        histograms = {}

        for region in ["A1", "A2"]:
            all_bins = []

            for obs_num in range(44):
                section_start = f"[DOSIMETRY_LET_B_{region}: {obs_num:02d}]"
                if section_start not in content:
                    continue

                lines = self._extract_section(content, section_start)
                if len(lines) >= 3:
                    energies = self._parse_number_line(lines[0])
                    widths = self._parse_number_line(lines[1])
                    counts = self._parse_number_line(lines[2])

                    if len(energies) == len(widths) == len(counts) and len(energies) > 0:
                        if not all_bins:
                            all_bins = [
                                LETBin(energy=e, width=w, count=c)
                                for e, w, c in zip(energies, widths, counts)
                            ]
                        else:
                            for i, c in enumerate(counts):
                                if i < len(all_bins):
                                    all_bins[i] = LETBin(
                                        energy=all_bins[i].energy,
                                        width=all_bins[i].width,
                                        count=all_bins[i].count + c,
                                    )

            if all_bins:
                histograms[region] = all_bins

        self.let_data = histograms
        return histograms

    def identify_particle_peaks(self, region: str = "A1") -> dict[str, tuple[float, float]]:
        """Identify characteristic peaks for different particle species"""
        if not self.let_data:
            self.parse_let_histograms()

        if region not in self.let_data:
            return {}

        bins = self.let_data[region]
        energies = np.array([b.energy for b in bins])
        counts = np.array([b.count for b in bins])

        peaks, properties = signal.find_peaks(
            counts, prominence=5.0, width=1, distance=5
        )

        identified_peaks = {}

        for peak_idx in peaks:
            peak_energy = energies[peak_idx]

            if self.PROTON_PEAK_RANGE[0] <= peak_energy <= self.PROTON_PEAK_RANGE[1]:
                identified_peaks["proton_minimum_ionizing"] = (
                    peak_energy,
                    counts[peak_idx],
                )
            elif self.ALPHA_RANGE[0] <= peak_energy <= self.ALPHA_RANGE[1]:
                identified_peaks["alpha_peak"] = (peak_energy, counts[peak_idx])
            elif peak_energy >= self.HEAVY_ION_THRESHOLD:
                identified_peaks["heavy_ion_contribution"] = (peak_energy, counts[peak_idx])

        return identified_peaks

    def calculate_particle_flux(self, region: str = "A1") -> ParticleFlux:
        """Estimate particle flux by species from LET spectrum"""
        if not self.let_data:
            self.parse_let_histograms()

        bins = self.let_data[region]
        energies = np.array([b.energy for b in bins])
        counts = np.array([b.count for b in bins])

        proton_mask = (energies >= self.PROTON_PEAK_RANGE[0]) & (
            energies <= self.PROTON_PEAK_RANGE[1]
        )
        alpha_mask = (energies >= self.ALPHA_RANGE[0]) & (energies <= self.ALPHA_RANGE[1])
        heavy_mask = energies >= self.HEAVY_ION_THRESHOLD

        proton_flux = counts[proton_mask].sum()
        alpha_flux = counts[alpha_mask].sum()
        heavy_flux = counts[heavy_mask].sum()
        total_flux = counts.sum()

        return ParticleFlux(
            protons=proton_flux,
            alphas=alpha_flux,
            heavy_ions=heavy_flux,
            total=total_flux,
            proton_fraction=proton_flux / total_flux if total_flux > 0 else 0,
        )

    def _extract_section(self, content: str, section_marker: str) -> list[str]:
        """Extract data lines from a section"""
        lines = []
        in_section = False

        for line in content.split("\n"):
            if section_marker in line:
                in_section = True
                continue
            if in_section:
                stripped = line.strip()
                if stripped.startswith("["):
                    break
                if stripped and not stripped.startswith("---"):
                    lines.append(stripped)

        return lines

    def _parse_number_line(self, line: str) -> list[float]:
        """Parse space-separated numbers from line"""
        numbers = []
        parts = line.split()

        for part in parts:
            try:
                numbers.append(float(part))
            except ValueError:
                continue

        return numbers
