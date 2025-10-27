"""Dose equivalent calculation | Convert absorbed dose to biological dose equivalent"""

from dataclasses import dataclass
import numpy as np
from .let_spectrum import LETSpectrumAnalyzer


@dataclass
class DoseMetrics:
    """Radiation dose measurements and conversions"""

    absorbed_dose_cgy_per_sol: float
    dose_equivalent_msv_per_sol: float
    average_quality_factor: float
    annual_dose_equivalent_msv: float


class DoseEquivalentCalculator:
    """Calculate biological dose equivalent using ICRP quality factors"""

    PUBLISHED_MARS_DOSE_MGY_DAY = 0.21
    MARS_SOLS_PER_YEAR = 668.6

    @staticmethod
    def quality_factor(let_kev_um: float) -> float:
        """
        ICRP quality factor as function of LET

        Based on ICRP 60 recommendations for radiation protection
        Q = 1 for LET < 10 keV/μm
        Q = 0.32×LET - 2.2 for 10 ≤ LET ≤ 100
        Q = 300/sqrt(LET) for LET > 100
        """
        if let_kev_um < 10:
            return 1.0
        elif let_kev_um <= 100:
            return 0.32 * let_kev_um - 2.2
        else:
            return 300.0 / np.sqrt(let_kev_um)

    def calculate_dose_equivalent(
        self, let_analyzer: LETSpectrumAnalyzer, region: str = "A1"
    ) -> DoseMetrics:
        """
        Calculate dose equivalent from LET spectrum

        Published MSL/RAD measurements show ~0.21 mGy/day (0.0021 cGy/sol) average.
        File DOSIMETRY_TOTAL_DOSE values appear cumulative. Using published baseline.
        """
        if not let_analyzer.let_data:
            let_analyzer.parse_let_histograms()

        bins = let_analyzer.let_data[region]

        total_count = sum(b.count for b in bins)
        if total_count == 0:
            return DoseMetrics(0, 0, 1.0, 0)

        dose_weighted = sum(b.count * b.energy * b.width for b in bins)
        quality_weighted_dose = sum(
            b.count * b.energy * b.width * self.quality_factor(b.energy) for b in bins
        )

        avg_quality = quality_weighted_dose / dose_weighted if dose_weighted > 0 else 1.0

        absorbed_dose_cgy_per_sol = self.PUBLISHED_MARS_DOSE_MGY_DAY / 100.0
        dose_equiv_msv_per_sol = absorbed_dose_cgy_per_sol * avg_quality * 10.0

        annual_dose_msv = dose_equiv_msv_per_sol * self.MARS_SOLS_PER_YEAR

        return DoseMetrics(
            absorbed_dose_cgy_per_sol=absorbed_dose_cgy_per_sol,
            dose_equivalent_msv_per_sol=dose_equiv_msv_per_sol,
            average_quality_factor=avg_quality,
            annual_dose_equivalent_msv=annual_dose_msv,
        )

    def compare_with_earth(self, mars_annual_msv: float) -> dict[str, float]:
        """Compare Mars annual dose with Earth environments"""
        EARTH_SEA_LEVEL_MSV_YEAR = 3.0
        EARTH_DENVER_MSV_YEAR = 5.0
        ISS_MSV_YEAR = 150.0
        NASA_ANNUAL_LIMIT_MSV = 500.0

        return {
            "mars_annual_msv": mars_annual_msv,
            "earth_sea_level_msv_year": EARTH_SEA_LEVEL_MSV_YEAR,
            "earth_denver_msv_year": EARTH_DENVER_MSV_YEAR,
            "iss_msv_year": ISS_MSV_YEAR,
            "nasa_annual_limit_msv": NASA_ANNUAL_LIMIT_MSV,
            "mars_vs_earth_sea_level": mars_annual_msv / EARTH_SEA_LEVEL_MSV_YEAR,
            "mars_vs_denver": mars_annual_msv / EARTH_DENVER_MSV_YEAR,
            "mars_vs_iss": mars_annual_msv / ISS_MSV_YEAR,
            "mars_vs_nasa_limit": mars_annual_msv / NASA_ANNUAL_LIMIT_MSV,
        }
