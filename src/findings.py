"""Analysis findings | Interpret Sol 200 data and generate insights"""

from typing import Any
from .analysis import Sol200Analysis


class Sol200Findings:
    """Generate findings and insights from Sol 200 data"""

    def __init__(self, analysis: Sol200Analysis):
        self.analysis = analysis

    def detector_analysis(self) -> dict[str, Any]:
        """Analyze detector configuration and measurements"""
        stats = self.analysis.detector_statistics()

        findings = {
            "detector_b": {
                "thickness_um": stats["detector_b_thickness_um"],
                "mass_g_cm2": stats["detector_b_mass_g_cm2"],
                "purpose": "Measures low-energy charged particles",
                "interpretation": f"Silicon detector at {stats['detector_b_thickness_um']} μm thickness",
            },
            "detector_e": {
                "mass_mg_cm2": stats["detector_e_mass_mg_cm2"],
                "purpose": "Measures high-energy particles and radiation dose",
                "mass_ratio": stats["detector_e_mass_mg_cm2"]
                / (stats["detector_b_mass_g_cm2"] * 1000),
            },
        }

        return findings

    def mission_context(self) -> dict[str, str]:
        """Provide Sol 200 mission context"""
        metadata = self.analysis.get_metadata()

        return {
            "sol": metadata["START_SOL"],
            "earth_date": "2013-058/059 (February 27-28, 2013)",
            "mission_day": "Sol 200 of Mars Science Laboratory mission",
            "location": "Gale Crater, Mars",
            "rover": "Curiosity",
            "instrument": "RAD (Radiation Assessment Detector)",
            "significance": "Early mission characterization of Mars radiation environment",
        }

    def summary_report(self) -> str:
        """Generate summary report of findings"""
        detector = self.detector_analysis()
        context = self.mission_context()
        obs = self.analysis.observation_summary()

        report = f"""
NASA Mars Curiosity Rover - Sol 200 Analysis Summary

Mission Context:
  Sol: {context['sol']}
  Earth Date: {context['earth_date']}
  Location: {context['location']}
  Instrument: {context['instrument']}

Detector Configuration:
  Detector B (Low-energy particles):
    - Thickness: {detector['detector_b']['thickness_um']} μm
    - Mass: {detector['detector_b']['mass_g_cm2']} g/cm²

  Detector E (High-energy particles):
    - Mass: {detector['detector_e']['mass_mg_cm2']} mg/cm²
    - Mass Ratio to B: {detector['detector_e']['mass_ratio']:.2f}x

Observations:
  Count: {obs['observation_count']} RAD measurements

Significance:
  {context['significance']}
""".strip()

        return report
