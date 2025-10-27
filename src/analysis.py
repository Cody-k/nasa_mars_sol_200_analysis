"""Main analysis module | Curiosity Rover Sol 200 data analysis"""

from pathlib import Path
from typing import Any
import polars as pl
from .rad_parser import RADDataParser


class Sol200Analysis:
    """Sol 200 mission data analysis"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.rad_file = self.data_dir / "RAD_RDR_2013_058_02_42_0200_V00.TXT"

    def get_metadata(self) -> dict[str, Any]:
        """Load Sol 200 metadata from JSON"""
        import json
        metadata_file = self.data_dir.parent.parent / "metadata.json"
        with metadata_file.open() as f:
            return json.load(f)

    def parse_rad_observations(self) -> pl.DataFrame:
        """Parse RAD detector observations into DataFrame"""
        parser = RADDataParser(self.rad_file)
        observations = list(parser.parse_observations())

        if not observations:
            return pl.DataFrame()

        return pl.DataFrame(observations)

    def detector_statistics(self) -> dict[str, float]:
        """Calculate detector parameter statistics"""
        metadata = self.get_metadata()

        return {
            "detector_b_thickness_um": float(metadata["DETECTOR_B_THICKNESS"]),
            "detector_b_mass_g_cm2": float(metadata["DETECTOR_B_MASS"]),
            "detector_e_mass_mg_cm2": float(metadata["DETECTOR_E_MASS"]),
        }

    def observation_summary(self) -> dict[str, Any]:
        """Summarize observation timing and count"""
        df = self.parse_rad_observations()

        if df.is_empty():
            return {"observation_count": 0}

        return {
            "observation_count": len(df),
            "start_sol": df["START_OBS_MARS"][0] if "START_OBS_MARS" in df.columns else None,
            "start_utc": df["START_OBS_UTC"][0] if "START_OBS_UTC" in df.columns else None,
        }
