"""Visualization module | Generate analysis plots and charts"""

from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from .analysis import Sol200Analysis


class Sol200Visualizer:
    """Generate visualizations for Sol 200 analysis"""

    def __init__(self, analysis: Sol200Analysis, output_dir: Path = Path("output")):
        self.analysis = analysis
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def detector_comparison(self) -> Figure:
        """Compare detector B and E parameters"""
        stats = self.analysis.detector_statistics()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.bar(
            ["Thickness (μm)"],
            [stats["detector_b_thickness_um"]],
            color="#2E86AB",
        )
        ax1.set_ylabel("Micrometers (μm)")
        ax1.set_title("Detector B Thickness")
        ax1.grid(axis="y", alpha=0.3)

        masses = [stats["detector_b_mass_g_cm2"], stats["detector_e_mass_mg_cm2"]]
        ax2.bar(["Detector B (g/cm²)", "Detector E (mg/cm²)"], masses, color=["#2E86AB", "#A23B72"])
        ax2.set_ylabel("Mass")
        ax2.set_title("Detector Mass Comparison")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / "detector_comparison.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        return fig

    def observation_timeline(self) -> Figure:
        """Plot observation timing if data available"""
        df = self.analysis.parse_rad_observations()

        if df.is_empty():
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        if "START_OBS_UTC" in df.columns:
            obs_times = df["START_OBS_UTC"].to_list()
            ax.plot(range(len(obs_times)), [1] * len(obs_times), "o", color="#F18F01", markersize=8)
            ax.set_xlabel("Observation Index")
            ax.set_ylabel("Observation Event")
            ax.set_title(f"Sol 200 RAD Observations (n={len(obs_times)})")
            ax.grid(alpha=0.3)

            output_file = self.output_dir / "observation_timeline.png"
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()

        return fig

    def generate_all(self) -> None:
        """Generate all visualizations"""
        self.detector_comparison()
        self.observation_timeline()
        print(f"Visualizations saved to {self.output_dir}/")
