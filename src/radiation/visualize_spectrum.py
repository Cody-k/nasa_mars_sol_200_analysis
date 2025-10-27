"""LET spectrum visualization | Publication-quality radiation physics plots"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from .let_spectrum import LETSpectrumAnalyzer


class LETVisualizer:
    """Generate publication-quality LET spectrum visualizations"""

    def __init__(self, analyzer: LETSpectrumAnalyzer, output_dir: Path = Path("output")):
        self.analyzer = analyzer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_let_spectrum(self, save: bool = True) -> plt.Figure:
        """Plot LET spectrum with particle species regions annotated"""
        if not self.analyzer.let_data:
            self.analyzer.parse_let_histograms()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        for ax, region in zip([ax1, ax2], ["A1", "A2"]):
            bins = self.analyzer.let_data[region]
            energies = np.array([b.energy for b in bins])
            counts = np.array([b.count for b in bins])

            ax.semilogy(energies, counts, "-o", linewidth=2, markersize=4, color="#2E86AB")
            ax.axvspan(0.15, 0.30, alpha=0.2, color="#4CAF50", label="Protons (min. ionizing)")
            ax.axvspan(1.5, 15.0, alpha=0.2, color="#FFA726", label="Alphas")
            ax.axvspan(15.0, max(energies), alpha=0.2, color="#EF5350", label="Heavy Ions")

            peaks = self.analyzer.identify_particle_peaks(region)
            for species, (energy, count) in peaks.items():
                ax.plot(energy, count, "r*", markersize=15, label=f"{species}")

            ax.set_xlabel("LET (keV/μm in silicon)", fontsize=11)
            ax.set_ylabel("Particle Count", fontsize=11)
            ax.set_title(f"Sol 200 LET Spectrum - Detector Region {region}", fontsize=12, weight="bold")
            ax.grid(True, alpha=0.3, which="both")
            ax.legend(loc="upper right", fontsize=9)

        plt.tight_layout()

        if save:
            output_file = self.output_dir / "let_spectrum_analysis.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()
            return None

        return fig

    def plot_particle_composition(self, save: bool = True) -> plt.Figure:
        """Plot particle flux composition pie chart"""
        flux = self.analyzer.calculate_particle_flux("A1")

        fig, ax = plt.subplots(figsize=(8, 8))

        sizes = [flux.protons, flux.alphas, flux.heavy_ions]
        labels = [
            f"Protons\n({flux.proton_fraction:.1%})",
            f"Alphas\n({flux.alphas/flux.total:.1%})",
            f"Heavy Ions\n({flux.heavy_ions/flux.total:.1%})",
        ]
        colors = ["#4CAF50", "#FFA726", "#EF5350"]

        ax.pie(sizes, labels=labels, colors=colors, autopct="", startangle=90)
        ax.set_title("Sol 200 Particle Composition (by LET)", fontsize=14, weight="bold", pad=20)

        if save:
            output_file = self.output_dir / "particle_composition.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()
            return None

        return fig

    def plot_dose_comparison(self, mars_dose_msv: float, save: bool = True) -> plt.Figure:
        """Compare Mars radiation with Earth environments"""
        environments = {
            "Earth\n(Sea Level)": 3.0,
            "Earth\n(Denver)": 5.0,
            "Commercial\nFlight": 20.0,
            "ISS": 150.0,
            "Mars Surface\n(Sol 200 × 668 sols)": mars_dose_msv,
        }

        fig, ax = plt.subplots(figsize=(10, 6))

        names = list(environments.keys())
        doses = list(environments.values())
        colors = ["#4CAF50", "#8BC34A", "#FFC107", "#FF9800", "#EF5350"]

        bars = ax.bar(names, doses, color=colors, edgecolor="black", linewidth=1.2)

        ax.set_ylabel("Annual Dose Equivalent (mSv/year)", fontsize=11)
        ax.set_title("Radiation Environment Comparison", fontsize=14, weight="bold")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3, which="both")

        ax.axhline(50, color="orange", linestyle="--", linewidth=1.5, label="NASA Annual Limit (50 mSv)")
        ax.axhline(20, color="red", linestyle="--", linewidth=1.5, label="Public Exposure Limit (20 mSv)")
        ax.legend(loc="upper left")

        if save:
            output_file = self.output_dir / "dose_comparison.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()
            return None

        return fig
