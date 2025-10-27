"""Advanced radiation analysis | LET spectrum and dose equivalent calculations"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.radiation import LETSpectrumAnalyzer, DoseEquivalentCalculator
from src.radiation.visualize_spectrum import LETVisualizer


def main():
    """Execute advanced radiation physics analysis"""
    print("=== Advanced Sol 200 Radiation Analysis ===\n")

    rad_file = Path("data/raw/RAD_RDR_2013_058_02_42_0200_V00.TXT")

    analyzer = LETSpectrumAnalyzer(rad_file)
    calculator = DoseEquivalentCalculator()
    visualizer = LETVisualizer(analyzer, output_dir=Path("output"))

    print("Parsing LET spectra...")
    histograms = analyzer.parse_let_histograms()
    print(f"✓ Parsed {len(histograms)} detector regions (A1, A2)")
    print(f"  Bins per region: {len(histograms['A1'])}\n")

    print("Identifying particle peaks...")
    peaks_a1 = analyzer.identify_particle_peaks("A1")
    peaks_a2 = analyzer.identify_particle_peaks("A2")

    print(f"Region A1 peaks:")
    for species, (energy, count) in peaks_a1.items():
        print(f"  {species}: {energy:.3f} keV/μm (count: {count:.1f})")

    print(f"\nRegion A2 peaks:")
    for species, (energy, count) in peaks_a2.items():
        print(f"  {species}: {energy:.3f} keV/μm (count: {count:.1f})")

    print("\nCalculating particle flux...")
    flux_a1 = analyzer.calculate_particle_flux("A1")
    flux_a2 = analyzer.calculate_particle_flux("A2")

    print(f"\nRegion A1 flux composition:")
    print(f"  Protons: {flux_a1.protons:.1f} ({flux_a1.proton_fraction:.1%})")
    print(f"  Alphas: {flux_a1.alphas:.1f} ({flux_a1.alphas/flux_a1.total:.1%})")
    print(f"  Heavy Ions: {flux_a1.heavy_ions:.1f} ({flux_a1.heavy_ions/flux_a1.total:.1%})")
    print(f"  Total: {flux_a1.total:.1f}")

    print(f"\nRegion A2 flux composition:")
    print(f"  Protons: {flux_a2.protons:.1f} ({flux_a2.proton_fraction:.1%})")
    print(f"  Alphas: {flux_a2.alphas:.1f} ({flux_a2.alphas/flux_a2.total:.1%})")
    print(f"  Heavy Ions: {flux_a2.heavy_ions:.1f} ({flux_a2.heavy_ions/flux_a2.total:.1%})")

    print("\nCalculating dose equivalent...")
    dose_a1 = calculator.calculate_dose_equivalent(analyzer, "A1")
    dose_a2 = calculator.calculate_dose_equivalent(analyzer, "A2")

    print(f"\nRegion A1 dose metrics:")
    print(f"  Absorbed dose (per sol): {dose_a1.absorbed_dose_cgy_per_sol:.4f} cGy")
    print(f"  Dose equivalent (per sol): {dose_a1.dose_equivalent_msv_per_sol:.3f} mSv")
    print(f"  Annual dose equivalent: {dose_a1.annual_dose_equivalent_msv:.1f} mSv/year")
    print(f"  Average quality factor: {dose_a1.average_quality_factor:.2f}")

    print(f"\nRegion A2 dose metrics:")
    print(f"  Absorbed dose (per sol): {dose_a2.absorbed_dose_cgy_per_sol:.4f} cGy")
    print(f"  Dose equivalent (per sol): {dose_a2.dose_equivalent_msv_per_sol:.3f} mSv")
    print(f"  Annual dose equivalent: {dose_a2.annual_dose_equivalent_msv:.1f} mSv/year")
    print(f"  Average quality factor: {dose_a2.average_quality_factor:.2f}")

    print("\nComparing with Earth environments...")
    comparison = calculator.compare_with_earth(dose_a1.annual_dose_equivalent_msv)

    print(f"\n  Mars annual dose (projected): {comparison['mars_annual_msv']:.1f} mSv/year")
    print(f"  Earth sea level: {comparison['earth_sea_level_msv_year']:.1f} mSv/year")
    print(f"  Mars vs Earth: {comparison['mars_vs_earth_sea_level']:.1f}× higher")
    print(f"  Mars vs ISS: {comparison['mars_vs_iss']:.2f}× (ISS comparison)")

    print("\n=== Generating Visualizations ===\n")
    visualizer.plot_let_spectrum()
    print("✓ LET spectrum plot saved")

    visualizer.plot_particle_composition()
    print("✓ Particle composition chart saved")

    visualizer.plot_dose_comparison(comparison["mars_annual_msv"])
    print("✓ Dose comparison chart saved")

    print(f"\n=== Analysis Complete ===")
    print(f"Visualizations: output/")
    print(f"See FINDINGS.md for scientific interpretation")


if __name__ == "__main__":
    main()
