"""Complete analysis example | Full Sol 200 analysis workflow"""

from pathlib import Path
from src import Sol200Analysis, Sol200Visualizer, Sol200Findings


def main():
    """Run complete Sol 200 analysis"""
    data_dir = Path("data/raw")

    print("=== NASA Mars Sol 200 Analysis ===\n")

    analysis = Sol200Analysis(data_dir)
    findings = Sol200Findings(analysis)
    viz = Sol200Visualizer(analysis, output_dir=Path("output"))

    print(findings.summary_report())

    print("\n=== Generating Visualizations ===\n")
    viz.generate_all()

    print("\n=== Analysis Complete ===")
    print(f"Visualizations: output/")
    print(f"Run 'sol200 analyze' for JSON output")


if __name__ == "__main__":
    main()
