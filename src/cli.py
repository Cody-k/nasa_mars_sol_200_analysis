"""CLI interface | Command-line interface for Sol 200 analysis"""

from pathlib import Path
import click
import json
from .analysis import Sol200Analysis


@click.group()
@click.version_option(version="1.0.0")
def main():
    """NASA Curiosity Rover Sol 200 data analysis"""
    pass


@main.command()
@click.option("--data-dir", default="data/raw", help="Data directory path")
def metadata(data_dir: str):
    """Display Sol 200 mission metadata"""
    analysis = Sol200Analysis(Path(data_dir))
    meta = analysis.get_metadata()

    click.echo("\n=== Sol 200 Mission Metadata ===\n")
    click.echo(f"Sol: {meta['START_SOL']}")
    click.echo(f"Start Time (UTC): {meta['START_OBS_UTC'].strip('\"')}")
    click.echo(f"Start Time (Mars): {meta['START_OBS_MARS'].strip('\"')}")
    click.echo(f"\nDetector B Thickness: {meta['DETECTOR_B_THICKNESS']} μm")
    click.echo(f"Detector B Mass: {meta['DETECTOR_B_MASS']} g/cm²")
    click.echo(f"Detector E Mass: {meta['DETECTOR_E_MASS']} mg/cm²")


@main.command()
@click.option("--data-dir", default="data/raw", help="Data directory path")
def observations(data_dir: str):
    """Parse and summarize RAD observations"""
    analysis = Sol200Analysis(Path(data_dir))
    summary = analysis.observation_summary()

    click.echo("\n=== Observation Summary ===\n")
    click.echo(f"Observation Count: {summary['observation_count']}")
    if summary.get("start_sol"):
        click.echo(f"Start Sol: {summary['start_sol']}")
    if summary.get("start_utc"):
        click.echo(f"Start UTC: {summary['start_utc']}")


@main.command()
@click.option("--data-dir", default="data/raw", help="Data directory path")
@click.option("--output", default="analysis_output.json", help="Output file path")
def analyze(data_dir: str, output: str):
    """Run complete Sol 200 analysis"""
    analysis = Sol200Analysis(Path(data_dir))

    results = {
        "metadata": analysis.get_metadata(),
        "detector_stats": analysis.detector_statistics(),
        "observation_summary": analysis.observation_summary(),
    }

    output_path = Path(output)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    click.echo(f"\nAnalysis complete. Results written to {output_path}")


if __name__ == "__main__":
    main()
