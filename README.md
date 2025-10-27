# NASA Mars Sol 200 Analysis

Curiosity Rover mission data analysis for Sol 200.

## Overview

Analysis of NASA's Mars Science Laboratory (Curiosity Rover) data from Sol 200 (Mars day 200 of mission). Processes Radiation Assessment Detector (RAD) observations and mission telemetry.

**Dataset**: NASA PDS (Planetary Data System) EDR files from Sol 200
**Mission**: Mars Science Laboratory
**Date**: 2013-058/059 (Earth), Sol 200 (Mars)
**Location**: Gale Crater, Mars

## Installation

```bash
uv venv
source .venv/bin/activate  # or `.venv/bin/activate.fish`
uv pip install -e ".[dev]"
```

## Usage

```bash
# View mission metadata
sol200 metadata

# Parse RAD observations
sol200 observations

# Run full analysis
sol200 analyze --output results.json
```

## Python API

```python
from pathlib import Path
from src import Sol200Analysis

analysis = Sol200Analysis(Path("data/raw"))

# Get detector statistics
stats = analysis.detector_statistics()

# Parse observations
observations_df = analysis.parse_rad_observations()

# Observation summary
summary = analysis.observation_summary()
```

## Testing

```bash
pytest
```

## Structure

```
├── src/
│   ├── analysis.py       # Main analysis module
│   ├── rad_parser.py     # RAD data file parser
│   ├── visualize.py      # Visualization generation
│   ├── findings.py       # Scientific interpretation
│   └── cli.py            # Command-line interface
├── tests/                # Test suite
├── examples/             # Complete analysis workflow
├── data/raw/             # NASA EDR files (145MB)
└── FINDINGS.md           # Analysis results
```

## Technologies

**Python 3.11+** | Polars | Click | pytest | Ruff

## License

MIT

---

**Author**: Cody Kickertz
**Contact**: [LinkedIn](https://linkedin.com/in/Cody-Kickertz/)
