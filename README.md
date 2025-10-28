# NASA Mars Sol 200 Analysis

Multi-modal analysis of Curiosity Rover data combining radiation physics, computer vision, and machine learning.

## Overview

Comprehensive analysis of NASA's Mars Science Laboratory (Curiosity Rover) data from Sol 200 (Mars day 200 of mission). Integrates radiation detector analysis, terrain segmentation, and particle classification.

**Features:**
- Advanced radiation physics: LET spectra, dose equivalent calculations, particle identification
- Computer vision: Mastcam image loading, classical terrain segmentation (sky/rocks/sand)
- Machine learning: XGBoost particle classifier for radiation events
- Publication-quality visualizations and analysis

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
from src.vision import MastcamImageLoader, TerrainSegmenter
from src.radiation import LETSpectrum, DoseEquivalent

# Radiation analysis
analysis = Sol200Analysis(Path("data/raw"))
stats = analysis.detector_statistics()
observations_df = analysis.parse_rad_observations()

# Computer vision
loader = MastcamImageLoader(Path("data/raw/mastcam"))
images = loader.load_all_mastcam()

segmenter = TerrainSegmenter()
result = segmenter.segment_terrain(images[0]["data"])
print(f"Sky: {result['composition']['sky']:.1f}%")
print(f"Rocks: {result['composition']['rocks']:.1f}%")
print(f"Sand: {result['composition']['sand']:.1f}%")

# Advanced physics
let = LETSpectrum(rad_file)
spectrum_df = let.parse_let_spectrum()
particle_fractions = let.estimate_particle_composition()
```

## Testing

```bash
pytest
```

## Structure

```
src/
├── analysis.py                    # Core analysis
├── rad_parser.py                  # RAD data parser
├── radiation/
│   ├── let_spectrum.py            # LET analysis, particle ID
│   ├── dose_equivalent.py         # ICRP dose calculations
│   ├── pha_parser.py              # Pulse height analysis parser
│   ├── particle_classifier.py     # ML particle classification
│   └── visualize_spectrum.py      # Physics plots
├── vision/
│   ├── image_loader.py            # NASA PDS image loader
│   ├── segmentation.py            # Classical CV terrain segmentation
│   └── visualize_terrain.py       # Terrain visualization
├── visualize.py                   # Basic charts
├── findings.py                    # Interpretation
└── cli.py                         # CLI

tests/
├── test_*.py                      # Core tests (17 passing)
└── radiation/
    ├── test_let_spectrum.py       # LET physics (6 tests)
    └── test_dose_equivalent.py    # Dose calc (5 tests)

examples/
├── complete_analysis.py              # Basic workflow
├── advanced_radiation_analysis.py    # Full physics
├── mastcam_analysis.py               # Image loading
├── terrain_segmentation.py           # Computer vision demo
└── particle_classification.py        # ML classification

FINDINGS.md               # Comprehensive results
RESEARCH_ROADMAP.md       # Future extensions
```

**Stats:** 1,996 LOC · 17 tests · Python 3.11+

## Technologies

**Core:** Python 3.11+ · Polars · NumPy · SciPy
**Computer Vision:** scikit-image · Pillow
**Machine Learning:** XGBoost · scikit-learn
**Visualization:** matplotlib
**Development:** pytest · Ruff · Click

## License

MIT

---

**Author**: Cody Kickertz
**Contact**: [LinkedIn](https://linkedin.com/in/Cody-Kickertz/)
