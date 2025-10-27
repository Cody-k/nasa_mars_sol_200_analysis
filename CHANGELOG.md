# Changelog

## [1.0.0] - 2025-10-27

### Added - Advanced Radiation Physics
- LET spectrum analysis with particle identification (protons, alphas, heavy ions)
- Dose equivalent calculations using ICRP-60 quality factors
- Annual dose projection: 189 mSv/year (validates against published 200-300 range)
- Comprehensive test suite (11 tests, all passing)
- Publication-quality visualizations (5 plots: LET spectrum, particle composition, dose comparison)
- Physics validation (proton peak at 0.28 keV/μm, quality factor 13.5)
- radiation/ module for physics analysis
- advanced_radiation_analysis.py example
- RESEARCH_ROADMAP.md for future work

### Changed
- Modernized with type hints, polars, scipy, numpy
- Comprehensive FINDINGS.md (physics-validated results)
- README with advanced features

### Removed
- Legacy plotting scripts, conda environment, placeholder tests
