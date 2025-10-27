# Changelog

## [1.0.0] - 2025-10-27

### Changed
- Modernized codebase with type hints and modern Python patterns
- Replaced pandas with polars for better performance
- Added proper CLI with click
- Restructured as installable package with pyproject.toml
- Professional test suite with pytest fixtures
- Ruff for linting and formatting

### Added
- RAD data parser for observation records
- Analysis module with detector statistics and observation summaries
- CLI commands: metadata, observations, analyze
- Comprehensive test coverage

### Removed
- Legacy plotting scripts (basic matplotlib examples)
- Conda environment.yml (using uv + pyproject.toml)
- Old Makefile (replaced with modern tooling)
- Placeholder tests (replaced with real assertions)
