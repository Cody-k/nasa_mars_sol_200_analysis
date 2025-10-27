# NASA Mars Sol 200 Analysis

**Project Structure (Template-Aligned):**

- `src/`: Main analysis modules and code
- `data/raw/`: All raw data files (including mastcam/)
- `data/processed/`, `data/external/`: Reserved for future processed/external data
- `tests/`: Minimal test suite (pytest)
- `docs/`: Project documentation, onboarding, AI orientation, changelog
- `notebooks/`: (Optional) Jupyter notebooks for exploration
- `config/`: Configuration files (YAML)
- `.gitea/`: Issue/PR templates for Gitea
- `Makefile`, `environment.yml`, `pyproject.toml`, `.gitignore`: Project management

**Quickstart:**

1. `conda env create -f environment.yml` (or `conda env update -f environment.yml`)
2. `conda activate nasa-mars-sol-200`
3. `make test` (run tests)
4. Explore code in `src/` and data in `data/raw/`

**Onboarding:**
- See `docs/AI_ORIENTATION.md` for AI contributor guidance
- See `docs/CHANGELOG.md` for project history
- See `docs/DATABASE_SETUP.md` for database/config info

**Original Project Summary:**
This project analyzes data collected by the Curiosity Rover on Sol 200 of the Mars Science Laboratory (MSL) mission. The primary focus is on environmental data and high-resolution Mastcam images, with an emphasis on understanding detector parameters.

**Goals, Analysis, and Future Work:**
- Analyze detector parameters (Detector B/E)
- Explore observation time for Sol 200
- Foundation for future analysis, including ML if more data is acquired

**Contact:**
- Maintainer: Your Name (<your.email@example.com>)
- For issues, use Gitea issue templates
