# Sol 200 Research Roadmap

Advanced analysis plan using cutting-edge techniques on Curiosity Rover Sol 200 data.

---

## Available Data (Verified)

### RAD (Radiation Assessment Detector)
**Current:** 17MB text file with rich radiation physics data

**Untapped:**
- **LET Spectra** (44 energy bins × 2 detectors) - Particle energy deposition distributions
- **Dosimetry Totals** - 9.243 (B), 8.993 (E) in cGy units
- **PHA Data** (Pulse Height Analysis) - 36 detector channels × thousands of events
- **Counter Data** - Particle event classifications (L1, L2 logic levels)
- **44 Observation Records** - Temporal radiation measurements

### Mastcam
**Available:** 24 paired .img/.lbl files + 1 thumbnail .jpg
**Size:** 1.6MB - 4.7MB per image
**Coverage:** Sol 200 surface imagery from Gale Crater

### External (Can Fetch)
- CHEMCAM spectroscopy for Sol 200 (NASA PDS)
- APXS elemental composition (NASA PDS)
- Environmental data (pressure, temp) from REMS
- Solar activity data for Feb 2013 (NOAA)

---

## Tier 1: Advanced Radiation Physics (High Scientific Value)

### 1.1 LET Spectrum Analysis ⭐ **Most Impactful**

**Objective:** Analyze Linear Energy Transfer distributions to identify particle types and radiation quality

**Scientific Value:**
- LET spectrum shows energy deposition patterns
- Different particles (protons, alpha, heavy ions) have distinct signatures
- Critical for human Mars mission radiation safety
- Published literature uses this exact analysis

**Data Available:**
- DOSIMETRY_LET_B_A1, A2 (44 energy bins each)
- Energy bins from 0.167 to 290.133 keV/μm
- Particle counts per bin (actual measurements)

**Analysis Plan:**
1. Parse LET histogram data (44 bins × 2 detector regions × 44 observations)
2. Identify particle species from energy deposition patterns:
   - Minimum ionizing peak (~0.2 keV/μm) = high-energy protons
   - Low-LET shoulder = electrons, low-energy protons
   - High-LET tail (>10 keV/μm) = heavy ions (Fe, O, C, He)
3. Calculate total particle flux by species
4. Compare A1 vs A2 regions (detector spatial variation)

**Cutting-Edge Techniques:**
- **Deconvolution with PyMC** - Bayesian inference to separate overlapping particle contributions
- **Gaussian Mixture Models** - Cluster energy deposition into particle types
- **Physics-Informed Neural Networks (PINN)** - Constrain ML with radiation physics equations

**Outputs:**
- LET spectrum visualization with species identification
- Particle flux estimates (protons, alphas, heavy ions)
- Comparison with published Curiosity results (validation)

**Rigor:**
- Cross-validate with known physics (Bragg curve shapes)
- Compare with published MSL/RAD papers
- Error propagation through analysis chain
- Comprehensive testing of parsing logic

---

### 1.2 Dose Equivalent Calculation (Human Mission Planning)

**Objective:** Convert absorbed dose to biological dose equivalent for crew exposure assessment

**Scientific Value:**
- Raw dose (cGy) ≠ biological risk
- Quality factor Q depends on LET (high-LET particles more damaging)
- Critical for Mars mission duration limits

**Method:**
1. Apply ICRP quality factor function: Q(LET)
2. Weight LET spectrum by quality factors
3. Calculate dose equivalent (Sv) from absorbed dose (Gy)
4. Compare with safe exposure limits

**Cutting-Edge:**
- **Monte Carlo Simulation** - Propagate measurement uncertainties
- **Risk Modeling** - Integrate with NASA's Space Cancer Risk model
- **Shielding Optimization** - Calculate required habitat shielding thickness

**Outputs:**
- Dose equivalent for Sol 200 (comparison with Earth background)
- Estimated annual crew exposure at Gale Crater
- Shielding requirements for habitats

---

### 1.3 Particle Event Classification (ML on Physics Data)

**Objective:** Train classifier on PHA (Pulse Height Analysis) data to identify particle types

**Scientific Value:**
- 36 detector channels × thousands of events = rich dataset
- Supervised learning with physics labels
- Novel application of ML to planetary science

**Data:**
- PHA arrays (36 channels: rad_00 through rad_35)
- Energy depositions + corrections
- Priority flags, hardware triggers, logic masks

**ML Approach:**
1. **Feature Engineering:**
   - Energy deposition patterns across 36 channels
   - Coincidence patterns (which detectors fired together)
   - Energy ratios between detectors
   - Trigger logic patterns

2. **Labels (Physics-Based):**
   - Protons: specific E1/E2 energy ratio
   - Alphas: higher E-loss, characteristic pattern
   - Heavy ions: saturate detectors, specific signatures
   - Electrons: low energy, different penetration

3. **Models to Test:**
   - **Random Forest** - Baseline, interpretable
   - **XGBoost** - High performance on tabular data
   - **1D CNN** - Spatial pattern recognition across detector array
   - **Transformer** - Attention mechanism for detector correlations

**Cutting-Edge:**
- **Physics-Informed ML** - Loss function incorporates conservation laws
- **Explainable AI** - SHAP values to understand what detector patterns identify particles
- **Active Learning** - Iteratively label uncertain predictions

**Validation:**
- Cross-validate with LET spectrum results
- Compare with published particle identification algorithms
- Precision/recall/F1 scores per particle type

---

## Tier 2: Computer Vision on Mars Surface (High Visibility)

### 2.1 Mastcam Image Analysis

**Objective:** Analyze 24 Mastcam images from Sol 200 using modern computer vision

**Available:**
- 24 stereo image pairs (.img format - NASA PDS IMG)
- Label files (.lbl) with camera parameters
- 1 thumbnail .jpg

**Analyses:**

**A. Terrain Classification**
- Segment images into: rock, sand, sky, distant features
- Use **Segment Anything Model (SAM)** for zero-shot segmentation
- Quantify surface composition percentages

**B. Rock Detection and Characterization**
- **YOLO or Faster R-CNN** for rock instance detection
- Size distribution analysis
- Shape classification (rounded vs angular)
- Spatial distribution patterns

**C. Geological Feature Identification**
- **Vision Transformer (ViT)** fine-tuned on Mars imagery
- Classify terrain types (bedrock, regolith, aeolian features)
- Compare with orbital imagery from HiRISE

**D. 3D Reconstruction**
- Stereo pair analysis (if we have both left/right Mastcam)
- Depth estimation using **MiDaS or Depth Anything**
- Digital elevation model of visible terrain

**E. Color Analysis**
- True color reconstruction from filter data
- Dust opacity estimation
- Atmospheric haze analysis

**Cutting-Edge:**
- **Foundation Models:** Use pre-trained SAM, CLIP, DINOv2
- **Self-Supervised Learning:** Train on unlabeled Mars images
- **Neural Radiance Fields (NeRF):** 3D scene reconstruction from multiple views
- **Diffusion Models:** Enhance image quality, inpaint missing data

**Scientific Validation:**
- Compare with published Mastcam analyses
- Validate against ground truth from mission team
- Cross-reference with CHEMCAM targets (if rocks analyzed)

---

## Tier 3: Multi-Instrument Integration (Data Engineering Showcase)

### 3.1 Correlate RAD with Environmental Conditions

**Fetch from NASA PDS:**
- **REMS** (Rover Environmental Monitoring Station) - Pressure, temperature, humidity, UV, wind
- **DAN** (Dynamic Albedo of Neutrons) - Subsurface hydrogen detection
- **Atmospheric opacity** - Dust levels

**Correlations:**
- Does atmospheric pressure affect particle flux? (atmospheric shielding)
- UV vs charged particle correlation
- Dust storms impact on radiation? (Sol 200 specific conditions)

**Techniques:**
- **Time-series cross-correlation**
- **Granger causality** testing
- **VAR (Vector Autoregression)** modeling

---

### 3.2 CHEMCAM Integration (Geochemistry + Radiation)

**Objective:** Correlate radiation environment with surface composition

**If CHEMCAM data available for Sol 200:**
- Element abundances (Fe, Si, O, etc.)
- Rock vs soil spectra
- Target locations

**Analysis:**
- Does local geology affect radiation? (secondary particles from rock interactions)
- Shielding effectiveness of different rock types
- Neutron production correlations

---

## Tier 4: Comparative & Predictive Analysis

### 4.1 Earth vs Mars Radiation Comparison

**Objective:** Quantify Mars radiation hazard relative to Earth environments

**Comparisons:**
- Denver (high altitude) vs Mars surface
- ISS vs Mars surface vs deep space (from cruise data)
- Aircraft altitudes vs Mars

**Data Sources:**
- Sol 200 dose: 9.243 cGy (Detector B)
- Earth background: ~0.3 cGy/year sea level
- ISS: ~15-20 cGy/year

**Visualization:**
- Comparative dose charts
- Time-to-exposure-limit calculations
- Annual dose projections

---

### 4.2 Anomaly Detection in Observation Time Series

**Objective:** Identify unusual radiation events in 44 observations

**Techniques:**
- **Isolation Forest** - Detect outlier observations
- **LSTM Autoencoder** - Learn normal patterns, flag anomalies
- **Prophet** (Facebook) - Time series decomposition with anomaly flagging

**Look for:**
- Solar energetic particle (SEP) events
- Instrument anomalies
- Unexpected radiation spikes

---

## Tier 5: Predictive Modeling (ML Showcase)

### 5.1 Radiation Forecasting Model

**Objective:** Predict radiation levels from environmental parameters

**Features:**
- Atmospheric pressure (affects shielding)
- Solar activity index (from external data)
- Time of day (diurnal variations)
- Seasonal position

**Model:**
- **Gradient Boosting** (XGBoost, LightGBM)
- **Neural Network** with physics constraints
- **Gaussian Process** for uncertainty quantification

**Validation:**
- Train on Sols 1-199, test on Sol 200
- Cross-validate with other published models

---

### 5.2 Particle Flux Prediction

**Objective:** Forecast particle arrival rates using temporal patterns

**Time Series Methods:**
- **ARIMA** - Classical baseline
- **LSTM** - Deep learning for sequences
- **Temporal Fusion Transformer** - State-of-the-art time series

**If successful:** Could predict radiation conditions for mission planning

---

## Tier 6: Novel Interdisciplinary Research

### 6.1 Radiation Impact on Instrument Performance

**Hypothesis:** Does high radiation affect camera or spectrometer performance?

**Test:**
- Correlate RAD measurements with Mastcam image quality metrics
- Analyze if high-energy events cause sensor artifacts
- Timing correlation between RAD spikes and imaging

---

### 6.2 Martian Radiation "Weather" Characterization

**Concept:** Treat radiation like weather - identify patterns, regimes, events

**Analysis:**
- Cluster observations into radiation "conditions" (calm, elevated, storm)
- Identify diurnal patterns
- Seasonal variations (if extended to multiple sols)

**Visualization:**
- Radiation "forecast" style charts
- Risk levels for astronaut EVA activities

---

## Implementation Priority (Ranked by Impact × Feasibility)

### Phase 1: High Impact, Currently Feasible (Do First)

**Week 1:**
1. **LET Spectrum Analysis** - Parse, visualize, identify particle peaks
2. **Dose Equivalent Calculation** - Apply quality factors, human relevance

**Week 2:**
3. **Mastcam Image Processing** - Load .img files, basic segmentation with SAM
4. **PHA Data Parsing** - Extract event data, exploratory analysis

**Outputs:**
- 2-3 new analysis modules
- Publication-quality visualizations
- Scientifically rigorous findings
- Add to FINDINGS.md

---

### Phase 2: Advanced ML (2-3 Weeks)

5. **Particle Event Classification** - Train RF/XGBoost on PHA data
6. **Anomaly Detection** - Identify unusual observations
7. **Advanced Image Analysis** - Rock detection, terrain classification

**Outputs:**
- ML models with validation metrics
- Feature importance analysis
- Documented methodology

---

### Phase 3: Integration (If Time Permits)

8. **Fetch external data** - REMS, CHEMCAM from NASA PDS
9. **Multi-instrument correlation**
10. **Comparative analysis** (Earth vs Mars)

---

## Technology Stack (Cutting-Edge)

**Current:**
- Python 3.11+, Polars, matplotlib

**Add:**

**Radiation Physics:**
- `astropy` - Units, coordinates, time handling
- `scipy.stats` - Statistical distributions, fitting
- `pymc` - Bayesian inference for deconvolution

**Machine Learning:**
- `scikit-learn` - RF, GMM, Isolation Forest
- `xgboost` / `lightgbm` - Gradient boosting
- `pytorch` - Deep learning (CNN, LSTM, Transformer)
- `shap` - Model explainability

**Computer Vision:**
- `segment-anything (SAM)` - Zero-shot segmentation
- `ultralytics (YOLO)` - Object detection
- `transformers` (ViT) - Vision transformers
- `pillow` / `opencv-python` - Image processing
- `planetaryimage` - NASA PDS .img file reader

**Time Series:**
- `prophet` - Facebook's time series library
- `statsmodels` - ARIMA, VAR
- `sktime` - Specialized time series ML

**Data Integration:**
- `requests` - Fetch from NASA PDS API
- `pandas` - Multi-source data merging (when needed with polars)

---

## Detailed Research Spec: LET Spectrum Analysis

### Objective
Decompose Linear Energy Transfer spectrum into particle species contributions using physics-informed machine learning.

### Scientific Background
**LET (keV/μm)** = Energy deposited per unit path length in silicon
- Protons (minimum ionizing): ~0.2 keV/μm peak
- Alphas: ~2-10 keV/μm
- Heavy ions (Fe, O, C): 10-300 keV/μm

### Data Structure (Verified)
```
DOSIMETRY_LET_B_A1:
Row 1: Energy bins (44 values, 0.167 to 290.133 keV/μm)
Row 2: Energy bin widths (0.027 to 47.787 keV/μm)
Row 3: Particle counts (9.412, 51.241, 158.951... down to 0.000)
```

### Analysis Pipeline

**1. Data Extraction**
```python
def parse_let_spectrum(rad_file: Path) -> pl.DataFrame:
    """Parse LET histogram from RAD data file"""
    # Extract energy bins, widths, counts
    # Return structured DataFrame
```

**2. Peak Identification**
```python
# scipy.signal.find_peaks with prominence
# Identify: proton peak (~0.2), alpha region (2-10), heavy ion tail (>10)
```

**3. Species Deconvolution**
- **Gaussian Mixture Model** - Fit overlapping Gaussian peaks
- **Or Bayesian:** PyMC model with physical priors
  - Proton peak: μ ~0.2, σ constrained by physics
  - Alpha: broader peak, higher LET
  - Heavy ions: power-law tail

**4. Flux Calculations**
```python
# Integrate counts under each peak
# Convert to particles/cm²/s using detector geometry
```

**5. Validation**
- Compare with published Curiosity LET spectra (Zeitlin et al. 2013, 2019)
- Check particle ratios match GCR composition (~90% protons, ~9% alphas, ~1% heavy)
- Dose sanity check (flux × LET = dose)

### Testing Strategy
```python
def test_let_parsing():
    """Verify 44 bins extracted correctly"""

def test_peak_identification():
    """Ensure proton peak found at ~0.2 keV/μm"""

def test_flux_calculation():
    """Validate flux units and magnitudes"""

def test_dose_reconstruction():
    """Sum(flux × LET) should equal measured dose"""
```

### Expected Results
- Proton flux: ~50-100 particles/cm²/s (based on published MSL values)
- Alpha: ~5-10 particles/cm²/s
- Heavy ions: <1 particles/cm²/s
- **Mission Impact:** Annual crew dose estimate for Gale Crater

---

## Detailed Research Spec: Mastcam Image Analysis

### Objective
Apply computer vision to characterize Sol 200 Martian surface imagery

### Data Format
**NASA PDS IMG files:**
- Binary raster format
- Label (.lbl) contains metadata (dimensions, bit depth, encoding)
- Need `planetaryimage` library to parse

### Analysis Pipeline

**1. Image Loading**
```python
from planetaryimage import PDS3Image

def load_mastcam_image(img_file: Path) -> np.ndarray:
    """Load NASA PDS IMG file to numpy array"""
    image = PDS3Image.open(img_file)
    return image.image
```

**2. Terrain Segmentation with SAM**
```python
from segment_anything import sam_model_registry, SamPredictor

# Segment into: rocks, sand, sky, distant hills
# Calculate surface composition percentages
```

**3. Rock Detection**
```python
from ultralytics import YOLO

# Train or use pre-trained YOLO on Mars rock imagery
# Detect individual rocks, measure sizes, distribution
```

**4. 3D Reconstruction**
- If stereo pair available (left + right Mastcam)
- Or monocular depth with MiDaS/Depth Anything

**5. Scientific Measurements**
- Rock size distribution (power law expected)
- Surface roughness estimation
- Horizon detection (distance estimation)

### Cutting-Edge Techniques
- **Foundation Models:** SAM (segment anything), CLIP (zero-shot classification)
- **Self-Supervised:** DINOv2 for feature extraction
- **Depth Estimation:** MiDaS v3.1, Depth Anything v2
- **Image Enhancement:** Real-ESRGAN for super-resolution

### Validation
- Compare with published Mastcam analyses
- Cross-reference with mission team annotations
- Geological reasonableness (expert review if possible)

---

## Detailed Research Spec: Particle Type Classifier

### Objective
Train supervised classifier on PHA data to identify particle species

### Dataset Creation

**Features (per event):**
- 36 detector energy depositions (rad_00 to rad_35)
- Trigger priority (pri)
- Hardware priority (hw_pri)
- Logic masks (L2_mask, slow_token_mask)
- SCLK (spacecraft clock - timing)

**Labels (derived from physics):**
```python
def label_particle_type(event: dict) -> str:
    """Apply physics rules to label particle type"""
    # Proton: E_D/E_B ratio ~2-3, specific pattern
    # Alpha: E_D/E_B ratio ~1-2, higher total energy
    # Heavy ion: Saturates detectors, high energy all channels
    # Electron: Low energy, specific penetration pattern
```

### Model Architecture

**Option A: Classical ML (Interpretable)**
```python
from xgboost import XGBClassifier

# Features: 36 energies + derived (ratios, sums, patterns)
# Target: [proton, alpha, heavy_ion, electron, unknown]
# Metrics: Precision, recall, F1 per class
```

**Option B: Deep Learning (Higher Performance)**
```python
import torch
import torch.nn as nn

class ParticleClassifier(nn.Module):
    """1D CNN over detector array"""
    # Conv1D to detect spatial patterns
    # Attention mechanism for detector correlations
    # Physics-informed loss (conservation constraints)
```

### Validation Strategy
1. **K-Fold Cross-Validation** (5 folds)
2. **Hold-out test set** (20% of data)
3. **Physics sanity checks:**
   - Proton fraction ~90% (GCR composition)
   - Total flux matches published values
4. **Ablation studies:** Which features matter most?
5. **Confusion matrix:** Where does model fail?

### Expected Performance
- **Accuracy:** 85-95% (protons easiest, heavy ions hardest)
- **Precision:** >90% for protons, >70% for alphas
- **Interpretability:** SHAP shows detector patterns per particle type

---

## Proposed Project Structure (After Enhancements)

```
nasa-mars-sol200-analysis/
├── src/
│   ├── radiation/
│   │   ├── let_spectrum.py          # LET analysis
│   │   ├── dose_equivalent.py       # Q-factor calculations
│   │   └── pha_classifier.py        # Particle classification
│   ├── vision/
│   │   ├── image_loader.py          # PDS IMG parser
│   │   ├── segmentation.py          # SAM integration
│   │   ├── rock_detection.py        # YOLO rocks
│   │   └── depth_estimation.py      # 3D reconstruction
│   ├── integration/
│   │   ├── pds_fetcher.py           # Fetch REMS, CHEMCAM
│   │   └── correlations.py          # Multi-instrument analysis
│   └── [existing modules]
├── models/
│   ├── particle_classifier.pkl      # Trained XGBoost
│   └── rock_detector.pt             # YOLO weights
├── notebooks/
│   ├── 01_let_analysis.ipynb        # Interactive exploration
│   ├── 02_mastcam_vision.ipynb
│   └── 03_particle_classification.ipynb
├── results/
│   ├── let_spectrum_plot.png
│   ├── particle_flux_estimates.csv
│   ├── mastcam_segmented/
│   └── dose_equivalent_report.pdf
├── tests/
│   ├── test_let_spectrum.py
│   ├── test_particle_classifier.py
│   └── test_image_processing.py
└── FINDINGS.md                      # Expanded with all results
```

---

## Recommended Roadmap (Bulletproof Execution)

### Phase 1: LET Spectrum (Week 1-2)
**Rationale:** Richest data, highest scientific value, no external dependencies

**Deliverables:**
- LET spectrum parser (tested)
- Peak identification (validated against physics)
- Species deconvolution (Bayesian or GMM)
- Flux calculations (cross-checked with published values)
- Visualizations (publication-quality)
- Add to FINDINGS.md

**Success Criteria:**
- Proton peak at 0.15-0.25 keV/μm ✓
- Heavy ion tail matches GCR expectations ✓
- Total flux within 50% of published Curiosity averages ✓

---

### Phase 2: Mastcam Vision (Week 3-4)
**Rationale:** High visibility, demonstrates CV expertise

**Deliverables:**
- PDS IMG loader (tested with all 24 images)
- SAM segmentation (rock/sand/sky percentages)
- Rock detection (count, sizes, distribution)
- Visualizations (annotated images)

**Success Criteria:**
- All 24 images load correctly ✓
- Segmentation looks geologically reasonable ✓
- Rock detection matches visual inspection ✓

---

### Phase 3: Particle Classification (Week 5-6)
**Rationale:** Novel ML application, shows advanced skills

**Deliverables:**
- PHA data parser (tested)
- Labeled dataset (physics rules)
- Trained classifier (XGBoost + optionally NN)
- Validation metrics (precision/recall/F1)
- SHAP analysis (feature importance)

**Success Criteria:**
- Accuracy >85% on hold-out test ✓
- Proton fraction ~90% ✓
- Particle ratios match GCR composition ✓

---

## Quality Assurance (Bulletproof Standards)

### Code Quality
- ✅ Type hints on all functions
- ✅ Comprehensive docstrings
- ✅ Unit tests (>80% coverage target)
- ✅ Integration tests
- ✅ Physics validation tests
- ✅ Ruff linting (zero errors)
- ✅ No magic numbers (constants defined)

### Scientific Rigor
- ✅ Compare with published MSL/RAD papers (Zeitlin et al.)
- ✅ Physics sanity checks (conservation laws, known ratios)
- ✅ Error propagation and uncertainty quantification
- ✅ Peer-reviewable methodology
- ✅ Reproducible (all code + data included)

### Documentation
- ✅ Methodology clearly explained
- ✅ Assumptions stated explicitly
- ✅ Limitations acknowledged
- ✅ References to scientific literature
- ✅ No overclaiming

---

## Expected Outcomes

**Technical Showcase:**
- Radiation physics expertise
- Computer vision on planetary data
- ML for scientific applications
- Multi-modal data analysis
- Production-quality code

**Scientific Contribution:**
- Novel particle classification approach
- Comprehensive Sol 200 characterization
- Human mission planning insights
- Could lead to actual paper (if results strong)

**Career Impact:**
- **NASA/Firefly:** Shows you can do space science research
- **Research orgs:** Demonstrates scientific rigor
- **ML roles:** Shows physics-informed ML
- **Data engineering:** Multi-source integration expertise

---

## My Recommendation: Start with LET Spectrum

**Why:**
- Richest untapped data in current dataset
- No external dependencies (data already present)
- Scientifically rigorous (published methodology)
- High impact for human Mars missions
- Showcases physics + data science

**This weekend:** Build LET spectrum analysis module with proper testing

**Would this be:**
1. Scientifically publishable (with proper validation)
2. Technically impressive (cutting-edge ML + physics)
3. Career-relevant (shows space science capability)

**My assessment: Yes to all three.**

Want me to start implementing LET spectrum analysis with bulletproof testing?
