# Elevator Vibration Data Analysis — High-Level Report

**Project:** `elevator-vibration-analysis`  
**Author:** Steven Zhang  
**Date:** 2026-04-30  
**Version:** 0.3.0 — Sensor mounting clarified; state_detection implemented

---

## ⚠️ Critical: Sensor Mounting

**The accelerometer is mounted on the elevator DOOR (single side, car door panel).**

| Axis | Direction | What It Measures |
|:-----|:----------|:-----------------|
| **X** | Door open/close direction (through doorway) | Door panel vibration along travel path; door rattling during motion; door operation events |
| **Y** | Lateral along door plane (side-to-side) | Door panel sideways sway; guide rail-induced lateral shake |
| **Z** | Vertical (gravity-aligned) | Vertical ride dynamics; gravity (~1g static); bounce and jerk |

**Key implication:** We are measuring **door panel vibration**, not car body vibration. The door panel is mechanically coupled to the car but has its own resonance and rattling behavior. All interpretations must account for this mounting context.

---

## 1. Executive Summary

This report defines the analytical framework for triaxial elevator vibration data collected at
**100 Hz** sampling rate. The framework covers the full lifecycle from raw signal ingestion to
predictive maintenance, organized into six analysis domains. Approximately **55% of the analysis
capabilities require zero labeled data**; the remainder can be progressively unlocked with modest
annotation effort or fault records.

The project structure under `~/Workspace/elevator-vibration-analysis/` provides a Python toolkit
(`elevator_vibration`) implementing all analysis modules, ready to receive sample data.

---

## 2. Data Specification

### 2.1 Minimum Required Fields

| Field | Type | Unit | Note |
|-------|------|------|------|
| `timestamp` | float / datetime | s | Relative or absolute time |
| `ax` | float | m/s² or g | **X-axis: door open/close direction** (through doorway). Captures door panel rattling and door operation |
| `ay` | float | m/s² or g | **Y-axis: lateral along door plane** (sideways sway) |
| `az` | float | m/s² or g | **Z-axis: vertical** (gravity-aligned). Includes ~1g static |

### 2.2 Sensor Mounting Implications

Since the sensor is on the **door panel** (not the car frame):

| Aspect | Impact |
|:-------|:-------|
| Door operation | X-axis will spike dramatically during door open/close — these are NOT ride quality events |
| Door rattling | Elevated X-axis RMS during motion indicates door panel looseness or guide rail issues |
| Y-axis lateral | Captures both car sway AND door panel side-to-side play |
| Z-axis vertical | Most reliable axis for ride quality — least affected by door-specific artifacts |
| Gravity reference | Sensor Z may tilt slightly when door moves; gravity estimate should use stationary segments only |
| Composite acceleration | Now represents **door panel total vibration**, not car body vibration |

### 2.3 Optional Enrichment Fields

| Field | Benefit |
|-------|---------|
| `floor` | Enables floor-aligned analysis, guide wear localization |
| `direction` (up/down) | Simplifies run segmentation |
| `door_state` (open/close/moving) | Separates door vibration from ride vibration |
| `load` (kg or %) | Load-dependent analysis |
| `motor_speed` (RPM or Hz) | Enables order tracking |

### 2.4 Nyquist Coverage

100 Hz → Nyquist **50 Hz**. This covers the elevator's primary frequency range:

| Source | Typical Frequency | Covered? |
|--------|:---:|:---:|
| Ride dynamics (run frequency) | 0.5 – 5 Hz | ✅ |
| Traction motor rotation | 10 – 30 Hz | ✅ |
| Rope / suspension vibration | 1 – 10 Hz | ✅ |
| Guide shoe–rail friction | Broadband | ✅ |
| Gear mesh (if geared) | 20 – 50 Hz | ✅ (if ≤50 Hz) |
| Rolling-element bearing faults | 50 – 2000+ Hz | ⚠️ Partial (envelope only) |

> Bearing fault *frequencies* are above Nyquist, but their *time-domain signatures*
> (kurtosis elevation, crest factor spikes) remain detectable.

---

## 3. Signal Processing Pipeline

```
Raw Data
    │
    ▼
┌─────────────────────┐
│ 1. Quality Checks   │  → Clipping, missing samples, timestamp monotonicity
├─────────────────────┤
│ 2. DC Removal       │  → High-pass filter or segment-wise mean subtraction
├─────────────────────┤
│ 3. Gravity          │  → Separate static 1g from dynamic az (requires sensor
│    Separation        │     orientation knowledge)
├─────────────────────┤
│ 4. Axis Alignment   │  → Optional: rotate to principal motion direction via PCA
├─────────────────────┤
│ 5. Composite Accel  │  → a_total = sqrt(ax² + ay² + (az - g)²)
├─────────────────────┤
│ 6. Run Segmentation │  → Energy detector / change-point detection
└─────────────────────┘
    │
    ▼
 Per-Segment Analysis
 (time-domain, frequency-domain, ride quality, diagnostics)
```

### 3.1 Noise Floor Estimation

Isolate stationary segments (e.g., elevator parked with doors closed). Compute per-axis
RMS noise → establishes the sensor's effective dynamic range and minimum detectable
vibration level. Plots: noise histogram per axis, Allan deviation if long stationary
recordings are available.

### 3.2 Gravity Separation

The Z-axis accelerometer measures **g + a_z_dynamic**. Techniques:

- **Known orientation**: subtract 1×g in the sensor's Z direction (simplest, most common)
- **Rotation matrix**: if sensor orientation is known (e.g., from installation spec)
- **Long-term average**: for stationary segments, the mean equals gravity

Without orientation metadata, the default assumption is that the sensor Z-axis is aligned
with gravity (vertical mounting on car frame).

---

## 4. Analysis Domains

### 4.1 Time-Domain Analysis

| Metric | Formula / Description | Diagnostic Value |
|--------|----------------------|------------------|
| **RMS** | √(1/N Σ a²) | Overall vibration energy |
| **Peak** | max\|a\| | Maximum instantaneous shock |
| **Peak-to-Peak** | max(a) − min(a) | Excursion envelope |
| **Crest Factor** | Peak / RMS | Shock content; >6 flags problem |
| **Kurtosis** | E[(a−μ)⁴] / σ⁴ | Bearing degradation; >3 indicates impact content |
| **Skewness** | E[(a−μ)³] / σ³ | Asymmetry; directional impact |
| **Jerk** | da/dt (finite diff) | Ride comfort; ISO 18738 key metric |
| **Zero-crossing rate** | Count per window | Rough spectral centroid proxy |
| **Impulse count** | Samples exceeding N×RMS | Quantify shock events |

#### Run Event Detection

**Note: Sensor is door-mounted.** Door open/close events produce massive X-axis spikes and must be separated from ride events.

```
Event             | Signal Signature (door-mounted sensor)
──────────────────|──────────────────────────────────────────────────
Run start         | Sudden az jump + sustained non-zero
Acceleration      | Non-zero mean az, elevated RMS (all axes)
Constant speed    | az ~0, low RMS; X/Y may show door rattle
Deceleration      | Non-zero mean az (opposite sign)
Leveling          | Short low-amplitude bursts in Z
Door operation    | DOMINANT X-axis spike (1-3s); Y may also spike; Z minimal
Stationary        | All axes at noise floor
Door rattle       | Elevated X-axis RMS during cruise (key diagnostic signal)
```

Detection methods: sliding-window RMS threshold with hysteresis, change-point detection
(PELT algorithm from `ruptures`), or HMM with expected state topology.

---

### 4.2 Frequency-Domain Analysis

#### 4.2.1 FFT and Power Spectral Density (PSD)

- **Method**: Welch's averaged periodogram with 50% overlap, 1-second Hanning windows
- **Output**: PSD (dB re 1 (m/s²)²/Hz) for each axis + composite
- **Key features to identify**:
  - Dominant peaks and their harmonics
  - Noise floor level
  - Spectral roll-off frequency
  - 1×, 2×, 3× running-frequency components (unbalance / misalignment indicators)

#### 4.2.2 Spectrogram (STFT)

- **Purpose**: Track how frequency content evolves during a ride
- **Parameters**: 256-sample window (~2.5 s), 75% overlap
- **Visualization**: Time–frequency heatmap with run phases annotated
- **Insights**: Transient excitation during start/stop, resonance excitation, speed-dependent tones

#### 4.2.3 Envelope Spectrum

- **Purpose**: Demodulate high-frequency carrier to reveal low-frequency fault patterns
- **Method**: Band-pass filter around a resonance zone → Hilbert transform → FFT of envelope
- **Application**: Detecting repetitive impacts (bearing faults, gear mesh) despite Nyquist limit

#### 4.2.4 Cross-Spectral Analysis

- **Inter-axis coherence**: γ²(f) — how much of Y's vibration is linearly predictable from X
- **Cross-phase**: Phase relationship between axes at shared frequencies
- **Diagnostic value**: High coherence at 1×/2× run frequency suggests misalignment or
  structural coupling

#### 4.2.5 Cepstrum

- **Purpose**: Detect periodic structures in the spectrum (harmonic families, sidebands)
- **Method**: IFFT of log-magnitude spectrum
- **Application**: Gear mesh and bearing fault families — limited by 50 Hz Nyquist but
  still useful for low-frequency harmonic patterns

---

### 4.3 Ride Quality Assessment

#### 4.3.1 ISO 2631-1: Mechanical Vibration and Shock — Evaluation of Human Exposure to Whole-Body Vibration

**Frequency weightings:**

| Axis | Weighting | Application |
|------|-----------|-------------|
| X (horizontal, door direction) | Wd | ⚠️ Door-mounted: heavily contaminated by door rattle |
| Y (horizontal, lateral) | Wd | Seated/standing occupant |
| Z (vertical) | Wk | Seated/standing occupant |

**Key metrics:**

| Metric | Symbol | Definition |
|--------|--------|------------|
| Frequency-weighted RMS | a_w | √(1/T ∫ a_w²(t) dt) |
| Vibration Dose Value | VDV | (∫ a_w⁴(t) dt)^(1/4) |
| Maximum Transient Vibration Value | MTVV | max of running RMS (1 s window) |
| Crest Factor | CF | Peak / RMS |

**Decision logic** (ISO 2631-1, Section 8):

- CF ≤ 9 → report a_w alone
- CF > 9 → report both a_w AND VDV (or MTVV)

**Health guidance caution zones** (ISO 2631-1 Annex B): compare VDV against thresholds
for 8-hour equivalent daily exposure.

#### 4.3.2 ISO 18738: Measurement of Ride Quality — Lifts (Elevators)

**Core metrics:**

| Metric | Symbol | Typical Limit | Note |
|--------|--------|:---:|------|
| Peak acceleration | a_peak | ≤ 1.5 m/s² | Start / stop |
| Peak deceleration | | ≤ 1.5 m/s² | Normal stop |
| Maximum jerk | j_max | ≤ 2.5 m/s³ | da/dt |
| A95 acceleration | a_95 | — | 95th percentile |
| Constant-speed vibration | a_v | ≤ 0.15 m/s² | RMS during cruise |

**Measurement interval**: From door-close completion to door-open initiation (excludes
door operation).

**Reporting**: Per-direction (up/down), per-run, aggregated statistics (mean, max, 95th
percentile across runs).

#### 4.3.3 Additional Comfort Indices

| Index | Method |
|-------|--------|
| **Janeway comfort** | Weighted acceleration in 1–60 Hz range |
| **Sperling ride index** | Used in rail; adapted for vertical transport |
| **Overall vibration total value** | √(1.4²·a_wx² + 1.4²·a_wy² + a_wz²) — ISO 2631 vector sum |

---

### 4.4 Fault Diagnostics

#### 4.4.1 Detection Matrix (Door-Mounted Sensor)

**Reminder:** All vibration is measured at the door panel, not the car frame.

| Fault | Primary Indicator | Key Axis | Detectable? | Note |
|-------|-------------------|:--------:|:---:|------|
| **Door rattle / looseness** | Elevated X-axis RMS during cruise; broadband X noise | X | ✅ Yes | Primary diagnostic for door-mounted sensor |
| **Guide rail wear** | X-axis vibration increase at specific floors; Y-axis lateral shake | X, Y | ✅ Yes | Door panel amplifies rail irregularities |
| **Unbalance** | 1× run-frequency peak in Z | Z | ✅ Yes | Z-axis least affected by door artifacts |
| **Misalignment** | 2×, 3× harmonics in Z; high X-Y coherence | Z | ✅ Yes | Z cleanest for harmonic analysis |
| **Mechanical looseness** | Sub-harmonics + raised noise floor; chaotic X spectrum | X, Z | ✅ Yes | X-axis door rattle is a looseness indicator itself |
| **Rope/suspension anomaly** | Low-frequency Z modulation | Z | ✅ Yes | Z is clean axis for traction analysis |
| **Damper degradation** | Resonance frequency shift in Z over time | Z | ✅ Yes | Baseline comparison needed |
| **Bearing early degradation** | Kurtosis > 3 in Z; crest factor rising trend | Z | ✅ Yes | Use Z to avoid door-related false positives |
| **Door mechanism wear** | X-axis spike amplitude/pattern change during door operation | X | ⚠️ | Need isolated door-operation segments |
| **Bearing advanced fault** | BPFO/BPFI in envelope spectrum | Z | ⚠️ Partial | Above Nyquist; envelope method only |

#### 4.4.2 Trending for Predictive Maintenance

For any metric _m_ (RMS, kurtosis, crest factor, dominant frequency amplitude):

1. **Baseline** (n ≥ 30 healthy runs): μ₀, σ₀
2. **Alert threshold**: μ₀ + 3σ₀
3. **Alarm threshold**: μ₀ + 6σ₀
4. **Trend test**: Mann-Kendall on sequential measurement windows

Degradation rate estimation requires ≥5 measurements over time.

---

### 4.5 Operational State Recognition

```
                    ┌──────────┐
                    │   IDLE   │
                    └────┬─────┘
                         │ door close detected
                    ┌────▼─────┐
                    │ ACCEL ↑  │  (or ↓)
                    └────┬─────┘
                         │ acceleration → 0
                    ┌────▼─────┐
                    │  CRUISE  │
                    └────┬─────┘
                         │ deceleration detected
                    ┌────▼─────┐
                    │ DECEL    │
                    └────┬─────┘
                         │ velocity → 0
                    ┌────▼─────┐
                    │ LEVELING │
                    └────┬─────┘
                         │ door open
                    ┌────▼─────┐
                    │   IDLE   │
                    └──────────┘
```

**Implementation approaches (ordered by complexity):**

1. **Energy-threshold state machine** — 0-label, ~20 lines of Python
2. **PELT change-point detection** (`ruptures` library) — 0-label
3. **HMM with known topology** — ~5 labeled runs for parameter estimation
4. **Time-series classifier** (sliding window → XGBoost) — ~10 labeled runs
5. **Deep learning** (1D-CNN / TCN) — 50+ labeled runs

---

### 4.6 Machine Learning & AI

#### 4.6.1 Unsupervised (Zero Label)

| Task | Method | Value |
|------|--------|-------|
| Anomaly detection | Isolation Forest, LOF, One-Class SVM | Flag unusual runs |
| Clustering run types | K-Means, DBSCAN, HDBSCAN | Discover floor patterns, load states |
| Dimensionality reduction | PCA, t-SNE, UMAP | Visualize run similarity, spot outliers |
| Self-supervised representation | SimCLR, BYOL (time-series adapted) | Generic features for downstream tasks |

#### 4.6.2 Supervised (Requires Labels)

| Task | Label Type | Method | Min Samples |
|------|------------|--------|:---:|
| Run-phase segmentation | Per-sample phase label | XGBoost / LightGBM | ~10 runs |
| Fault type classification | Fault type + normal | XGBoost / Random Forest | 30/class |
| Ride quality regression | ISO score or human rating | Gradient Boosting Regressor | 50 runs |
| Remaining useful life | Time-to-failure | XGBoost / LSTM / TCN | Run-to-failure data |
| Floor identification | Floor number | KNN / XGBoost on spectral features | 2 runs/floor |

---

## 5. Feasibility Matrix Summary

| Tier | Label Requirement | Analysis Count | % of Total |
|:----:|-------------------|:---:|:---:|
| ✅ | **Zero label** | ~28 analyses | 55% |
| 🔵 | Few labeled runs (3-10) | ~13 analyses | 25% |
| 🔴 | Fault labels / repair records | ~7 analyses | 13% |
| ⚫ | Extra sensors / metadata | ~4 analyses | 7% |

> **Recommended progression:** Exhaust all ✅ analyses first, then invest ~30 minutes
> labeling 3–5 complete elevator runs to unlock the 🔵 tier.

---

## 6. Implementation Plan

### Phase 1 — Foundation ✅ Complete (2026-04-30)
- [x] Project scaffolding (`pyproject.toml`, module skeleton)
- [x] High-level design document (this report)
- [x] Sensor mounting clarified: **door-mounted accelerometer** (Section 2.2)
- [x] `data_loader.py` — S3 ingestion (bucket: `lbb-simtech-raw`, 238 serials, AES256-SSE/KMS)
- [x] `preprocessing.py` — DC removal, gravity separation, clipping check, noise floor, gap detection (7 functions)
- [x] `time_domain.py` — RMS, peak, crest factor, kurtosis, skewness, jerk, run event detection, velocity-based segmentation (16 functions)
- [x] `frequency.py` — FFT, PSD, STFT, envelope spectrum, cepstrum, inter-axis coherence, cross-spectrum (13 functions)
- [x] `ride_quality.py` — ISO 2631-1 Wk/Wd weighting, weighted RMS, VDV, MTVV; ISO 18738 A95, peak accel, jerk (15 functions)
- [x] `state_detection.py` — Velocity integration with drift correction, state classification (standing/accel/cruise/decel × up/down), phase extraction (341 lines, 6 functions)
- [x] `visualization.py` — 3-axis timeseries, PSD, spectrograms, run overview, ride quality dashboard, cross-trip comparison auto-report (10 functions)
- [x] Sample data downloaded & validated — 5 trips from `S0783L01A` (2026-04-29 + 2026-03-30 for 1-month comparison)
- [x] Integration test — all modules verified on real data
- [x] 1-month comparison analysis — April 29 vs March 30, door-mounted reinterpretation
- [x] Exploratory Jupyter notebook

### Phase 2 — Core Analysis
- [x] `frequency.py` — FFT, PSD, STFT, envelope spectrum **(promoted from Phase 2)**
- [x] `ride_quality.py` — ISO 2631 & ISO 18738 **(promoted from Phase 2)**
- [ ] `state_detection.py` — Energy-threshold segmenter → proper state machine
- [ ] `visualization.py` — Spectrograms, ride quality dashboards

### Phase 3 — Diagnostics
- [ ] `diagnostics.py` — Unbalance, misalignment, looseness indicators
- [ ] Statistical baseline & trending across dates
- [ ] Anomaly detection (Isolation Forest)

### Phase 4 — Advanced
- [ ] Labeled data → supervised phase segmentation
- [ ] Fault classification (if labels available)
- [ ] Automated report generation

---

## 7. Python Toolkit Architecture

```
src/elevator_vibration/
├── __init__.py          # Package metadata, exports EleveatorDataLoader/TripRef/VibrationRecord
├── data_loader.py       # S3 ingestion: lbb-simtech-raw bucket, zip parsing, .env credential loading
├── preprocessing.py     # Quality checks, DC removal, gravity separation, axis alignment (7 functions)
├── time_domain.py       # RMS, peak, kurtosis, jerk, run detection, velocity segmentation (16 functions)
├── frequency.py         # FFT, PSD, STFT, envelope, coherence, cepstrum, cross-spectrum (13 functions)
├── ride_quality.py      # ISO 2631-1 Wk/Wd, VDV, MTVV; ISO 18738 A95, peak accel, jerk (15 functions)
├── state_detection.py   # State machine, change-point, HMM segmenter (stub)
├── diagnostics.py       # Fault indicators, trending, baseline (stub)
├── visualization.py     # Timeseries, PSD, spectrogram, run overview, ride quality dashboard, auto-report (10 functions)
```

### Key Dependencies

| Library | Purpose | Status |
|---------|---------|:---:|
| `numpy`, `scipy` | Core computation, signal processing, filters | ✅ In use |
| `boto3` | S3 data access | ✅ In use |
| `pandas` | Data loading, time-series management | Lazy import |
| `matplotlib` | Visualization | ✅ In use |
| `seaborn`, `plotly` | Optional visualization | Not installed |
| `scikit-learn` | ML models (anomaly, classification, clustering) | Pending |
| `pywavelets` | Wavelet transforms for time-frequency | Pending |
| `jupyter` | Interactive exploration | Notebook ready |

### Data Sources

| Source | Format | Status |
|--------|--------|:---:|
| **S3: `lbb-simtech-raw`** | Zip → single-line `"HH:MM:SS:us,x,y,z"` records | ✅ Operational |
| Local JSONL | Pre-parsed trip data (data/samples/) | ✅ Test cache |
| CSV, HDF5, Parquet | Planned extensible formats | TBD |

---

## 8. Standards Reference

| Standard | Title | Key Metrics |
|----------|-------|-------------|
| **ISO 2631-1:1997** | Whole-body vibration | a_w, VDV, MTVV, Crest Factor |
| **ISO 18738:2003** | Lift ride quality | a_peak, a_95, j_max, a_v |
| **ISO 8041:2005** | Instrumentation requirements | Sensor specs, frequency weightings |
| **ISO 18738-1:2012** | Ride quality (updated) | Supersedes 2003 edition |
| **GB/T 24474-2009** | Chinese equivalent of ISO 18738 | Same metrics, localized |
| **VDI 2566** | Noise/vibration limits for lifts | German engineering guideline |

---

## 9. References

1. Griffin, M. J. (1990). *Handbook of Human Vibration*. Academic Press.
2. Randall, R. B. (2011). *Vibration-based Condition Monitoring*. Wiley.
3. Smith, S. W. (1997). *The Scientist and Engineer's Guide to Digital Signal Processing*.
4. ISO/TC 108/SC 4. (1997). ISO 2631-1: Mechanical vibration and shock.
5. ISO/TC 178. (2012). ISO 18738-1: Measurement of ride quality — Lifts.
6. Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of offline change point
   detection methods. *Signal Processing*, 167, 107299.

---

*This is a living document. Revisions will be tracked as analysis modules are implemented.*
