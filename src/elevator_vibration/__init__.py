"""
Elevator Vibration Analysis Toolkit
====================================

Analysis of triaxial (X-Y-Z) accelerometer data from elevator systems.
Sampling rate: 100 Hz.  Nyquist: 50 Hz.

Modules:
    data_loader     — S3 data ingestion (bucket: lbb-simtech-raw, ap-southeast-1)
    preprocessing   — DC removal, gravity separation, outlier detection, clipping check
    time_domain     — RMS, peak, crest factor, kurtosis, jerk, event detection
    frequency       — FFT, PSD, STFT, cepstrum, envelope spectrum, cross-spectrum
    ride_quality    — ISO 2631-1, ISO 18738 compliance metrics
    state_detection — Run segmentation (standing/accel/cruise/decel × up/down) + door detection
    diagnostics     — Fault indicators: door rattle, unbalance, misalignment, looseness, bearing
    visualization   — Time-series plots, spectrograms, polar plots, dashboards
"""

from elevator_vibration.data_loader import (
    ElevatorDataLoader,
    TripRef,
    VibrationRecord,
    quick_browse,
)
from elevator_vibration.state_detection import (
    StateAnalysis,
    Phase,
    analyze_trip_states,
    classify_motion_state,
    detect_direction,
    integrate_velocity,
    extract_phases,
    STATE_LABELS,
)

__version__ = "0.1.0"
