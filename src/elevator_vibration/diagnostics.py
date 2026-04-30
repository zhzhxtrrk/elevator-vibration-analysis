"""Fault diagnostics: unbalance, misalignment, looseness, guide wear, bearing degradation.

Detection matrix for door-mounted sensor (Section 4.4.1):
  - X-axis = door open/close direction (door rattle, door operation spikes)
  - Y-axis = lateral / side sway (guide wear, lateral imbalance)
  - Z-axis = vertical (clean mechanical axis for rotating faults)

All functions return dicts with ``severity`` drawn from ``SEVERITY_LEVELS``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elevator_vibration.frequency import (
    compute_fft,
    compute_psd,
    dominant_frequency,
)
from elevator_vibration.time_domain import (
    compute_crest_factor,
    compute_kurtosis,
    compute_rms,
)

if TYPE_CHECKING:
    from elevator_vibration.state_detection import StateAnalysis

# ══════════════════════════════════════════════
#  Module-level constants
# ══════════════════════════════════════════════

SEVERITY_LEVELS = ["normal", "moderate", "elevated", "high", "critical"]

# --- Door rattle thresholds (rattle index = X_cruise_RMS / noise_floor) ---
DOOR_RATTLE_NORMAL = 5.0       # ratio < 5
DOOR_RATTLE_MODERATE = 10.0    # ratio 5–10
DOOR_RATTLE_ELEVATED = 15.0    # ratio 10–15
# ratio > 15 → high

# --- Unbalance thresholds (1× amplitude as multiple of baseline) ---
UNBALANCE_NORMAL = 2.0          # 1× amplitude < 2× baseline
UNBALANCE_MODERATE = 4.0        # 2–4× baseline
UNBALANCE_ELEVATED = 8.0        # 4–8× baseline
# > 8× → high

# --- Misalignment thresholds (2× / 1× ratio) ---
MISALIGNMENT_RATIO_2X = 0.3     # ratio > 0.3 suggests possible misalignment
MISALIGNMENT_RATIO_2X_STRONG = 0.5  # ratio > 0.5 strong indication

# --- Looseness thresholds ---
LOOSENESS_KURTOSIS = 4.0        # excess kurtosis > 4 (raw > 7)
LOOSENESS_CREST_FACTOR = 5.0    # crest factor > 5

# --- Bearing degradation ---
BEARING_TREND_WINDOW = 3        # consecutive trips with monotonic increase

# --- Health score deduction per fault severity ---
HEALTH_DEDUCTION = {
    "normal": 0,
    "moderate": 5,
    "elevated": 10,
    "high": 20,
    "critical": 35,
}

# ══════════════════════════════════════════════
#  Helper utilities
# ══════════════════════════════════════════════


def _find_harmonic_amplitude(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    target_freq: float,
    tolerance: float = 0.25,
) -> float:
    """Extract the amplitude at (or nearest to) a target frequency.

    Args:
        freqs: Frequency array (Hz).
        magnitude: Magnitude spectrum (same length as *freqs*).
        target_freq: Desired frequency (Hz).
        tolerance: Search window half-width in Hz.

    Returns:
        Amplitude at the nearest bin within *tolerance* of *target_freq*.
        Returns 0.0 if no bin falls within the window.
    """
    mask = (freqs >= target_freq - tolerance) & (freqs <= target_freq + tolerance)
    if not np.any(mask):
        return 0.0
    idx = np.argmax(magnitude[mask])
    return float(magnitude[mask][idx])


# ══════════════════════════════════════════════
#  Individual diagnostic functions
# ══════════════════════════════════════════════


def diagnose_door_rattle(
    ax_cruise: np.ndarray,
    noise_floor_x: float,
) -> dict:
    """Diagnose door rattle from X-axis cruise vibration vs noise floor.

    A door-mounted sensor's X-axis is the door open/close direction.
    Elevated RMS during cruise (relative to the stationary noise floor)
    indicates the door panel is loose or poorly adjusted.

    Args:
        ax_cruise: X-axis acceleration during cruise phases (m/s² or mg).
        noise_floor_x: X-axis RMS during stationary/standing periods (same units).

    Returns:
        Dict with keys:
            - ``rattle_index``: Cruise RMS / noise floor (float).
            - ``severity``: One of ``SEVERITY_LEVELS``.
            - ``recommendation``: Human-readable recommendation string.
    """
    if noise_floor_x <= 0 or len(ax_cruise) == 0:
        return {
            "rattle_index": 0.0,
            "severity": "normal",
            "recommendation": "Insufficient data for door rattle assessment.",
        }

    rms_cruise = float(np.sqrt(np.mean(ax_cruise**2)))
    rattle_index = rms_cruise / noise_floor_x

    if rattle_index < DOOR_RATTLE_NORMAL:
        severity = "normal"
        recommendation = "Door rattle within normal range. No action required."
    elif rattle_index < DOOR_RATTLE_MODERATE:
        severity = "moderate"
        recommendation = "Moderate door rattle detected. Inspect door guides and rollers at next maintenance."
    elif rattle_index < DOOR_RATTLE_ELEVATED:
        severity = "elevated"
        recommendation = "Elevated door rattle. Check door panel fasteners, guide shoes, and sill clearance."
    else:
        severity = "high"
        recommendation = "High door rattle — door panel may be loose. Immediate inspection recommended."

    return {
        "rattle_index": round(rattle_index, 2),
        "severity": severity,
        "recommendation": recommendation,
    }


def diagnose_unbalance(
    freqs: np.ndarray,
    mag_z: np.ndarray,
    dominant_freq: float,
    baseline_amplitude: float | None = None,
) -> dict:
    """Diagnose rotating unbalance from 1× running-speed peak in Z-axis spectrum.

    Unbalance produces a dominant peak at the machine's running speed (1×).
    Elevated 1× amplitude relative to a known-good baseline indicates unbalance.

    Args:
        freqs: Frequency bins (Hz) from FFT.
        mag_z: Z-axis magnitude spectrum (m/s²).
        dominant_freq: The dominant frequency in the spectrum (Hz). Used as 1×
            reference if it falls in the typical elevator running-speed range.
        baseline_amplitude: Known-good 1× amplitude. If None, severity is reported
            as 'normal' with a note that no baseline is available.

    Returns:
        Dict with keys:
            - ``amplitude_1x``: Magnitude at 1× running speed (float).
            - ``frequency_1x``: The 1× frequency used (float).
            - ``ratio_to_baseline``: 1× / baseline amplitude (float or None).
            - ``severity``: One of ``SEVERITY_LEVELS``.
            - ``recommendation``: Human-readable recommendation.
    """
    # Identify 1× running speed: typically 1.5–8 Hz for traction elevators
    # Use dominant frequency if it falls in plausible range; otherwise scan.
    if 1.0 <= dominant_freq <= 10.0:
        f1x = dominant_freq
    else:
        # Scan for the strongest peak in the 1–8 Hz band
        mask = (freqs >= 1.0) & (freqs <= 8.0)
        if np.any(mask):
            idx_peak = np.argmax(mag_z[mask])
            f1x = float(freqs[mask][idx_peak])
        else:
            f1x = 0.0

    amplitude_1x = _find_harmonic_amplitude(freqs, mag_z, f1x) if f1x > 0 else 0.0

    if baseline_amplitude is None:
        return {
            "amplitude_1x": round(amplitude_1x, 6),
            "frequency_1x": round(f1x, 3),
            "ratio_to_baseline": None,
            "severity": "normal",
            "recommendation": (
                "No baseline available for unbalance assessment. "
                "Store this measurement as a baseline for future comparison."
            ),
        }

    ratio = amplitude_1x / baseline_amplitude if baseline_amplitude > 0 else float("inf")

    if ratio < UNBALANCE_NORMAL:
        severity = "normal"
        recommendation = "Unbalance within normal range. No action required."
    elif ratio < UNBALANCE_MODERATE:
        severity = "moderate"
        recommendation = "Moderate unbalance detected. Schedule balancing at next maintenance."
    elif ratio < UNBALANCE_ELEVATED:
        severity = "elevated"
        recommendation = "Elevated unbalance. Plan traction sheave and motor inspection."
    else:
        severity = "high"
        recommendation = "High unbalance — risk of bearing damage. Immediate balancing recommended."

    return {
        "amplitude_1x": round(amplitude_1x, 6),
        "frequency_1x": round(f1x, 3),
        "ratio_to_baseline": round(ratio, 2) if baseline_amplitude > 0 else None,
        "severity": severity,
        "recommendation": recommendation,
    }


def diagnose_misalignment(
    freqs: np.ndarray,
    mag_z: np.ndarray,
    base_freq: float,
) -> dict:
    """Diagnose shaft/coupling misalignment from 2× and 3× harmonic ratios.

    Misalignment typically produces elevated 2× and 3× running-speed harmonics.
    A 2× / 1× amplitude ratio > 0.5 is a strong indicator.

    Args:
        freqs: Frequency bins (Hz) from FFT.
        mag_z: Z-axis magnitude spectrum (m/s²).
        base_freq: The 1× running speed frequency (Hz).

    Returns:
        Dict with keys:
            - ``amplitude_1x``: Magnitude at 1×.
            - ``amplitude_2x``: Magnitude at 2×.
            - ``amplitude_3x``: Magnitude at 3×.
            - ``ratio_2x``: 2× / 1× amplitude ratio.
            - ``ratio_3x``: 3× / 1× amplitude ratio.
            - ``severity``: One of ``SEVERITY_LEVELS``.
            - ``recommendation``: Human-readable recommendation.
    """
    if base_freq <= 0:
        return {
            "amplitude_1x": 0.0,
            "amplitude_2x": 0.0,
            "amplitude_3x": 0.0,
            "ratio_2x": 0.0,
            "ratio_3x": 0.0,
            "severity": "normal",
            "recommendation": "Cannot assess misalignment — no valid 1× frequency identified.",
        }

    amp_1x = _find_harmonic_amplitude(freqs, mag_z, base_freq)
    amp_2x = _find_harmonic_amplitude(freqs, mag_z, base_freq * 2)
    amp_3x = _find_harmonic_amplitude(freqs, mag_z, base_freq * 3)

    ratio_2x = amp_2x / amp_1x if amp_1x > 0 else 0.0
    ratio_3x = amp_3x / amp_1x if amp_1x > 0 else 0.0

    if ratio_2x >= MISALIGNMENT_RATIO_2X_STRONG:
        severity = "elevated"
        recommendation = (
            "Strong misalignment signature (2×/1× ratio ≥ 0.5). "
            "Inspect coupling alignment, shaft straightness, and bearing mounts."
        )
    elif ratio_2x >= MISALIGNMENT_RATIO_2X:
        severity = "moderate"
        recommendation = (
            "Possible misalignment (2×/1× ratio > 0.3). "
            "Monitor trend; check alignment at next scheduled maintenance."
        )
    else:
        severity = "normal"
        recommendation = "No significant misalignment indicators. Alignment appears normal."

    return {
        "amplitude_1x": round(amp_1x, 6),
        "amplitude_2x": round(amp_2x, 6),
        "amplitude_3x": round(amp_3x, 6),
        "ratio_2x": round(ratio_2x, 3),
        "ratio_3x": round(ratio_3x, 3),
        "severity": severity,
        "recommendation": recommendation,
    }


def diagnose_looseness(
    kurtosis_z: float,
    crest_factor_z: float,
    noise_floor_z: float = 0.0,
) -> dict:
    """Diagnose mechanical looseness from excess kurtosis and crest factor.

    Mechanical looseness produces impulsive vibration with a heavy-tailed
    amplitude distribution. Key indicators: excess kurtosis > 4 AND
    crest factor > 5.

    Args:
        kurtosis_z: Excess kurtosis (Fisher definition, normal = 0) of Z-axis.
        crest_factor_z: Crest factor (peak / RMS) of Z-axis.
        noise_floor_z: Z-axis noise floor RMS for context (m/s²).

    Returns:
        Dict with keys:
            - ``kurtosis``: Excess kurtosis value.
            - ``crest_factor``: Crest factor value.
            - ``noise_floor_z``: Z-axis noise floor RMS.
            - ``severity``: One of ``SEVERITY_LEVELS``.
            - ``recommendation``: Human-readable recommendation.
    """
    is_high_kurtosis = kurtosis_z > LOOSENESS_KURTOSIS
    is_high_cf = crest_factor_z > LOOSENESS_CREST_FACTOR

    if is_high_kurtosis and is_high_cf:
        # Both indicators elevated → severity scales with kurtosis
        if kurtosis_z > 10.0:
            severity = "high"
            recommendation = (
                "High mechanical looseness detected. Impulsive vibration may indicate "
                "loose fasteners, guide shoes, or mounting bolts. Immediate inspection required."
            )
        elif kurtosis_z > 7.0:
            severity = "elevated"
            recommendation = (
                "Elevated looseness indicators. Inspect bolted joints, roller guides, "
                "and isolation mounts at next maintenance."
            )
        else:
            severity = "moderate"
            recommendation = (
                "Moderate looseness signature. Check for loose panels, worn guide shoes, "
                "or degraded isolation pads."
            )
    elif is_high_kurtosis or is_high_cf:
        severity = "moderate"
        recommendation = (
            "Partial looseness indicators present. One of kurtosis or crest factor "
            "is elevated. Monitor trend."
        )
    else:
        severity = "normal"
        recommendation = "No looseness indicators detected. Structure appears tight."

    return {
        "kurtosis": round(kurtosis_z, 3),
        "crest_factor": round(crest_factor_z, 2),
        "noise_floor_z": round(noise_floor_z, 6),
        "severity": severity,
        "recommendation": recommendation,
    }


def diagnose_guide_wear(
    x_rms_by_state: dict[str, float] | None = None,
    y_rms_by_state: dict[str, float] | None = None,
) -> dict:
    """Diagnose guide rail wear from position-dependent X/Y axis RMS.

    Guide wear typically manifests as elevated lateral vibration (Y-axis)
    at specific floor positions, caused by uneven rail wear or misaligned
    guide shoes. Full diagnosis requires floor-indexed data across multiple
    trips and is not available from single-trip data alone.

    Args:
        x_rms_by_state: Optional per-state X-axis RMS values (currently unused).
        y_rms_by_state: Optional per-state Y-axis RMS values (currently unused).

    Returns:
        Dict with ``assessment`` key set to ``'insufficient_data'`` and
        a note that multi-floor data is needed.
    """
    return {
        "assessment": "insufficient_data",
        "note": (
            "Guide wear diagnosis requires position-correlated Y-axis RMS data "
            "across multiple floor stops. Single-trip data is insufficient. "
            "Collect 10+ trips with floor-level annotations for trend analysis."
        ),
        "x_rms_by_state": x_rms_by_state or {},
        "y_rms_by_state": y_rms_by_state or {},
    }


def diagnose_bearing_degradation(
    kurtosis_z_trend: list[float],
    crest_factor_trend: list[float] | None = None,
) -> dict:
    """Diagnose bearing degradation from trending kurtosis over multiple trips.

    Bearing degradation produces a monotonic increase in kurtosis as
    spalling or pitting progresses. Three or more consecutive trips with
    monotonically increasing kurtosis is a strong indicator.

    Args:
        kurtosis_z_trend: List of excess kurtosis values from consecutive trips
            (oldest first).
        crest_factor_trend: Optional list of crest factor values (same order).

    Returns:
        Dict with keys:
            - ``trend_detected``: True if monotonic increase over ≥ 3 trips.
            - ``n_increasing``: Length of the current monotonic-increase run at the tail.
            - ``severity``: One of ``SEVERITY_LEVELS``.
            - ``recommendation``: Human-readable recommendation.
    """
    if len(kurtosis_z_trend) < BEARING_TREND_WINDOW:
        return {
            "trend_detected": False,
            "n_increasing": len(kurtosis_z_trend),
            "n_trips": len(kurtosis_z_trend),
            "severity": "normal",
            "recommendation": (
                f"Insufficient trend data ({len(kurtosis_z_trend)} trip(s)). "
                f"Need at least {BEARING_TREND_WINDOW} trips for bearing assessment."
            ),
        }

    # Count consecutive monotonic increases at the tail of the trend
    n_increasing = 1
    for i in range(len(kurtosis_z_trend) - 1, 0, -1):
        if kurtosis_z_trend[i] > kurtosis_z_trend[i - 1]:
            n_increasing += 1
        else:
            break

    trend_detected = n_increasing >= BEARING_TREND_WINDOW

    if trend_detected:
        if n_increasing >= 6:
            severity = "critical"
            recommendation = (
                "Critical bearing degradation trend — monotonic kurtosis increase over "
                f"{n_increasing} consecutive trips. Immediate bearing inspection and "
                "replacement planning required."
            )
        elif n_increasing >= 4:
            severity = "high"
            recommendation = (
                f"Strong bearing degradation trend ({n_increasing} trips). "
                "Schedule bearing inspection and vibration monitoring."
            )
        else:
            severity = "elevated"
            recommendation = (
                f"Emerging bearing degradation trend ({n_increasing} trips). "
                "Increase monitoring frequency and inspect at next maintenance."
            )
    else:
        severity = "normal"
        recommendation = (
            f"No bearing degradation trend detected (consecutive increase: {n_increasing}). "
            "Bearings appear stable."
        )

    # Also check crest factor trend if provided
    cf_trending = False
    if crest_factor_trend is not None and len(crest_factor_trend) >= 3:
        cf_increasing = 1
        for i in range(len(crest_factor_trend) - 1, 0, -1):
            if crest_factor_trend[i] > crest_factor_trend[i - 1]:
                cf_increasing += 1
            else:
                break
        cf_trending = cf_increasing >= BEARING_TREND_WINDOW

    result: dict = {
        "trend_detected": trend_detected,
        "n_increasing": n_increasing,
        "n_trips": len(kurtosis_z_trend),
        "severity": severity,
        "recommendation": recommendation,
    }

    if crest_factor_trend is not None:
        result["cf_trending"] = cf_trending

    return result


# ══════════════════════════════════════════════
#  Master diagnostic report
# ══════════════════════════════════════════════


def full_diagnostic_report(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    t: np.ndarray,
    dt: float,
    state_analysis: StateAnalysis | None = None,
    baseline: dict | None = None,
) -> dict:
    """Run all diagnostics and produce a structured fault detection report.

    This is the master entry point for elevator fault diagnostics. It runs:

    1. Door rattle diagnosis (X-axis cruise vs noise floor).
    2. Unbalance diagnosis (1× peak in Z-axis spectrum).
    3. Misalignment diagnosis (2× and 3× harmonic ratios in Z-axis).
    4. Mechanical looseness diagnosis (kurtosis + crest factor in Z-axis).
    5. Guide wear assessment (stub — requires multi-floor data).
    6. Bearing degradation assessment (requires multi-trip trending data).

    A health score (0–100) is computed by deducting points per fault severity.

    Args:
        ax: X-axis acceleration array (door open/close direction, m/s²).
        ay: Y-axis acceleration array (lateral, m/s²).
        az: Z-axis acceleration array (vertical, includes gravity, m/s²).
        t: Time array in seconds (same length as ax/ay/az).
        dt: Sample interval in seconds.
        state_analysis: Optional ``StateAnalysis`` from
            :func:`elevator_vibration.state_detection.analyze_trip_states`.
            If provided, per-state data is used for door rattle and Z-axis
            RMS context. If None, the whole signal is used.
        baseline: Optional dict with baseline measurements for comparison.
            Expected keys: ``amplitude_1x_z`` (float, for unbalance baseline),
            ``kurtosis_z_trend`` (list[float], for bearing trending),
            ``crest_factor_trend`` (list[float], optional for bearing).

    Returns:
        Structured dict with keys:
            - ``door_rattle``: Result from :func:`diagnose_door_rattle`.
            - ``unbalance``: Result from :func:`diagnose_unbalance`.
            - ``misalignment``: Result from :func:`diagnose_misalignment`.
            - ``looseness``: Result from :func:`diagnose_looseness`.
            - ``guide_wear``: Result from :func:`diagnose_guide_wear`.
            - ``bearing_degradation``: Result from :func:`diagnose_bearing_degradation`.
            - ``health_score``: Integer 0–100.
            - ``health_summary``: Overall condition summary string.
            - ``faults_active``: List of fault names with severity > normal.
            - ``metadata``: Diagnostic metadata (timestamps, n_samples, etc.).
    """
    from elevator_vibration.preprocessing import separate_gravity

    # ── Gravity removal ──────────────────────────────────────────
    az_dynamic, gravity = separate_gravity(az)

    n = len(ax)
    meta: dict = {
        "n_samples": n,
        "duration_s": round(float(t[-1] - t[0]), 2),
        "sample_rate_hz": round(1.0 / dt, 1),
        "gravity_estimate": round(gravity, 4),
    }

    # ── State-aware segment extraction ───────────────────────────
    if state_analysis is not None:
        # Extract cruise data for door rattle
        cruise_codes = {2, 5}  # cruise_up=2, cruise_down=5
        cruise_mask = np.isin(state_analysis.state_codes, list(cruise_codes))
        ax_cruise = ax[cruise_mask]

        # Standing data for noise floor
        standing_mask = state_analysis.state_codes == 0  # standing
        if np.any(standing_mask):
            noise_floor_x = float(np.sqrt(np.mean(ax[standing_mask] ** 2)))
            noise_floor_z = float(np.sqrt(np.mean(az_dynamic[standing_mask] ** 2)))
        else:
            noise_floor_x = float(np.sqrt(np.mean(ax**2)))
            noise_floor_z = float(np.sqrt(np.mean(az_dynamic**2)))

        # Per-state RMS for guide wear stub
        x_rms_by_state: dict[str, float] = {}
        y_rms_by_state: dict[str, float] = {}
        from elevator_vibration.state_detection import STATE_LABELS
        valid_states = set(state_analysis.state_codes)
        for sc in valid_states:
            mask = state_analysis.state_codes == sc
            if np.any(mask):
                label = STATE_LABELS.get(int(sc), f"state_{sc}")
                x_rms_by_state[label] = float(np.sqrt(np.mean(ax[mask] ** 2)))
                y_rms_by_state[label] = float(np.sqrt(np.mean(ay[mask] ** 2)))
    else:
        # No state analysis: use whole signal
        ax_cruise = ax
        noise_floor_x = float(np.sqrt(np.mean(ax**2)))
        noise_floor_z = float(np.sqrt(np.mean(az_dynamic**2)))
        x_rms_by_state = {}
        y_rms_by_state = {}

    # ── Z-axis time-domain stats ─────────────────────────────────
    kurtosis_z = compute_kurtosis(az_dynamic, fisher=True)  # excess kurtosis
    crest_factor_z = compute_crest_factor(az_dynamic)
    rms_z = compute_rms(az_dynamic)

    # ── Z-axis frequency analysis ────────────────────────────────
    freqs, mag_z = compute_fft(az_dynamic, fs=1.0 / dt)
    dom_freq = dominant_frequency(freqs, mag_z, low=0.5, high=50.0)

    # Determine 1× running speed for unbalance/misalignment
    # Use dominant frequency if it's in the plausible elevator range
    if 1.0 <= dom_freq <= 10.0:
        base_freq = dom_freq
    else:
        # Scan for strongest peak in 1–8 Hz band
        mask = (freqs >= 1.0) & (freqs <= 8.0)
        if np.any(mask):
            idx_peak = int(np.argmax(mag_z[mask]))
            base_freq = float(freqs[mask][idx_peak])
        else:
            base_freq = 0.0

    meta["dominant_freq_z_hz"] = round(dom_freq, 3)
    meta["base_freq_hz"] = round(base_freq, 3)

    # ── Baseline handling ────────────────────────────────────────
    baseline_amp = baseline.get("amplitude_1x_z") if baseline else None
    kurtosis_z_trend = baseline.get("kurtosis_z_trend", [kurtosis_z]) if baseline else [kurtosis_z]
    crest_factor_trend = baseline.get("crest_factor_trend") if baseline else None

    # ── Run individual diagnostics ───────────────────────────────
    diag_door_rattle = diagnose_door_rattle(ax_cruise, noise_floor_x)
    diag_unbalance = diagnose_unbalance(freqs, mag_z, base_freq, baseline_amp)
    diag_misalignment = diagnose_misalignment(freqs, mag_z, base_freq)
    diag_looseness = diagnose_looseness(kurtosis_z, crest_factor_z, noise_floor_z)
    diag_guide_wear = diagnose_guide_wear(x_rms_by_state, y_rms_by_state)
    diag_bearing = diagnose_bearing_degradation(kurtosis_z_trend, crest_factor_trend)

    # ── Health score calculation ─────────────────────────────────
    health_score = 100
    faults_active: list[str] = []

    for fault_name, result in [
        ("door_rattle", diag_door_rattle),
        ("unbalance", diag_unbalance),
        ("misalignment", diag_misalignment),
        ("looseness", diag_looseness),
        ("bearing_degradation", diag_bearing),
    ]:
        sev = result.get("severity", "normal")
        health_score -= HEALTH_DEDUCTION.get(sev, 0)
        if sev != "normal":
            faults_active.append(f"{fault_name}:{sev}")

    health_score = max(0, min(100, health_score))

    if health_score >= 90:
        health_summary = "Good — no significant faults detected."
    elif health_score >= 75:
        health_summary = "Fair — minor faults present; monitor at next maintenance."
    elif health_score >= 50:
        health_summary = "Degraded — multiple faults detected; schedule inspection."
    elif health_score >= 25:
        health_summary = "Poor — significant faults; prioritize maintenance."
    else:
        health_summary = "Critical — immediate inspection required."

    return {
        "door_rattle": diag_door_rattle,
        "unbalance": diag_unbalance,
        "misalignment": diag_misalignment,
        "looseness": diag_looseness,
        "guide_wear": diag_guide_wear,
        "bearing_degradation": diag_bearing,
        "health_score": health_score,
        "health_summary": health_summary,
        "faults_active": faults_active,
        "metadata": meta,
    }
