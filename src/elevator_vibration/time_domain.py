"""Time-domain analysis: statistical features, event detection, run segmentation."""

from __future__ import annotations

import numpy as np
from scipy import signal as sig
from scipy.stats import kurtosis as scipy_kurtosis


# ──────────────────────────────────────────────
#  Basic statistical features
# ──────────────────────────────────────────────


def compute_rms(signal: np.ndarray) -> float:
    """Root-mean-square of a signal."""
    return float(np.sqrt(np.mean(signal**2)))


def compute_peak(signal: np.ndarray) -> float:
    """Peak absolute value (max amplitude)."""
    return float(np.max(np.abs(signal)))


def compute_peak_to_peak(signal: np.ndarray) -> float:
    """Peak-to-peak amplitude."""
    return float(np.max(signal) - np.min(signal))


def compute_crest_factor(signal: np.ndarray) -> float:
    """Crest factor = peak / RMS. High values indicate impulsive content."""
    rms = compute_rms(signal)
    if rms < 1e-12:
        return 0.0
    return compute_peak(signal) / rms


def compute_kurtosis(signal: np.ndarray, fisher: bool = True) -> float:
    """Excess kurtosis (fisher=True subtracts 3, giving 0 for normal dist).
    High kurtosis suggests bearing faults or impulsive vibration."""
    return float(scipy_kurtosis(signal, fisher=fisher))


def compute_skewness(signal: np.ndarray) -> float:
    """Skewness of the amplitude distribution."""
    from scipy.stats import skew

    return float(skew(signal))


# ──────────────────────────────────────────────
#  Jerk (derivative of acceleration)
# ──────────────────────────────────────────────


def compute_jerk(
    acceleration: np.ndarray,
    dt: float = 0.01,
) -> np.ndarray:
    """Numerical derivative of acceleration (m/s³). Central difference,
    with forward/backward difference at endpoints."""
    jerk = np.zeros_like(acceleration)
    jerk[1:-1] = (acceleration[2:] - acceleration[:-2]) / (2 * dt)
    jerk[0] = (acceleration[1] - acceleration[0]) / dt
    jerk[-1] = (acceleration[-1] - acceleration[-2]) / dt
    return jerk


def compute_jerk_rms(acceleration: np.ndarray, dt: float = 0.01) -> float:
    """RMS of jerk (m/s³)."""
    return compute_rms(compute_jerk(acceleration, dt))


def compute_jerk_peak(acceleration: np.ndarray, dt: float = 0.01) -> float:
    """Peak jerk magnitude (m/s³)."""
    return compute_peak(compute_jerk(acceleration, dt))


# ──────────────────────────────────────────────
#  Per-axis statistical summary
# ──────────────────────────────────────────────


def axis_statistics(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    dt: float = 0.01,
) -> dict:
    """Compute per-axis time-domain statistics.

    Returns a dict with keys like 'ax_rms', 'ay_peak', 'az_kurtosis', etc.
    """
    axes = {"ax": ax, "ay": ay, "az": az}
    stats: dict[str, float] = {}
    for name, signal in axes.items():
        stats[f"{name}_rms"] = compute_rms(signal)
        stats[f"{name}_peak"] = compute_peak(signal)
        stats[f"{name}_peak_to_peak"] = compute_peak_to_peak(signal)
        stats[f"{name}_crest_factor"] = compute_crest_factor(signal)
        stats[f"{name}_kurtosis"] = compute_kurtosis(signal)
        stats[f"{name}_skewness"] = compute_skewness(signal)
        stats[f"{name}_jerk_rms"] = compute_jerk_rms(signal, dt)
        stats[f"{name}_jerk_peak"] = compute_jerk_peak(signal, dt)
    return stats


# ──────────────────────────────────────────────
#  Event / transient detection
# ──────────────────────────────────────────────


def detect_events(
    signal: np.ndarray,
    threshold: float | None = None,
    n_std: float = 3.0,
    min_gap_samples: int = 10,
) -> np.ndarray:
    """Detect events where |signal| exceeds threshold.

    Args:
        signal: 1-D array.
        threshold: Absolute threshold. If None, threshold = n_std * std(signal).
        n_std: Multiplier for adaptive threshold (ignored if threshold is given).
        min_gap_samples: Merge events separated by fewer than this many samples.

    Returns:
        Boolean array of same length, True where event is active.
    """
    if threshold is None:
        threshold = n_std * np.std(signal)

    above = np.abs(signal) > threshold

    # Merge close events
    if min_gap_samples > 0:
        gap = np.diff(np.where(above)[0])
        if len(gap) > 0:
            for idx in np.where((gap > 1) & (gap <= min_gap_samples))[0]:
                start = np.where(above)[0][idx] + 1
                end = np.where(above)[0][idx + 1]
                above[start:end] = True

    return above


def event_boundaries(events: np.ndarray) -> list[tuple[int, int]]:
    """Extract (start_idx, end_idx) pairs from boolean event array."""
    if not np.any(events):
        return []
    edges = np.diff(np.concatenate(([False], events, [False])).astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return list(zip(starts, ends))


# ──────────────────────────────────────────────
#  Run segmentation (basic velocity-based)
# ──────────────────────────────────────────────


def segment_by_velocity(
    az_dynamic: np.ndarray,
    dt: float = 0.01,
    velocity_threshold: float = 0.05,
    min_still_samples: int = 50,
) -> np.ndarray:
    """Segment into 'moving' vs 'still' based on integrated velocity from az_dynamic.

    Returns an integer array: 0 = still, 1 = moving.
    """
    # Integrate vertical acceleration to get velocity
    velocity = np.cumsum(az_dynamic) * dt
    # Detrend (remove drift from integration)
    velocity = velocity - np.polyval(np.polyfit(np.arange(len(velocity)), velocity, 1), np.arange(len(velocity)))

    speed = np.abs(velocity)
    moving = speed > velocity_threshold

    # Clean up short flickers
    if min_still_samples > 0:
        for _ in range(2):  # two passes: close short stills, then short moves
            moving = _close_short_gaps(moving, min_still_samples)

    return moving.astype(int)


def _close_short_gaps(mask: np.ndarray, min_samples: int) -> np.ndarray:
    """Fill gaps (False runs) shorter than min_samples."""
    result = mask.copy()
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    # Fill gaps between runs
    for i in range(len(ends) - 1):
        gap_len = starts[i + 1] - ends[i]
        if gap_len < min_samples:
            result[ends[i] : starts[i + 1]] = True

    # Remove short runs
    for s, e in zip(starts, ends):
        if (e - s) < min_samples:
            result[s:e] = False

    return result


def extract_run_intervals(
    moving_mask: np.ndarray,
    time_seconds: np.ndarray,
    min_duration: float = 1.0,
    dt: float = 0.01,
) -> list[dict]:
    """Extract elevator run intervals (start/end times and indices) from a moving mask.

    Returns list of dicts with keys: start_idx, end_idx, start_time, end_time, duration.
    """
    boundaries = event_boundaries(moving_mask.astype(bool))
    intervals = []
    for start_idx, end_idx in boundaries:
        duration = time_seconds[end_idx - 1] - time_seconds[start_idx]
        if duration < min_duration:
            continue
        intervals.append(
            {
                "start_idx": int(start_idx),
                "end_idx": int(end_idx - 1),
                "start_time": float(time_seconds[start_idx]),
                "end_time": float(time_seconds[end_idx - 1]),
                "duration": duration,
            }
        )
    return intervals


# ──────────────────────────────────────────────
#  VibrationRecord convenience wrapper
# ──────────────────────────────────────────────


def time_domain_report(
    ax: np.ndarray,
    ay: np.ndarray,
    az_dynamic: np.ndarray,
    time_seconds: np.ndarray,
    dt: float | None = None,
) -> dict:
    """Full time-domain analysis report.

    Args:
        ax, ay: Lateral acceleration arrays (m/s²).
        az_dynamic: Vertical acceleration with gravity removed (m/s²).
        time_seconds: Timestamp array (seconds, relative or absolute).
        dt: Sampling interval (seconds). Auto-computed if None.

    Returns:
        Dict with axis statistics, composite stats, and run segmentation.
    """
    if dt is None:
        dt = float(np.median(np.diff(time_seconds)))
    composite = np.sqrt(ax**2 + ay**2 + az_dynamic**2)

    report: dict = {
        "n_samples": len(ax),
        "duration": float(time_seconds[-1] - time_seconds[0]),
        "sample_rate_actual": float(1.0 / np.median(np.diff(time_seconds))),
        "axis_stats": axis_statistics(ax, ay, az_dynamic, dt),
        "composite_rms": compute_rms(composite),
        "composite_peak": compute_peak(composite),
        "composite_crest_factor": compute_crest_factor(composite),
        "composite_kurtosis": compute_kurtosis(composite),
    }

    # Run segmentation
    moving = segment_by_velocity(az_dynamic, dt)
    runs = extract_run_intervals(moving, time_seconds, dt=dt)
    report["n_runs"] = len(runs)
    report["runs"] = runs
    report["moving_mask"] = moving

    return report
