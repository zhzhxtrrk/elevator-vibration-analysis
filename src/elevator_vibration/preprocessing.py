"""Signal preprocessing: DC removal, gravity separation, quality checks.

Sensor context: Accelerometer is mounted on the elevator DOOR PANEL (single side).
- X-axis = door open/close direction (through doorway)
- Y-axis = lateral along door plane (side-to-side)
- Z-axis = vertical (gravity-aligned)

The door panel may tilt during elevator motion, introducing bias in
whole-signal gravity estimates. Functions support stationary masking
for more robust gravity separation and noise floor estimation.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sig


def remove_dc(data: np.ndarray) -> np.ndarray:
    """Subtract mean from signal."""
    return data - np.mean(data)


def separate_gravity(
    az: np.ndarray, stationary_mask: np.ndarray | None = None
) -> tuple[np.ndarray, float]:
    """Separate static gravity (~1g) from dynamic vertical acceleration.

    Returns (az_dynamic, gravity_estimate). By default, gravity is estimated
    as the mean of the entire signal (assumes symmetric up/down trips). When
    ``stationary_mask`` is provided, only stationary (door-closed, elevator-parked)
    segments are used, avoiding bias from door-panel tilt during motion.
    """
    if stationary_mask is not None and np.any(stationary_mask):
        g_est = float(np.mean(az[stationary_mask]))
    else:
        g_est = float(np.mean(az))
    return az - g_est, g_est

def estimate_gravity_from_stationary(
    az: np.ndarray, stationary_mask: np.ndarray | None = None
) -> tuple[float, np.ndarray]:
    """Estimate gravity using only stationary (door-closed, elevator-parked) segments.

    Door-panel-mounted accelerometers may experience tilt during elevator motion,
    biasing whole-signal gravity estimates. This function uses only stationary
    segments to recover a clean gravity reference.

    Args:
        az: Vertical acceleration array (Z-axis).
        stationary_mask: Boolean mask where True marks stationary samples.
            If None, defaults to first 5% and last 5% of the signal.

    Returns:
        (gravity_estimate, stationary_mask_used) — the mask used is returned
        so callers can reuse it for other operations.
    """
    if stationary_mask is None:
        n = len(az)
        stationary_mask = np.zeros(n, dtype=bool)
        stationary_mask[: max(1, n // 20)] = True
        stationary_mask[-max(1, n // 20) :] = True
    g_est = float(np.mean(az[stationary_mask]))
    return g_est, stationary_mask


def compute_composite(ax: np.ndarray, ay: np.ndarray, az_dynamic: np.ndarray) -> np.ndarray:
    """Total acceleration magnitude (removes gravity from Z)."""
    return np.sqrt(ax**2 + ay**2 + az_dynamic**2)


def check_clipping(
    signal: np.ndarray,
    max_range: float | None = None,
    threshold_pct: float = 0.95,
) -> dict:
    """Check for signal clipping (samples near max range)."""
    if max_range is None:
        max_range = np.max(np.abs(signal)) * 1.1  # heuristic
    near_clip = np.abs(signal) > (max_range * threshold_pct)
    return {
        "clipped_pct": np.mean(near_clip) * 100,
        "near_clip_count": int(np.sum(near_clip)),
        "max_range": max_range,
    }


def estimate_noise_floor(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    stationary_mask: np.ndarray | None = None,
) -> dict:
    """Measure RMS noise in stationary segments.

    During cruise, door-panel vibration may elevate apparent noise.
    Stationary (door-closed, elevator-parked) segments give the
    true sensor noise floor. If no mask is provided, the first and
    last 10% of the signal are used as a heuristic for parked segments.
    """
    if stationary_mask is None:
        # Use first 10% and last 10% as stationary (elevator parked)
        n = len(ax)
        mask = np.zeros(n, dtype=bool)
        mask[: n // 10] = True
        mask[-n // 10 :] = True
        stationary_mask = mask
    return {
        "ax_noise_rms": float(np.std(ax[stationary_mask])),
        "ay_noise_rms": float(np.std(ay[stationary_mask])),
        "az_noise_rms": float(np.std(az[stationary_mask])),
    }

def door_operation_mask(
    ax: np.ndarray, threshold_mg: float = 200.0
) -> np.ndarray:
    """Detect door open/close events from X-axis acceleration spikes.

    In the door-panel-mounted sensor frame, the X-axis points through the
    doorway. Door operation produces characteristic >``threshold_mg``
    (default 200 mg) spikes in X lasting 1–3 seconds. This function returns
    a boolean mask marking samples where door operation is detected.

    Args:
        ax: X-axis acceleration array (door open/close direction).
        threshold_mg: Threshold in milli-g for spike detection.

    Returns:
        Boolean array, True where door operation is detected.
    """
    threshold_mg = float(threshold_mg)
    return np.abs(ax) > threshold_mg


def compute_sampling_rate(timestamps: np.ndarray) -> float:
    """Estimate actual sampling rate from timestamp array (seconds)."""
    diffs = np.diff(timestamps)
    return float(1.0 / np.median(diffs))


def check_timestamp_gaps(
    timestamps: np.ndarray,
    expected_dt: float | None = None,
) -> dict:
    """Detect gaps in timestamp sequence."""
    diffs = np.diff(timestamps)
    if expected_dt is None:
        expected_dt = np.median(diffs)
    gaps = diffs > (expected_dt * 1.5)
    return {
        "n_gaps": int(np.sum(gaps)),
        "max_gap": float(np.max(diffs)) if len(diffs) > 0 else 0,
        "expected_dt": float(expected_dt),
    }
