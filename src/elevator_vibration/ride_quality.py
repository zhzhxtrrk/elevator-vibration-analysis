"""Ride comfort assessment per ISO 2631-1 and ISO 18738.

NOTE — Door-Mounted Sensor Context
    The accelerometer is mounted on the elevator door panel.
    - X-axis = door open/close direction → heavily contaminated by door rattle.
    - Y-axis = lateral → moderately contaminated by door artifacts.
    - Z-axis = vertical → cleanest axis, least affected by door noise.
    For door-mounted sensors, prefer Z-only assessment modes
    (``iso2631_z_only``, ``iso18738_z_only``, or ``iso2631_assessment(z_only=True)``)
    as X/Y axes can severely overestimate ride harshness.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sig


# ══════════════════════════════════════════════
#  ISO 2631-1 Frequency Weighting Filters
# ══════════════════════════════════════════════


def _butterworth_sos(
    btype: str,
    freqs: list[float],
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Build Butterworth SOS filter. freqs should be a list even for single-cutoff types."""
    nyq = fs / 2.0
    wn = [f / nyq for f in freqs]
    if btype in ("highpass", "lowpass"):
        wn = wn[0]  # scalar expected by scipy
    return sig.butter(order, wn, btype=btype, output="sos")


def weighting_wk(fs: float = 100.0) -> np.ndarray:
    """ISO 2631-1 Wk weighting (vertical, z-axis) as SOS coefficients.

    Band-pass + transition: high-pass ~0.4 Hz, low-pass ~100 Hz,
    with emphasis between ~4–12.5 Hz.

    This is a simplified IIR approximation of the Wk curve.
    Cutoffs clamped to Nyquist (fs/2) - 1 Hz.
    """
    nyq = fs / 2.0
    sos_hp = _butterworth_sos("highpass", [0.4], fs, order=2)
    sos_bp = _butterworth_sos("bandpass", [0.5, min(80.0, nyq - 1)], fs, order=2)
    sos_lp = _butterworth_sos("lowpass", [min(100.0, nyq - 1)], fs, order=2)
    return np.vstack([sos_hp, sos_bp, sos_lp])


def weighting_wd(fs: float = 100.0) -> np.ndarray:
    """ISO 2631-1 Wd weighting (horizontal, x/y axes) as SOS coefficients.

    Horizontal sensitivity peaks around 1–2 Hz.
    """
    nyq = fs / 2.0
    sos_hp = _butterworth_sos("highpass", [0.2], fs, order=2)
    sos_bp = _butterworth_sos("bandpass", [0.3, min(40.0, nyq - 1)], fs, order=2)
    sos_lp = _butterworth_sos("lowpass", [min(50.0, nyq - 1)], fs, order=2)
    return np.vstack([sos_hp, sos_bp, sos_lp])


def apply_frequency_weighting(
    signal: np.ndarray,
    fs: float = 100.0,
    axis: str = "z",
) -> np.ndarray:
    """Apply ISO 2631-1 frequency weighting to a signal.

    Args:
        signal: 1-D acceleration signal (m/s²).
        fs: Sampling frequency (Hz).
        axis: 'z' for vertical (Wk), 'x' or 'y' for horizontal (Wd).

    Returns:
        Frequency-weighted signal.
    """
    if axis in ("z", "Z"):
        sos = weighting_wk(fs)
    elif axis in ("x", "y", "X", "Y"):
        sos = weighting_wd(fs)
    else:
        raise ValueError(f"Unknown axis '{axis}'; use 'x', 'y', or 'z'.")

    return sig.sosfiltfilt(sos, signal)


# ══════════════════════════════════════════════
#  ISO 2631-1 Metrics
# ══════════════════════════════════════════════


def compute_weighted_rms(
    signal: np.ndarray,
    fs: float = 100.0,
    axis: str = "z",
) -> float:
    """ISO 2631-1 frequency-weighted RMS acceleration (m/s²).

    This is the primary metric for comfort assessment.

    .. note:: Door-mounted sensor context
        X/Y axes on a door panel carry door rattle and mechanical noise.
        Use ``axis='z'`` (vertical) for the cleanest signal, or prefer
        the Z-only assessment functions for ride comfort evaluation.
    """
    weighted = apply_frequency_weighting(signal - np.mean(signal), fs, axis)
    return float(np.sqrt(np.mean(weighted**2)))


def compute_vdv(
    signal: np.ndarray,
    fs: float = 100.0,
    axis: str = "z",
) -> float:
    """Vibration Dose Value (VDV) per ISO 2631-1.

    VDV = (∫ a_w⁴ dt)^{1/4}  [m/s^{1.75}]

    More sensitive to peaks than RMS. Useful when crest factor > 9.

    .. note:: Door-mounted sensor context
        X/Y axes carry door panel vibration artifacts.
        For door-mounted sensors, evaluate VDV only on the Z-axis
        to avoid overestimating ride harshness.
    """
    weighted = apply_frequency_weighting(signal - np.mean(signal), fs, axis)
    dt = 1.0 / fs
    vdv = (np.sum(weighted**4) * dt) ** 0.25
    return float(vdv)


def compute_mtvv(
    signal: np.ndarray,
    fs: float = 100.0,
    axis: str = "z",
    integration_time: float = 1.0,
) -> float:
    """Maximum Transient Vibration Value (MTVV) per ISO 2631-1.

    MTVV is the maximum of the running RMS with a 1-second integration window.

    .. note:: Door-mounted sensor context
        Peaks on X/Y axes may reflect door panel rattling rather than
        true ride harshness. Prefer Z-axis MTVV for door-mounted sensors.
    """
    weighted = apply_frequency_weighting(signal - np.mean(signal), fs, axis)
    n_window = int(integration_time * fs)
    if n_window < 2:
        n_window = 2

    # Running RMS via convolution
    squared = weighted**2
    window = np.ones(n_window) / n_window
    running_rms = np.sqrt(np.convolve(squared, window, mode="valid"))
    return float(np.max(running_rms))


def compute_crest_factor(signal: np.ndarray) -> float:
    """Crest factor = peak / RMS. ISO 2631-1 recommends VDV when CF > 9."""
    rms = np.sqrt(np.mean(signal**2))
    if rms < 1e-12:
        return 0.0
    return float(np.max(np.abs(signal)) / rms)


def iso2631_assessment(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    fs: float = 100.0,
    z_only: bool = False,
) -> dict:
    """Full ISO 2631-1 comfort assessment.

    .. warning:: Door-mounted sensor context
        X/Y axes on a door panel carry door rattle and mechanical noise
        that can severely overestimate ride harshness. Use ``z_only=True``
        (or the dedicated ``iso2631_z_only`` function) for door-mounted
        sensors — it evaluates only the clean Z (vertical) axis and sets
        X/Y metrics to zero.

    Args:
        ax, ay, az: Triaxial acceleration arrays (m/s²).
        fs: Sampling frequency (Hz).
        z_only: If True, ignore X/Y axes (set to 0) and use only Z-axis.
            Recommended for door-mounted sensors.

    Returns a dict with weighted RMS, VDV, MTVV per axis, plus
    the overall vibration total value and a comfort classification.
    """
    # Frequency-weighted RMS per axis (with correct axis multiplier)
    if z_only:
        aw_x = 0.0
        aw_y = 0.0
        aw_z = compute_weighted_rms(az, fs, axis="z") * 1.0
        vdv_x = 0.0
        vdv_y = 0.0
        vdv_z = compute_vdv(az, fs, axis="z")
        mtvv_x = 0.0
        mtvv_y = 0.0
        mtvv_z = compute_mtvv(az, fs, axis="z")
        cf_x = 0.0
        cf_y = 0.0
        cf_z = compute_crest_factor(az)
    else:
        aw_x = compute_weighted_rms(ax, fs, axis="x") * 1.4  # ISO 2631-1 multiplier for x
        aw_y = compute_weighted_rms(ay, fs, axis="y") * 1.4  # ISO 2631-1 multiplier for y
        aw_z = compute_weighted_rms(az, fs, axis="z") * 1.0  # No multiplier for z

        # VDV per axis
        vdv_x = compute_vdv(ax, fs, axis="x")
        vdv_y = compute_vdv(ay, fs, axis="y")
        vdv_z = compute_vdv(az, fs, axis="z")

        # MTVV per axis
        mtvv_x = compute_mtvv(ax, fs, axis="x")
        mtvv_y = compute_mtvv(ay, fs, axis="y")
        mtvv_z = compute_mtvv(az, fs, axis="z")

        # Crest factor check
        cf_x = compute_crest_factor(ax)
        cf_y = compute_crest_factor(ay)
        cf_z = compute_crest_factor(az)

    # Vibration total value
    av = float(np.sqrt(aw_x**2 + aw_y**2 + aw_z**2))

    return {
        "aw_x": aw_x,
        "aw_y": aw_y,
        "aw_z": aw_z,
        "av_total": av,
        "vdv_x": vdv_x,
        "vdv_y": vdv_y,
        "vdv_z": vdv_z,
        "mtvv_x": mtvv_x,
        "mtvv_y": mtvv_y,
        "mtvv_z": mtvv_z,
        "crest_factor_x": cf_x,
        "crest_factor_y": cf_y,
        "crest_factor_z": cf_z,
        "use_vdv": cf_x > 9.0 or cf_y > 9.0 or cf_z > 9.0,
        "comfort_class": _comfort_classification(av),
        "likely_uncomfortable": av > 0.315,
        "likely_very_uncomfortable": av > 0.5,
    }


def _comfort_classification(av: float) -> str:
    """Map vibration total value to ISO 2631-1 comfort category."""
    if av < 0.315:
        return "not uncomfortable"
    elif av < 0.5:
        return "a little uncomfortable"
    elif av < 0.8:
        return "fairly uncomfortable"
    elif av < 1.25:
        return "uncomfortable"
    elif av < 2.5:
        return "very uncomfortable"
    else:
        return "extremely uncomfortable"


def iso2631_z_only(az: np.ndarray, fs: float = 100.0) -> dict:
    """ISO 2631-1 ride comfort assessment using only the Z (vertical) axis.

    This is the **recommended mode for door-mounted sensors**, where
    X/Y axes are contaminated by door panel rattle and mechanical noise.
    X/Y metrics are set to 0.0 and the vibration total value (av_total)
    is based solely on the clean Z-axis signal.

    Args:
        az: Z-axis (vertical) acceleration array (m/s²).
        fs: Sampling frequency (Hz).

    Returns:
        Same dict format as ``iso2631_assessment``, with X/Y metrics zeroed out.
    """
    return iso2631_assessment(
        ax=np.zeros_like(az),
        ay=np.zeros_like(az),
        az=az,
        fs=fs,
        z_only=True,
    )


# ══════════════════════════════════════════════
#  ISO 18738 Elevator Ride Quality Metrics
# ══════════════════════════════════════════════


def compute_a95(signal: np.ndarray) -> float:
    """A95: 95th percentile of the absolute acceleration (ISO 18738).

    This is a standard metric for peak elevator vibration.
    """
    return float(np.percentile(np.abs(signal), 95))


def compute_peak_acceleration(signal: np.ndarray) -> float:
    """Peak absolute acceleration during the run (ISO 18738)."""
    return float(np.max(np.abs(signal)))


def compute_jerk_peak(signal: np.ndarray, dt: float = 0.01) -> float:
    """Peak jerk (m/s³) during the run. ISO 18738 jerk metric."""
    jerk = np.zeros_like(signal)
    jerk[1:-1] = (signal[2:] - signal[:-2]) / (2 * dt)
    jerk[0] = (signal[1] - signal[0]) / dt
    jerk[-1] = (signal[-1] - signal[-2]) / dt
    return float(np.max(np.abs(jerk)))


def compute_a95_jerk(signal: np.ndarray, dt: float = 0.01) -> float:
    """A95 of jerk (95th percentile absolute jerk)."""
    jerk = np.zeros_like(signal)
    jerk[1:-1] = (signal[2:] - signal[:-2]) / (2 * dt)
    jerk[0] = (signal[1] - signal[0]) / dt
    jerk[-1] = (signal[-1] - signal[-2]) / dt
    return float(np.percentile(np.abs(jerk), 95))


def iso18738_metrics(
    ax: np.ndarray,
    ay: np.ndarray,
    az_dynamic: np.ndarray,
    time_seconds: np.ndarray,
    dt: float = 0.01,
    run_intervals: list[dict] | None = None,
) -> dict:
    """Compute ISO 18738 elevator ride quality metrics.

    Args:
        ax, ay: Lateral acceleration (m/s²).
        az_dynamic: Vertical acceleration with gravity removed (m/s²).
        time_seconds: Timestamp array.
        dt: Sampling interval.
        run_intervals: Optional list of dicts with start_idx/end_idx for
            individual runs. If None, analyzes the entire signal.

    Returns:
        Dict with peak, A95, jerk metrics, and per-run breakdown.
    """
    composite = np.sqrt(ax**2 + ay**2 + az_dynamic**2)

    if run_intervals is None:
        # Analyze entire signal
        run_intervals = [{"start_idx": 0, "end_idx": len(ax) - 1, "label": "full"}]

    runs_metrics = []
    for i, run in enumerate(run_intervals):
        s, e = run["start_idx"], run["end_idx"] + 1
        runs_metrics.append(
            {
                "run": i,
                "label": run.get("label", f"run_{i}"),
                "start_time": float(time_seconds[s]),
                "end_time": float(time_seconds[e - 1]),
                "duration": float(time_seconds[e - 1] - time_seconds[s]),
                "ax_peak": compute_peak_acceleration(ax[s:e]),
                "ay_peak": compute_peak_acceleration(ay[s:e]),
                "az_peak": compute_peak_acceleration(az_dynamic[s:e]),
                "composite_peak": compute_peak_acceleration(composite[s:e]),
                "ax_a95": compute_a95(ax[s:e]),
                "ay_a95": compute_a95(ay[s:e]),
                "az_a95": compute_a95(az_dynamic[s:e]),
                "composite_a95": compute_a95(composite[s:e]),
                "jerk_peak": compute_jerk_peak(az_dynamic[s:e], dt),
                "jerk_a95": compute_a95_jerk(az_dynamic[s:e], dt),
            }
        )

    # Global metrics (full signal)
    result: dict = {
        "n_runs": len(run_intervals),
        "runs": runs_metrics,
        "global_composite_peak": compute_peak_acceleration(composite),
        "global_composite_a95": compute_a95(composite),
        "global_jerk_peak": compute_jerk_peak(az_dynamic, dt),
        "global_jerk_a95": compute_a95_jerk(az_dynamic, dt),
    }

    # ISO 18738 pass/fail thresholds (example values; actual depend on contract)
    # Typical criteria: A95 < 0.15 m/s², peak jerk < 2.5 m/s³
    result["a95_pass"] = result["global_composite_a95"] < 0.15
    result["jerk_pass"] = result["global_jerk_peak"] < 2.5

    return result


def iso18738_z_only(
    az_dynamic: np.ndarray,
    time_seconds: np.ndarray,
    dt: float = 0.01,
    run_intervals: list[dict] | None = None,
) -> dict:
    """ISO 18738 elevator ride quality using only the Z (vertical) axis.

    This is the **recommended mode for door-mounted sensors**. X and Y axes
    on a door panel carry door rattle and mechanical artifacts that would
    contaminate composite metrics. This function skips X/Y entirely —
    composite values equal Z-axis values.

    Args:
        az_dynamic: Vertical acceleration with gravity removed (m/s²).
        time_seconds: Timestamp array.
        dt: Sampling interval.
        run_intervals: Optional list of dicts with start_idx/end_idx for
            individual runs. If None, analyzes the entire signal.

    Returns:
        Same dict format as ``iso18738_metrics``, with X/Y metrics zeroed
        and composite values reflecting only the Z-axis.
    """
    # Use zero arrays for X and Y to eliminate door-panel contamination
    n = len(az_dynamic)
    dummy = np.zeros(n, dtype=az_dynamic.dtype)

    return iso18738_metrics(
        ax=dummy,
        ay=dummy,
        az_dynamic=az_dynamic,
        time_seconds=time_seconds,
        dt=dt,
        run_intervals=run_intervals,
    )
