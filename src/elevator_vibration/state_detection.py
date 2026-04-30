"""
Elevator operational state detection and run segmentation.

Detects per-sample states from triaxial accelerometer data:
  STANDING → ACCEL_UP/DOWN → CRUISE_UP/DOWN → DECEL_UP/DOWN → STANDING

Core approach:
  1. Velocity integration with drift correction (zero-anchored at standing segments)
  2. State classification via velocity + acceleration heuristics
  3. Phase extraction and statistics
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid


# ──────────────────────────────────────────────
#  State encoding
# ──────────────────────────────────────────────

STATE_STANDING = 0
STATE_ACCEL_UP = 1
STATE_CRUISE_UP = 2
STATE_DECEL_UP = 3
STATE_ACCEL_DOWN = 4
STATE_CRUISE_DOWN = 5
STATE_DECEL_DOWN = 6
STATE_DOOR_OPERATION = 9
STATE_UNKNOWN = -1

STATE_LABELS: dict[int, str] = {
    STATE_STANDING: "standing",
    STATE_ACCEL_UP: "accel_up",
    STATE_CRUISE_UP: "cruise_up",
    STATE_DECEL_UP: "decel_up",
    STATE_ACCEL_DOWN: "accel_down",
    STATE_CRUISE_DOWN: "cruise_down",
    STATE_DECEL_DOWN: "decel_down",
    STATE_DOOR_OPERATION: "door_operation",
    STATE_UNKNOWN: "unknown",
}

_MOTION_STATES = {
    STATE_ACCEL_UP, STATE_CRUISE_UP, STATE_DECEL_UP,
    STATE_ACCEL_DOWN, STATE_CRUISE_DOWN, STATE_DECEL_DOWN,
}


# ──────────────────────────────────────────────
#  Data structures
# ──────────────────────────────────────────────

@dataclass
class Phase:
    """A contiguous segment of a single state."""
    state: str
    state_code: int
    t_start: float
    t_end: float
    idx_start: int
    idx_end: int

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    @property
    def n_samples(self) -> int:
        return self.idx_end - self.idx_start


@dataclass
class StateAnalysis:
    """Complete per-trip state analysis result."""

    state_codes: np.ndarray       # per-sample int codes
    state_labels: list[str]       # labels for codes present in this trip
    phases: list[Phase]           # contiguous state segments

    direction: str                # 'up', 'down', or 'unknown'
    n_runs: int                   # number of distinct motion segments
    total_duration: float

    state_durations: dict[str, float]   # total time in each state
    state_rms_z: dict[str, float]       # RMS az per state (m/s²)

    velocity: np.ndarray | None = None  # drift-corrected velocity

    door_rattle_index: float = 0.0
    door_rattle_severity: str = "low"
    door_events_detected: int = 0

    @property
    def standing_duration(self) -> float:
        return self.state_durations.get("standing", 0.0)

    @property
    def motion_duration(self) -> float:
        return sum(
            d for s, d in self.state_durations.items()
            if s not in ("standing", "unknown")
        )

    def summary(self) -> str:
        lines = [
            f"StateAnalysis: direction={self.direction}, {self.n_runs} run(s), "
            f"{self.total_duration:.1f}s total",
        ]
        for s, d in sorted(self.state_durations.items(), key=lambda x: -x[1]):
            pct = d / self.total_duration * 100 if self.total_duration > 0 else 0
            lines.append(f"  {s:<16} {d:6.2f}s ({pct:5.1f}%)")
        return "\n".join(lines)


# ──────────────────────────────────────────────
#  Velocity integration with drift correction
# ──────────────────────────────────────────────

def integrate_velocity(
    az_dynamic: np.ndarray,
    dt: float = 0.01,
    standing_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Integrate acceleration to velocity with optional drift correction.

    Args:
        az_dynamic: Gravity-removed vertical acceleration (m/s²).
        dt: Sample interval in seconds.
        standing_mask: Boolean mask where velocity should be zero.

    Returns:
        Velocity array (m/s), drift-corrected if standing_mask provided.
    """
    n = len(az_dynamic)
    vel = cumulative_trapezoid(az_dynamic, dx=dt, initial=0.0)

    if standing_mask is None:
        return vel

    standing_mask = np.asarray(standing_mask, dtype=bool)

    # Find contiguous motion segments (standing→motion→standing transitions)
    changes = np.diff(np.concatenate([[False], standing_mask, [False]]).astype(int))
    starts = np.where(changes == -1)[0]  # standing→motion
    ends = np.where(changes == 1)[0]     # motion→standing

    for s, e in zip(starts, ends):
        if s >= e or s >= n:
            continue
        e = min(e, n)
        seg_len = e - s
        if seg_len < 2:
            vel[s:e] = 0.0
            continue

        # Remove linear drift between segment endpoints
        v_seg = vel[s:e].copy()
        t_seg = np.arange(seg_len, dtype=float)
        v_start, v_end = v_seg[0], v_seg[-1]
        trend = v_start + (v_end - v_start) * t_seg / (seg_len - 1)
        vel[s:e] = v_seg - trend

    vel[standing_mask] = 0.0
    return vel


# ──────────────────────────────────────────────
#  State classification
# ──────────────────────────────────────────────

def _running_rms(signal: np.ndarray, window: int = 50) -> np.ndarray:
    squared = signal ** 2
    kernel = np.ones(window) / window
    return np.sqrt(np.convolve(squared, kernel, mode='same'))


def classify_motion_state(
    az_dynamic: np.ndarray,
    velocity: np.ndarray,
    dt: float = 0.01,
    *,
    accel_threshold: float = 0.03,
    vel_threshold: float = 0.05,
    still_threshold: float = 0.005,
    rms_window: int = 50,
    min_state_duration: int = 15,
) -> np.ndarray:
    """Classify per-sample motion state from az_dynamic and velocity.

    Standing: low az RMS + low velocity.
    Accel:    |az| > threshold AND az·velocity > 0 (same sign).
    Decel:    |az| > threshold AND az·velocity < 0 (opposite sign).
    Cruise:   in motion, |az| ≤ threshold, |velocity| > 0.

    Returns integer state codes array.
    """
    n = len(az_dynamic)
    az_rms = _running_rms(az_dynamic, rms_window)

    state = np.full(n, STATE_UNKNOWN, dtype=int)

    # Standing: low RMS and low velocity
    is_still = (az_rms < still_threshold) & (np.abs(velocity) < vel_threshold * 0.5)
    state[is_still] = STATE_STANDING

    is_motion = ~is_still

    # Accel: |az| > threshold AND velocity and az have same sign
    accel_mask = (
        is_motion
        & (np.abs(az_dynamic) > accel_threshold)
        & (az_dynamic * velocity > 0)
    )
    # Decel: |az| > threshold AND velocity and az opposite signs
    decel_mask = (
        is_motion
        & (np.abs(az_dynamic) > accel_threshold)
        & (az_dynamic * velocity < 0)
    )
    # Cruise: in motion, low |az|, non-zero velocity
    cruise_mask = (
        is_motion
        & (np.abs(az_dynamic) <= accel_threshold)
        & (np.abs(velocity) > vel_threshold * 0.3)
    )

    for i in range(n):
        if not is_motion[i]:
            continue
        going_up = velocity[i] > 0
        if accel_mask[i]:
            state[i] = STATE_ACCEL_UP if going_up else STATE_ACCEL_DOWN
        elif decel_mask[i]:
            state[i] = STATE_DECEL_UP if going_up else STATE_DECEL_DOWN
        elif cruise_mask[i]:
            state[i] = STATE_CRUISE_UP if going_up else STATE_CRUISE_DOWN
        else:
            state[i] = STATE_STANDING

    # Merge short segments iteratively until stable
    prev_n = None
    while True:
        state = _merge_short_segments(state, min_state_duration)
        n_seg = _count_segments(state)
        if n_seg == prev_n:
            break
        prev_n = n_seg

    return state


def _count_segments(state: np.ndarray) -> int:
    if len(state) == 0:
        return 0
    return int(np.sum(np.diff(state) != 0)) + 1


def _merge_short_segments(state: np.ndarray, min_duration: int = 5) -> np.ndarray:
    """Merge state segments shorter than min_duration into neighbors."""
    result = state.copy()
    n = len(state)

    changes = np.diff(state) != 0
    boundary_idx = np.concatenate([[0], np.where(changes)[0] + 1, [n]])

    for i in range(len(boundary_idx) - 1):
        start = boundary_idx[i]
        end = boundary_idx[i + 1]
        length = end - start

        if length >= min_duration:
            continue
        if start == 0 and end == n:
            continue

        prev_len = start - (boundary_idx[i - 1] if i > 0 else 0)
        next_len = (boundary_idx[i + 2] if i + 2 < len(boundary_idx) else n) - end
        replacement = state[start - 1] if prev_len >= next_len else state[end]
        result[start:end] = replacement

    return result


def detect_direction(
    az_dynamic: np.ndarray,
    state_codes: np.ndarray,
) -> str:
    """Determine run direction from acceleration phases.

    Returns 'up', 'down', or 'unknown'.
    """
    accel_up = np.isin(state_codes, [STATE_ACCEL_UP])
    accel_down = np.isin(state_codes, [STATE_ACCEL_DOWN])

    n_up = np.sum(accel_up)
    n_down = np.sum(accel_down)

    if n_up > n_down:
        return "up"
    elif n_down > n_up:
        return "down"
    elif n_up == 0 and n_down == 0:
        return "unknown"

    all_accel = accel_up | accel_down
    if np.any(all_accel):
        return "up" if np.mean(az_dynamic[all_accel]) > 0 else "down"
    return "unknown"


# ──────────────────────────────────────────────
#  Phase extraction
# ──────────────────────────────────────────────

def extract_phases(
    state_codes: np.ndarray,
    time_seconds: np.ndarray,
) -> list[Phase]:
    """Extract contiguous state segments from state array."""
    n = len(state_codes)
    if n == 0:
        return []

    phases: list[Phase] = []
    current_code = state_codes[0]
    idx_start = 0

    for i in range(1, n):
        if state_codes[i] != current_code:
            dur = time_seconds[i - 1] - time_seconds[idx_start]
            if dur > 0:
                phases.append(Phase(
                    state=STATE_LABELS.get(current_code, "unknown"),
                    state_code=int(current_code),
                    t_start=float(time_seconds[idx_start]),
                    t_end=float(time_seconds[i - 1]),
                    idx_start=idx_start,
                    idx_end=i,
                ))
            current_code = state_codes[i]
            idx_start = i

    dur = time_seconds[-1] - time_seconds[idx_start]
    if dur > 0:
        phases.append(Phase(
            state=STATE_LABELS.get(current_code, "unknown"),
            state_code=int(current_code),
            t_start=float(time_seconds[idx_start]),
            t_end=float(time_seconds[-1]),
            idx_start=idx_start,
            idx_end=n,
        ))

    return phases


# ──────────────────────────────────────────────
#  Door operation detection (X-axis based)
# ──────────────────────────────────────────────

def detect_door_operation(
    ax: np.ndarray,
    dt: float = 0.01,
    x_threshold_mg: float = 150,
    min_duration_s: float = 0.5,
    max_duration_s: float = 4.0,
) -> list[dict]:
    """Detect door open/close events from X-axis acceleration spikes.

    Door-mounted sensors experience massive X-axis spikes (>200 mg) during
    door operation because X is the door opening direction. These spikes
    are characterized by:
      - Sharp onset/offset (abrupt transitions)
      - Duration between 0.5--4.0 seconds
      - Peak magnitude well above normal vibration levels

    Door data from S3 is UNRELIABLE; this function detects door events
    purely from the X-axis accelerometer signal.

    Args:
        ax: X-axis acceleration in mg (1 g = 1000 mg).
        dt: Sample interval in seconds.
        x_threshold_mg: Magnitude threshold for door event detection (mg).
        min_duration_s: Minimum event duration to accept (seconds).
        max_duration_s: Maximum event duration to accept (seconds).

    Returns:
        List of dicts, each with:
            type: 'door_event'
            t_start: Start time (s)
            t_end: End time (s)
            x_peak_mg: Peak |ax| during event (mg)
    """
    n = len(ax)
    if n == 0:
        return []

    # Binary mask: where |ax| exceeds threshold
    above_thresh = np.abs(ax) > x_threshold_mg

    # Find contiguous above-threshold segments
    min_samples = int(min_duration_s / dt)
    max_samples = int(max_duration_s / dt)

    # Detect rising/falling edges
    padded = np.concatenate([[False], above_thresh, [False]])
    rising = np.where(padded[1:] & ~padded[:-1])[0]
    falling = np.where(~padded[1:] & padded[:-1])[0]

    events: list[dict] = []
    for r, f in zip(rising, falling):
        if r >= n:
            continue
        f = min(f, n)
        seg_len = f - r
        if seg_len < 2:
            continue
        if seg_len < min_samples or seg_len > max_samples:
            continue

        # Check for sharp onset/offset: high derivative at edges
        # Onset: diff should spike at start of segment
        if r > 0:
            onset_diff = abs(ax[r] - ax[r - 1])
        else:
            onset_diff = abs(ax[r])

        if f < n:
            offset_diff = abs(ax[f - 1] - ax[f]) if f < n else abs(ax[f - 1])
        else:
            offset_diff = abs(ax[f - 1])

        # Require at least one edge to be sharp (> 50% of threshold)
        if onset_diff < x_threshold_mg * 0.5 and offset_diff < x_threshold_mg * 0.5:
            continue

        x_peak = float(np.max(np.abs(ax[r:f])))

        events.append({
            "type": "door_event",
            "t_start": r * dt,
            "t_end": f * dt,
            "x_peak_mg": x_peak,
        })

    return events


def compute_door_rattle_index(
    ax_cruise: np.ndarray,
    noise_floor_x: float,
) -> tuple[float, str]:
    """Compute door rattle index from X-axis RMS during cruise vs noise floor.

    High rattle index during cruise (>10) suggests the door may be loose
    or poorly adjusted, causing excessive lateral vibration when the cab
    is moving.

    Args:
        ax_cruise: X-axis acceleration during cruise phases (mg).
        noise_floor_x: X-axis noise floor RMS from standing/quiet periods (mg).

    Returns:
        Tuple of (rattle_index: float, severity: 'low' | 'moderate' | 'high').

    Interpretations:
        index ≤ 3:   low (normal)
        index 3-10:  moderate (elevated vibration)
        index > 10:  high (door may be loose)
    """
    if noise_floor_x <= 0 or len(ax_cruise) == 0:
        return 0.0, "low"

    rms_cruise = float(np.sqrt(np.mean(ax_cruise ** 2)))
    rattle_index = rms_cruise / noise_floor_x

    if rattle_index > 10:
        severity = "high"
    elif rattle_index > 3:
        severity = "moderate"
    else:
        severity = "low"

    return rattle_index, severity


# ──────────────────────────────────────────────
#  Statistics
# ──────────────────────────────────────────────

def compute_state_summary(
    state_codes: np.ndarray,
    az_dynamic: np.ndarray,
    time_seconds: np.ndarray,
    dt: float = 0.01,
) -> dict:
    """Compute per-state duration and RMS statistics."""
    durations: dict[str, float] = {}
    rms_per_state: dict[str, float] = {}

    for code, label in STATE_LABELS.items():
        mask = state_codes == code
        durations[label] = float(np.sum(mask)) * dt
        rms_per_state[label] = (
            float(np.sqrt(np.mean(az_dynamic[mask] ** 2)))
            if np.any(mask) else 0.0
        )

    n_transitions = int(np.sum(np.diff(state_codes) != 0))

    return {
        "durations": durations,
        "rms_z": rms_per_state,
        "n_transitions": n_transitions,
    }


# ──────────────────────────────────────────────
#  Master analysis function
# ──────────────────────────────────────────────

def analyze_trip_states(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    time_seconds: np.ndarray,
    dt: float = 0.01,
    *,
    accel_threshold: float = 0.03,
    vel_threshold: float = 0.05,
    still_threshold: float = 0.005,
) -> StateAnalysis:
    """Run full state detection on a single elevator trip.

    Args:
        ax, ay, az: Raw triaxial acceleration (az includes gravity).
        time_seconds: Relative time from trip start.
        dt: Sample interval.
        accel_threshold: |az| threshold for accel/decel (m/s²).
        vel_threshold: |v| threshold for motion (m/s).
        still_threshold: RMS az threshold for standing (m/s²).

    Returns:
        StateAnalysis with per-sample states, phases, and statistics.
    """
    from elevator_vibration.preprocessing import separate_gravity
    from elevator_vibration.time_domain import segment_by_velocity

    # 1. Gravity removal
    az_dynamic, _ = separate_gravity(az)

    # 2. Standing mask (segment_by_velocity returns True=moving)
    moving_mask = segment_by_velocity(
        az_dynamic, dt=dt, velocity_threshold=vel_threshold * 0.3,
    ).astype(bool)
    standing_mask = ~moving_mask

    # 3. Drift-corrected velocity
    velocity = integrate_velocity(az_dynamic, dt, standing_mask)

    # 4. Door operation detection (from X-axis, before state classification)
    #    Convert ax from m/s² to mg for thresholding (1 g ≈ 9.81 m/s²)
    ax_mg = ax * 1000 / 9.81  # m/s² → mg
    door_events = detect_door_operation(ax_mg, dt=dt)

    # Compute X-axis noise floor from standing periods
    if np.any(standing_mask):
        noise_floor_x = float(np.sqrt(np.mean(ax_mg[standing_mask] ** 2)))
    else:
        noise_floor_x = float(np.sqrt(np.mean(ax_mg ** 2)))

    # 5. State classification
    state_codes = classify_motion_state(
        az_dynamic, velocity, dt,
        accel_threshold=accel_threshold,
        vel_threshold=vel_threshold,
        still_threshold=still_threshold,
    )

    # 6. Inject door operation states into timeline
    for evt in door_events:
        i_start = int(evt["t_start"] / dt)
        i_end = int(evt["t_end"] / dt)
        i_start = max(0, min(i_start, len(state_codes) - 1))
        i_end = max(i_start + 1, min(i_end, len(state_codes)))
        state_codes[i_start:i_end] = STATE_DOOR_OPERATION

    # 7. Direction
    direction = detect_direction(az_dynamic, state_codes)

    # 8. Phases
    phases = extract_phases(state_codes, time_seconds)

    # 9. Statistics
    stats = compute_state_summary(state_codes, az_dynamic, time_seconds, dt)

    # 10. Count distinct motion runs
    n_runs = 0
    in_motion = False
    for p in phases:
        if p.state_code in _MOTION_STATES:
            if not in_motion:
                n_runs += 1
                in_motion = True
        else:
            in_motion = False

    # 11. Door rattle index (X-axis RMS during cruise / noise floor)
    cruise_mask = np.isin(state_codes, [STATE_CRUISE_UP, STATE_CRUISE_DOWN])
    ax_cruise = ax_mg[cruise_mask]
    rattle_index, rattle_severity = compute_door_rattle_index(ax_cruise, noise_floor_x)

    return StateAnalysis(
        state_codes=state_codes,
        state_labels=[STATE_LABELS.get(c, "unknown") for c in np.unique(state_codes)],
        phases=phases,
        direction=direction,
        n_runs=n_runs,
        total_duration=float(time_seconds[-1]),
        state_durations=stats["durations"],
        state_rms_z=stats["rms_z"],
        velocity=velocity,
        door_rattle_index=rattle_index,
        door_rattle_severity=rattle_severity,
        door_events_detected=len(door_events),
    )
