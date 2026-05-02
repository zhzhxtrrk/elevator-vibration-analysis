"""Kinematic calculations for elevator trips using ZUPT and tilt compensation.

Calculates velocity and distance from vertical acceleration (Z-axis).
Applies Zero-Velocity Updates (ZUPT) to correct integration drift.
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import cumulative_trapezoid

def compute_kinematics_zupt(
    timestamps: np.ndarray,
    az_dynamic: np.ndarray,
    stationary_mask: np.ndarray,
    scale_factor: float = 1.0,
    gravity: float = 9.80665
) -> dict:
    """
    Computes velocity and distance using double integration with Zero-Velocity Updates (ZUPT).
    
    Args:
        timestamps: Array of timestamps in seconds.
        az_dynamic: Dynamic vertical acceleration (gravity removed) in g.
        stationary_mask: Boolean array where True indicates the elevator is stationary.
        scale_factor: Physical calibration multiplier for calculated distance.
        gravity: Gravity constant (m/s^2) to convert g to m/s^2.
        
    Returns:
        Dictionary containing:
            - velocity: Array of corrected velocities in m/s.
            - distance: Array of distances in m.
            - raw_velocity: Array of uncorrected velocities (for debugging drift).
    """
    # 1. Convert acceleration to m/s^2
    a_ms2 = az_dynamic * gravity
    
    # 2. First integration: Acceleration to Velocity
    raw_velocity = cumulative_trapezoid(a_ms2, x=timestamps, initial=0.0)
    
    # 3. Apply ZUPT (Zero-Velocity Update)
    corrected_velocity = np.copy(raw_velocity)
    
    # Identify moving segments
    is_moving = ~stationary_mask
    
    # Force velocity to 0 during stationary phases
    corrected_velocity[stationary_mask] = 0.0
    
    # Linearly distribute the accumulated velocity error (drift) 
    # across each movement phase to correct tilt/noise bleed.
    diff_moving = np.diff(is_moving.astype(int))
    starts = np.where(diff_moving == 1)[0] + 1
    stops = np.where(diff_moving == -1)[0] + 1
    
    # Handle boundary conditions
    if is_moving[0]:
        starts = np.insert(starts, 0, 0)
    if is_moving[-1]:
        stops = np.append(stops, len(is_moving))
        
    for start, stop in zip(starts, stops):
        # Accumulate error based on the raw velocity exactly when it hits a stationary block
        if stop < len(raw_velocity):
            # The velocity error at the end of the motion phase
            end_error = raw_velocity[stop] - raw_velocity[start] 
            
            # Linearly distribute error
            n_samples = stop - start
            if n_samples > 0:
                drift_correction = np.linspace(0, end_error, n_samples, endpoint=False)
                # Apply correction so that velocity ends at ~0 precisely
                corrected_velocity[start:stop] = raw_velocity[start:stop] - raw_velocity[start] - drift_correction

    # 4. Second integration: Velocity to Distance
    distance = cumulative_trapezoid(corrected_velocity, x=timestamps, initial=0.0)
    
    # 5. Apply physical scale factor calibration
    distance = distance * scale_factor
    
    return {
        "velocity": corrected_velocity,
        "distance": distance,
        "raw_velocity": raw_velocity,
    }
