"""Visualization: time series, spectrograms, ride quality dashboards.

All functions accept matplotlib Axes objects so they can be composed.
Uses matplotlib only (no seaborn).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec


# ══════════════════════════════════════════════
#  Color palette & style defaults
# ══════════════════════════════════════════════

AXIS_COLORS = {"x": "#E74C3C", "y": "#2ECC71", "z": "#3498DB", "total": "#9B59B6",
               "ax": "#E74C3C", "ay": "#2ECC71", "az": "#3498DB"}
AXIS_LABELS = {"x": "X (door dir)", "y": "Y (lateral)", "z": "Z (vertical)"}

plt.rcParams.update({
    "figure.dpi": 120,
    "figure.figsize": (14, 8),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


# ══════════════════════════════════════════════
#  Single-axis plots
# ══════════════════════════════════════════════

def plot_timeseries(
    time_seconds: np.ndarray,
    signals: dict[str, np.ndarray],
    title: str = "Vibration Time Series",
    ax: Axes | None = None,
    alpha: float = 0.8,
) -> Axes:
    """Plot multiple signals against time.

    Args:
        time_seconds: 1-D time array.
        signals: Dict mapping label → signal array.
        title: Plot title.
        ax: Optional existing axes.
    """
    ax = ax or plt.gca()
    for label, sig in signals.items():
        color = AXIS_COLORS.get(label, None)
        ax.plot(time_seconds, sig, label=label, alpha=alpha, linewidth=0.6, color=color)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=len(signals))
    return ax


def plot_three_axis(
    time_seconds: np.ndarray,
    ax_data: np.ndarray,
    ay_data: np.ndarray,
    az_data: np.ndarray,
    title: str = "Triaxial Vibration",
    figsize: tuple = (14, 8),
) -> Figure:
    """3-subplot layout: one axis per row, time-aligned.

    Returns the Figure for save/show.
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    for i, (ax, data, label) in enumerate(zip(
        axes,
        [ax_data, ay_data, az_data],
        ["ax", "ay", "az"]
    )):
        color = AXIS_COLORS[label]
        axes[i].plot(time_seconds, data, linewidth=0.5, color=color)
        axes[i].set_ylabel(f"{AXIS_LABELS[label]} (m/s²)")
        axes[i].set_title(f"{label.upper()} — {AXIS_LABELS[label]}" if i == 0 else f"{label.upper()}")
        axes[i].axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    label: str = "",
    ax: Axes | None = None,
    log_scale: bool = True,
    max_freq: float | None = None,
) -> Axes:
    """Plot Power Spectral Density.

    Args:
        freqs: Frequency bins.
        psd: PSD values.
        label: Legend label.
        ax: Optional axes.
        log_scale: Use dB scale (10*log10).
        max_freq: Clip x-axis.
    """
    ax = ax or plt.gca()
    y = 10 * np.log10(psd + 1e-15) if log_scale else psd
    color = AXIS_COLORS.get(label, None)
    ax.plot(freqs, y, linewidth=0.8, color=color, label=label)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB)" if log_scale else "PSD")
    if max_freq:
        ax.set_xlim(0, max_freq)
    if label:
        ax.legend()
    return ax


def plot_spectrogram(
    time_seconds: np.ndarray,
    freqs: np.ndarray,
    Sxx: np.ndarray,
    title: str = "Spectrogram",
    ax: Axes | None = None,
    max_freq: float | None = None,
) -> Axes:
    """Plot STFT spectrogram as a heatmap.

    Args:
        time_seconds: Time bins (from STFT).
        freqs: Frequency bins.
        Sxx: Spectrogram magnitude.
        title: Plot title.
        ax: Optional axes.
        max_freq: Clip y-axis.
    """
    ax = ax or plt.gca()
    Sxx_db = 10 * np.log10(Sxx + 1e-15)
    extent = [time_seconds[0], time_seconds[-1], freqs[0], freqs[-1]]
    im = ax.imshow(
        Sxx_db, aspect="auto", origin="lower", extent=extent, cmap="inferno"
    )
    if max_freq:
        ax.set_ylim(0, max_freq)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="dB")
    return ax


# ══════════════════════════════════════════════
#  Composite dashboards
# ══════════════════════════════════════════════

def plot_run_overview(
    time_seconds: np.ndarray,
    ax_data: np.ndarray,
    ay_data: np.ndarray,
    az_data: np.ndarray,
    az_dynamic: np.ndarray,
    run_mask: np.ndarray | None = None,
    title: str = "Elevator Run Overview",
    figsize: tuple = (16, 10),
) -> Figure:
    """4-panel overview: 3-axis time series + composite + moving mask.

    Args:
        run_mask: Boolean array, True where elevator is moving.
    """
    composite = np.sqrt(ax_data**2 + ay_data**2 + az_dynamic**2)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 1, figure=fig, hspace=0.3)

    # Panel 1: 3-axis overlay
    ax1 = fig.add_subplot(gs[0])
    for label, sig in [("x", ax_data), ("y", ay_data), ("z", az_data)]:
        ax1.plot(time_seconds, sig, linewidth=0.5, color=AXIS_COLORS[label], label=label, alpha=0.7)
    ax1.set_ylabel("Accel (m/s²)")
    ax1.set_title("Raw Triaxial Acceleration")
    ax1.legend(ncol=3, fontsize=8)

    # Panel 2: Vertical dynamic + composite
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(time_seconds, az_dynamic, linewidth=0.5, color=AXIS_COLORS["z"], label="Z dyn", alpha=0.8)
    ax2.plot(time_seconds, composite, linewidth=0.5, color=AXIS_COLORS["total"], label="Composite", alpha=0.6)
    if run_mask is not None:
        mask_fill = np.where(run_mask, np.max(composite) * 0.9, 0)
        ax2.fill_between(time_seconds, 0, mask_fill, alpha=0.1, color="orange", label="Moving")
    ax2.set_ylabel("Accel (m/s²)")
    ax2.set_title("Z-Dynamic & Composite (gravity removed)")
    ax2.legend(fontsize=8)
    ax2.axhline(y=0, color="gray", linestyle=":", alpha=0.4)

    # Panel 3: Velocity estimate (cumulative integral of Z-dynamic)
    ax3 = fig.add_subplot(gs[2])
    dt = float(np.median(np.diff(time_seconds)))
    velocity = np.cumsum(az_dynamic) * dt
    ax3.plot(time_seconds, velocity, linewidth=0.8, color="steelblue")
    ax3.set_ylabel("Est. Velocity (m/s)")
    ax3.set_title("Estimated Vertical Velocity (∫ az_dyn dt)")
    ax3.axhline(y=0, color="gray", linestyle=":", alpha=0.4)

    # Panel 4: Composite PSD
    ax4 = fig.add_subplot(gs[3])
    from scipy import signal as sig
    nperseg = min(256, len(composite) // 4)
    freqs_psd, psd = sig.welch(composite, fs=1.0/dt, nperseg=nperseg)
    ax4.semilogy(freqs_psd, psd, linewidth=0.8, color=AXIS_COLORS["total"])
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("PSD")
    ax4.set_title("Composite PSD")
    ax4.set_xlim(0, min(50, 1.0/(2*dt)))

    fig.suptitle(title, fontsize=14, fontweight="bold")
    return fig


def plot_ride_quality_dashboard(
    iso2631: dict,
    iso18738: dict,
    time_domain_stats: dict,
    title: str = "Ride Quality Dashboard",
    figsize: tuple = (14, 6),
) -> Figure:
    """Text/metric dashboard summarizing ride quality assessment."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Left: metrics table
    ax1 = axes[0]
    ax1.axis("off")
    lines = [
        "╔══════════════════════════╗",
        "║   ISO 2631-1 Comfort     ║",
        "╠══════════════════════════╣",
        f"║  Wgt RMS X:  {iso2631.get('aw_x', 0):8.4f} m/s²  ║",
        f"║  Wgt RMS Y:  {iso2631.get('aw_y', 0):8.4f} m/s²  ║",
        f"║  Wgt RMS Z:  {iso2631.get('aw_z', 0):8.4f} m/s²  ║",
        f"║  VTV:        {iso2631.get('av_total', 0):8.4f} m/s²  ║",
        f"║  Classification: {iso2631.get('comfort_class', 'N/A'):>13s}  ║",
        "╚══════════════════════════╝",
        "",
        "╔══════════════════════════╗",
        "║   ISO 18738 Ride Quality ║",
        "╠══════════════════════════╣",
        f"║  A95 (composite): {iso18738.get('global_composite_a95', 0):6.4f} m/s²  ║",
        f"║  Peak (composite):{iso18738.get('global_composite_peak', 0):6.3f} m/s²  ║",
        f"║  Jerk peak:       {iso18738.get('global_jerk_peak', 0):6.3f} m/s³  ║",
        "╚══════════════════════════╝",
    ]
    ax1.text(0.5, 0.5, "\n".join(lines), transform=ax1.transAxes,
             fontfamily="monospace", fontsize=9, va="center", ha="center")

    # Right: time-domain stats bar chart
    ax2 = axes[1]
    astats = time_domain_stats.get("axis_stats", {})
    metrics = ["rms", "peak", "crest_factor", "kurtosis"]
    x = np.arange(len(metrics))
    width = 0.25
    labels_axis = ["ax", "ay", "az"]
    colors_axis = [AXIS_COLORS[l] for l in labels_axis]

    for i, (label, color) in enumerate(zip(labels_axis, colors_axis)):
        values = [astats.get(f"{label}_{m}", 0) for m in metrics]
        # Normalize kurtosis (can be large)
        if metrics[-1] == "kurtosis":
            values[-1] = min(values[-1], 10) / 10  # cap at 10
        bars = ax2.bar(x + i * width, values, width, label=label, color=color, alpha=0.8)
        for bar, val in zip(bars, [astats.get(f"{label}_{m}", 0) for m in metrics]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.2f}", ha="center", fontsize=7)

    ax2.set_xticks(x + width)
    ax2.set_xticklabels([m.replace("_", "\n") for m in metrics])
    ax2.set_ylabel("Value")
    ax2.set_title("Per-Axis Time-Domain Metrics")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_frequency_overview(
    ax_data: np.ndarray,
    ay_data: np.ndarray,
    az_data: np.ndarray,
    fs: float,
    title: str = "Frequency Analysis Overview",
    figsize: tuple = (14, 10),
    max_freq: float = 50.0,
) -> Figure:
    """3×2 grid: PSD + spectrogram for each axis."""
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    from scipy import signal as sig

    nperseg = min(256, len(ax_data) // 4)

    for row, (label, data) in enumerate(
        [("x", ax_data), ("y", ay_data), ("z", az_data)]
    ):
        color = AXIS_COLORS[label]

        # PSD
        freqs_psd, psd = sig.welch(data, fs=fs, nperseg=nperseg)
        axes[row, 0].semilogy(freqs_psd, psd, linewidth=0.8, color=color)
        axes[row, 0].set_xlim(0, max_freq)
        axes[row, 0].set_ylabel("PSD")
        axes[row, 0].set_title(f"{label.upper()} PSD")

        # Spectrogram
        freqs_stft, t_stft, Sxx = sig.spectrogram(data, fs=fs, nperseg=nperseg)
        Sxx_db = 10 * np.log10(Sxx + 1e-15)
        extent = [t_stft[0], t_stft[-1], freqs_stft[0], min(freqs_stft[-1], max_freq)]
        im = axes[row, 1].imshow(
            Sxx_db, aspect="auto", origin="lower", extent=extent, cmap="inferno"
        )
        axes[row, 1].set_ylim(0, max_freq)
        axes[row, 1].set_ylabel("Freq (Hz)")
        axes[row, 1].set_xlabel("Time (s)" if row == 2 else "")
        axes[row, 1].set_title(f"{label.upper()} Spectrogram")
        plt.colorbar(im, ax=axes[row, 1], label="dB", shrink=0.8)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════
#  Save helper
# ══════════════════════════════════════════════

def save_figure(fig: Figure, path: str, dpi: int = 150) -> str:
    """Save figure to disk. Returns the absolute path."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path


# ══════════════════════════════════════════════
#  One-shot: Full report with outputs
# ══════════════════════════════════════════════

def generate_report_figures(
    time_seconds: np.ndarray,
    ax_data: np.ndarray,
    ay_data: np.ndarray,
    az_data: np.ndarray,
    az_dynamic: np.ndarray,
    run_mask: np.ndarray,
    iso2631: dict,
    iso18738: dict,
    time_domain_stats: dict,
    fs: float,
    output_dir: str,
    prefix: str = "report",
) -> list[str]:
    """Generate all report figures and save to output_dir.

    Returns list of saved file paths.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    # 1. Run overview
    fig1 = plot_run_overview(time_seconds, ax_data, ay_data, az_data, az_dynamic, run_mask)
    p1 = save_figure(fig1, os.path.join(output_dir, f"{prefix}_run_overview.png"))
    paths.append(p1)
    plt.close(fig1)

    # 2. Frequency overview
    fig2 = plot_frequency_overview(ax_data, ay_data, az_data, fs)
    p2 = save_figure(fig2, os.path.join(output_dir, f"{prefix}_frequency.png"))
    paths.append(p2)
    plt.close(fig2)

    # 3. Ride quality dashboard
    fig3 = plot_ride_quality_dashboard(iso2631, iso18738, time_domain_stats)
    p3 = save_figure(fig3, os.path.join(output_dir, f"{prefix}_ride_quality.png"))
    paths.append(p3)
    plt.close(fig3)

    return paths
