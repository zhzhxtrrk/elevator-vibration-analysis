"""Frequency-domain analysis: FFT, PSD, STFT, envelope spectrum, coherence."""

from __future__ import annotations

import numpy as np
from scipy import signal as sig
from scipy.fft import fft, fftfreq, rfft, rfftfreq


# ──────────────────────────────────────────────
#  FFT / Magnitude spectrum
# ──────────────────────────────────────────────


def compute_fft(
    signal: np.ndarray,
    fs: float = 100.0,
    remove_dc: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute single-sided FFT magnitude spectrum.

    Args:
        signal: 1-D real signal.
        fs: Sampling frequency (Hz).
        remove_dc: If True, subtract mean before FFT.

    Returns:
        (freqs, magnitude) — positive frequencies only.
    """
    if remove_dc:
        signal = signal - np.mean(signal)

    n = len(signal)
    mag = np.abs(rfft(signal)) / n
    mag[1:] *= 2.0  # double except DC
    freqs = rfftfreq(n, 1.0 / fs)
    return freqs, mag


def compute_fft_complex(
    signal: np.ndarray,
    fs: float = 100.0,
    remove_dc: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute full FFT returning (freqs, magnitude, phase).

    Returns negative and positive frequencies.
    """
    if remove_dc:
        signal = signal - np.mean(signal)

    n = len(signal)
    spectrum = fft(signal) / n
    freqs = fftfreq(n, 1.0 / fs)
    mag = np.abs(spectrum)
    phase = np.angle(spectrum)
    return freqs, mag, phase


# ──────────────────────────────────────────────
#  Power Spectral Density (Welch's method)
# ──────────────────────────────────────────────


def compute_psd(
    signal: np.ndarray,
    fs: float = 100.0,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PSD using Welch's method.

    Returns (freqs, psd) in (m/s²)²/Hz.
    """
    if noverlap is None:
        noverlap = nperseg // 2
    freqs, psd = sig.welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)
    return freqs, psd


def compute_psd_density(
    signal: np.ndarray,
    fs: float = 100.0,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PSD in amplitude spectral density: sqrt(PSD) in m/s²/sqrt(Hz)."""
    freqs, psd = compute_psd(signal, fs, nperseg, noverlap, window)
    asd = np.sqrt(psd)
    return freqs, asd


# ──────────────────────────────────────────────
#  STFT / Spectrogram
# ──────────────────────────────────────────────


def compute_spectrogram(
    signal: np.ndarray,
    fs: float = 100.0,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
    scaling: str = "magnitude",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute STFT spectrogram.

    Args:
        scaling: 'magnitude' or 'power' (dB).

    Returns:
        (freqs, times, Sxx) — Sxx is magnitude or dB depending on scaling.
    """
    if noverlap is None:
        noverlap = nperseg // 2

    freqs, times, Sxx = sig.spectrogram(
        signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        mode="magnitude" if scaling == "magnitude" else "psd",
    )

    if scaling == "power":
        Sxx = 10.0 * np.log10(np.maximum(Sxx, 1e-20))  # dB

    return freqs, times, Sxx


# ──────────────────────────────────────────────
#  Envelope spectrum (Hilbert demodulation)
# ──────────────────────────────────────────────


def compute_envelope_spectrum(
    signal: np.ndarray,
    fs: float = 100.0,
    bandpass: tuple[float, float] | None = None,
    bandpass_order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute envelope spectrum via Hilbert transform.

    Optionally bandpass-filters the signal first to isolate a bearing
    or gearmesh frequency band.

    Args:
        signal: 1-D real signal.
        fs: Sampling frequency (Hz).
        bandpass: Optional (low_cut, high_cut) in Hz for bandpass filter.
        bandpass_order: Butterworth filter order.

    Returns:
        (freqs, envelope_magnitude).
    """
    if bandpass is not None:
        nyq = fs / 2.0
        low = bandpass[0] / nyq
        high = bandpass[1] / nyq
        b, a = sig.butter(bandpass_order, [low, high], btype="band")
        signal = sig.filtfilt(b, a, signal)

    # Hilbert envelope
    analytic = sig.hilbert(signal)
    envelope = np.abs(analytic)

    # FFT of envelope
    freqs, mag = compute_fft(envelope, fs=fs)
    return freqs, mag


# ──────────────────────────────────────────────
#  Cepstrum (for harmonic/echo detection)
# ──────────────────────────────────────────────


def compute_cepstrum(
    signal: np.ndarray,
    fs: float = 100.0,
    cepstrum_type: str = "power",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the cepstrum (inverse FFT of log spectrum).

    Args:
        cepstrum_type: 'power' (log |FFT|²) or 'complex' (ifft(log FFT)).
        fs: Sampling frequency (Hz).

    Returns:
        (quefrency_seconds, cepstrum_amplitude).
    """
    n = len(signal)
    signal = signal - np.mean(signal)

    if cepstrum_type == "power":
        spectrum = np.abs(rfft(signal)) ** 2
        log_spectrum = np.log(np.maximum(spectrum, 1e-20))
        cepst = np.abs(fft(log_spectrum))[: n // 2]
        quef = np.arange(len(cepst)) / fs
    else:
        spectrum = fft(signal)
        log_spectrum = np.log(np.maximum(np.abs(spectrum), 1e-20)) + 1j * np.angle(spectrum)
        cepst = np.real(np.fft.ifft(log_spectrum))
        quef = np.arange(len(cepst)) / fs

    return quef, cepst


# ──────────────────────────────────────────────
#  Cross-spectrum & Coherence
# ──────────────────────────────────────────────


def compute_coherence(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    fs: float = 100.0,
    nperseg: int = 256,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude-squared coherence between two signals.

    Returns (freqs, coherence) where 0 ≤ coherence ≤ 1.
    """
    if noverlap is None:
        noverlap = nperseg // 2
    freqs, coh = sig.coherence(signal_a, signal_b, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return freqs, coh


def compute_cross_spectrum(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    fs: float = 100.0,
    nperseg: int = 256,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-spectral density: magnitude and phase.

    Returns (freqs, cross_magnitude, cross_phase_radians).
    """
    if noverlap is None:
        noverlap = nperseg // 2
    freqs, Pxy = sig.csd(signal_a, signal_b, fs=fs, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Pxy)
    phase = np.angle(Pxy)
    return freqs, mag, phase


def compute_transfer_function(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    fs: float = 100.0,
    nperseg: int = 256,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """H = Pxy / Pxx estimate of transfer function.

    Returns (freqs, magnitude, phase_radians).
    """
    if noverlap is None:
        noverlap = nperseg // 2
    freqs, Pxx = sig.welch(input_signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, Pxy = sig.csd(input_signal, output_signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    H = Pxy / np.maximum(Pxx, 1e-20)
    return freqs, np.abs(H), np.angle(H)


# ──────────────────────────────────────────────
#  Band energy extraction
# ──────────────────────────────────────────────


def compute_band_energy(
    freqs: np.ndarray,
    psd: np.ndarray,
    low: float,
    high: float,
) -> float:
    """Integrate PSD over a frequency band [low, high] in Hz."""
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    df = freqs[1] - freqs[0]
    return float(np.trapz(psd[mask], dx=df))


def dominant_frequency(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    low: float = 0.5,
    high: float = 50.0,
) -> float:
    """Find the frequency with the highest magnitude in a band."""
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    idx = np.argmax(magnitude[mask])
    return float(freqs[mask][idx])


# ──────────────────────────────────────────────
#  Convenience: per-axis frequency summary
# ──────────────────────────────────────────────


def frequency_analysis(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    fs: float = 100.0,
    nperseg: int = 256,
) -> dict:
    """Run full frequency-domain analysis on all three axes.

    Returns a dict with FFT, PSD, and dominant frequencies per axis,
    plus cross-axis coherence pairs.
    """
    axes = {"ax": ax, "ay": ay, "az": az}
    result: dict = {}

    for name, sig_array in axes.items():
        freqs, mag = compute_fft(sig_array, fs)
        freqs_psd, psd = compute_psd(sig_array, fs, nperseg=nperseg)
        dom_freq = dominant_frequency(freqs, mag)
        result[f"{name}_fft_freqs"] = freqs
        result[f"{name}_fft_mag"] = mag
        result[f"{name}_psd_freqs"] = freqs_psd
        result[f"{name}_psd"] = psd
        result[f"{name}_dominant_freq"] = dom_freq

    # Cross-axis coherence
    for pair in [("ax", "ay"), ("ax", "az"), ("ay", "az")]:
        f, coh = compute_coherence(axes[pair[0]], axes[pair[1]], fs, nperseg=nperseg)
        result[f"coh_{pair[0]}_{pair[1]}_freqs"] = f
        result[f"coh_{pair[0]}_{pair[1]}"] = coh

    return result
