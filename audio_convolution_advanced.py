"""
audio_convolution_advanced.py
B module: 1D convolution-based audio denoising.

This file provides one clear function for C/D modules:
    denoise_audio_by_convolution(signal, sample_rate, cutoff_hz, num_taps, strength)

The core DSP idea:
    noisy audio x[n] convolved with low-pass FIR kernel h[n]
    y[n] = x[n] * h[n]
"""

from __future__ import annotations

import numpy as np


# =========================================================
# 1. Manual 1D convolution for explanation/demo
# =========================================================
def manual_convolve_1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Manual 1D convolution using sliding window and weighted sum.

    This function is educational and easy to explain in slides.
    For long video audio tracks, use fast_convolve_1d by default.
    """
    x = np.asarray(x, dtype=np.float32)
    h = np.asarray(h, dtype=np.float32)

    n = len(x)
    m = len(h)
    h_flipped = h[::-1]
    pad = m // 2
    x_padded = np.pad(x, (pad, pad), mode="constant")

    y = np.zeros(n, dtype=np.float32)
    for i in range(n):
        window = x_padded[i : i + m]
        y[i] = np.sum(window * h_flipped)

    return y


# =========================================================
# 2. Faster 1D convolution for real video processing
# =========================================================
def fast_convolve_1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Fast convolution using NumPy.

    This is still a convolution operation, but it is much faster than
    the manual Python loop for long audio tracks.
    """
    x = np.asarray(x, dtype=np.float32)
    h = np.asarray(h, dtype=np.float32)
    return np.convolve(x, h, mode="same").astype(np.float32)


# =========================================================
# 3. Normalize audio
# =========================================================
def normalize_audio(x: np.ndarray, peak: float = 0.98) -> np.ndarray:
    """Normalize audio to avoid clipping."""
    x = np.asarray(x, dtype=np.float32)
    max_val = float(np.max(np.abs(x))) if x.size else 0.0
    if max_val <= 1e-12:
        return x
    return (x / max_val * peak).astype(np.float32)


# =========================================================
# 4. FIR low-pass filter design: windowed-sinc method
# =========================================================
def design_lowpass_fir(
    cutoff_hz: float,
    fs: int,
    num_taps: int = 101,
) -> np.ndarray:
    """
    Design a low-pass FIR filter kernel using a windowed-sinc method.

    Parameters
    ----------
    cutoff_hz:
        Frequencies above this value are attenuated.
    fs:
        Sampling rate.
    num_taps:
        Filter length. Odd number recommended.
    """
    if fs <= 0:
        raise ValueError("Sampling rate fs must be positive.")

    nyquist = fs / 2.0
    cutoff_hz = float(cutoff_hz)
    cutoff_hz = min(max(cutoff_hz, 20.0), nyquist * 0.95)

    num_taps = int(num_taps)
    if num_taps < 3:
        num_taps = 3
    if num_taps % 2 == 0:
        num_taps += 1

    fc = cutoff_hz / fs
    n = np.arange(num_taps, dtype=np.float32)
    alpha = (num_taps - 1) / 2.0

    # Ideal low-pass impulse response, multiplied by Hamming window.
    h_ideal = 2 * fc * np.sinc(2 * fc * (n - alpha))
    window = np.hamming(num_taps).astype(np.float32)
    h = h_ideal * window

    # Normalize DC gain to 1.
    h = h / np.sum(h)
    return h.astype(np.float32)


# =========================================================
# 5. Main interface for C module
# =========================================================
def denoise_audio_by_convolution(
    signal: np.ndarray,
    sample_rate: int = 44100,
    cutoff_hz: float = 3000.0,
    num_taps: int = 101,
    strength: float = 1.0,
    use_manual: bool = False,
    normalize_output: bool = False,
) -> np.ndarray:
    """
    Denoise an audio signal by convolving it with a low-pass FIR kernel.

    Parameters
    ----------
    signal:
        1D audio signal for one channel.
    sample_rate:
        Audio sampling rate.
    cutoff_hz:
        Low-pass cutoff frequency. Smaller value removes more high-frequency noise.
    num_taps:
        FIR kernel length. Larger value gives stronger/smoother filtering.
    strength:
        Blend between original and filtered signal.
        0.0 -> original only
        1.0 -> fully filtered
    use_manual:
        True uses manual_convolve_1d. Better for short demos.
        False uses fast NumPy convolution. Better for real videos.
    normalize_output:
        True normalizes the output peak. Usually False for video audio.

    Returns
    -------
    np.ndarray:
        Denoised 1D audio signal.
    """
    x = np.asarray(signal, dtype=np.float32)
    if x.size == 0:
        return x

    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 1e-8:
        return x.copy()

    h = design_lowpass_fir(cutoff_hz=cutoff_hz, fs=sample_rate, num_taps=num_taps)

    if use_manual:
        filtered = manual_convolve_1d(x, h)
    else:
        filtered = fast_convolve_1d(x, h)

    # Blend: useful for UI strength slider.
    y = (1.0 - strength) * x + strength * filtered
    y = np.clip(y, -1.0, 1.0).astype(np.float32)

    if normalize_output:
        y = normalize_audio(y)

    return y


# =========================================================
# 6. Optional demo helper: generate noisy audio and filter it
# =========================================================
def generate_noisy_demo_audio(
    fs: int = 44100,
    duration: float = 3.0,
    cutoff_hz: float = 2000.0,
    num_taps: int = 101,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Generate clean, noisy, and filtered audio arrays for the 2-minute demo.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    clean = (
        0.6 * np.sin(2 * np.pi * 440 * t)
        + 0.3 * np.sin(2 * np.pi * 880 * t)
        + 0.15 * np.sin(2 * np.pi * 1760 * t)
    ).astype(np.float32)

    hf_noise = (
        0.20 * np.sin(2 * np.pi * 5000 * t)
        + 0.15 * np.sin(2 * np.pi * 7000 * t)
    ).astype(np.float32)
    white_noise = (0.08 * np.random.randn(len(t))).astype(np.float32)

    noisy = clean + hf_noise + white_noise
    noisy = normalize_audio(noisy)

    filtered = denoise_audio_by_convolution(
        noisy,
        sample_rate=fs,
        cutoff_hz=cutoff_hz,
        num_taps=num_taps,
        strength=1.0,
        use_manual=False,
        normalize_output=True,
    )

    return normalize_audio(clean), noisy, filtered, fs


if __name__ == "__main__":
    import soundfile as sf

    clean, noisy, filtered, fs = generate_noisy_demo_audio()
    sf.write("clean_audio_demo.wav", clean, fs)
    sf.write("noisy_audio_demo.wav", noisy, fs)
    sf.write("filtered_audio_demo.wav", filtered, fs)
    print("Generated clean_audio_demo.wav, noisy_audio_demo.wav, filtered_audio_demo.wav")
