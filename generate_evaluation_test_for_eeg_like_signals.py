'''
File to generate a synthetic dataset to evaluate the performance of bump segmentation methods
'''

import os
import numpy as np
from scipy.interpolate import interp1d


def sample_eeg_peak_frequencies(f_min=0.5, f_max=45.0, min_gap=2.0, max_tries=1000):
    """
    Samples physiologically grounded EEG peak candidates across standard bands:
      - Delta (0.5 - 4.0 Hz): e.g., slow rhythms / pathology / sleep
      - Alpha (8.0 - 13.0 Hz): prominent posterior rhythm
      - Beta  (14.0 - 30.0 Hz): sensorimotor / frontal bump (optional)
      - Theta (4.0 - 8.0 Hz): frontal midline / memory (optional)
    Ensures a minimum separation of `min_gap` Hz between any co-occurring peaks.
    """
    for _ in range(max_tries):
        peaks_to_generate = []

        # 1. Delta candidate (high probability in EEG benchmarks)
        if np.random.rand() < 0.70:
            f0 = np.random.uniform(1.0, 3.5)
            peaks_to_generate.append({"band": "delta", "f0": f0})

        # 2. Theta candidate (moderate probability)
        if np.random.rand() < 0.35:
            f0 = np.random.uniform(4.5, 7.5)
            peaks_to_generate.append({"band": "theta", "f0": f0})

        # 3. Alpha candidate (dominant EEG feature, very high probability)
        if np.random.rand() < 0.90:
            f0 = np.random.uniform(8.5, 12.5)
            peaks_to_generate.append({"band": "alpha", "f0": f0})

        # 4. Beta candidate (optional sensorimotor rhythm)
        if np.random.rand() < 0.50:
            f0 = np.random.uniform(15.0, 28.0)
            peaks_to_generate.append({"band": "beta", "f0": f0})

        if len(peaks_to_generate) == 0:
            continue

        # Check the >= 2.0 Hz gap condition
        f0_vals = np.array([p["f0"] for p in peaks_to_generate])
        f0_vals.sort()
        if len(f0_vals) == 1 or np.all(np.diff(f0_vals) >= min_gap):
            return peaks_to_generate

    # Fallback to standard delta and alpha if rejection loop stalls
    return [{"band": "delta", "f0": 2.5}, {"band": "alpha", "f0": 10.5}]


def generate_single_eeg_psd_sample(f_canonical, f_min=0.5, f_max=45.0, min_peak_gap=2.0):
    """
    Synthesizes an EEG-like clean PSD:
      - 1/f^chi aperiodic background (with optional knee as in FOOOF/specparam)
      - Gaussian/Lorentzian rhythmic oscillations with physiologically plausible widths
    """
    # 1. Aperiodic Component: L(f) = b - log10(k + f^chi)
    # Exponent chi typically lies in [1.0, 2.5] in human EEG
    chi = np.random.uniform(1.2, 2.2)
    offset = np.random.uniform(0.5, 2.5)
    # Knee parameter k: 0 for pure power-law, >0 for spectra with low-frequency plateau
    has_knee = np.random.rand() < 0.40
    k_knee = np.random.uniform(1.0, 10.0) if has_knee else 0.0

    # Linear-domain aperiodic background: S_ap(f) = 10^offset / (k + f^chi)
    aperiodic_linear = (10.0**offset) / (k_knee + (np.maximum(f_canonical, 0.1))**chi)
    noise_floor = 1e-4
    aperiodic_linear = aperiodic_linear + noise_floor

    # 2. Periodic Oscillatory Peaks (Gaussian profiles are standard in FOOOF modeling)
    candidate_peaks = sample_eeg_peak_frequencies(f_min=f_min, f_max=f_max, min_gap=min_peak_gap)
    oscillatory_linear = np.zeros_like(f_canonical)
    peaks_meta = []

    for peak_info in candidate_peaks:
        f0 = peak_info["f0"]
        band = peak_info["band"]

        # Physiologically realistic bandwidths (standard deviation in Hz)
        if band == "delta":
            sigma = np.random.uniform(0.4, 0.8)
            amp = np.random.uniform(0.5, 3.0)
        elif band == "theta":
            sigma = np.random.uniform(0.5, 1.0)
            amp = np.random.uniform(0.4, 2.0)
        elif band == "alpha":
            sigma = np.random.uniform(0.6, 1.4)  # ~1.5 - 3.3 Hz FWHM
            amp = np.random.uniform(1.0, 6.0)    # Dominant peak
        else:  # Beta
            sigma = np.random.uniform(1.0, 2.5)  # Broader resonance
            amp = np.random.uniform(0.2, 1.5)

        # Add Gaussian bump
        gaussian_peak = amp * np.exp(-((f_canonical - f0)**2) / (2.0 * sigma**2))
        oscillatory_linear += gaussian_peak
        peaks_meta.append({"band": band, "f0": f0, "amp": amp, "sigma": sigma})

    clean_linear = aperiodic_linear + oscillatory_linear
    log_clean = np.log(np.maximum(clean_linear, 1e-12)).astype(np.float32)

    meta = {
        "peaks": peaks_meta,
        "chi": chi,
        "offset": offset,
        "k_knee": k_knee,
        "aperiodic_linear": aperiodic_linear,
    }
    return log_clean, meta


def generate_eeg_benchmark_dataset(
    filepath="eeg_benchmark_psd_100.npz",
    n_samples=100,
    n_freqs=250,
    f_min=0.5,
    f_max=45.0,
    min_peak_gap=2.0,
    n_obs_range=(80, 350),
    k_welch_range=(8, 40),
):
    """
    Generates a 100-sample standalone benchmark dataset designed to evaluate
    and compare FOOOF, IRASA, and the neural network method.
    """
    f_canonical = np.linspace(f_min, f_max, n_freqs, dtype=np.float32)

    log_clean_arr = np.zeros((n_samples, n_freqs), dtype=np.float32)
    log_noisy_arr = np.zeros((n_samples, n_freqs), dtype=np.float32)
    
    # Track physical observation metadata for benchmarking raw vs interpolated methods
    metadata_list = []

    print(f"Generating {n_samples} EEG-like PSD benchmark curves...")

    for i in range(n_samples):
        # 1. Synthesize ground-truth EEG spectrum
        log_clean, meta = generate_single_eeg_psd_sample(
            f_canonical, f_min=f_min, f_max=f_max, min_peak_gap=min_peak_gap
        )

        # 2. Draw unique observation resolution N_obs
        n_obs = int(np.random.randint(n_obs_range[0], n_obs_range[1] + 1))
        f_obs = np.linspace(f_min, f_max, n_obs, dtype=np.float32)

        # Sample clean curve onto physical grid
        clean_interp = interp1d(f_canonical, log_clean, kind="linear", fill_value="extrapolate")
        log_clean_obs = clean_interp(f_obs)

        # 3. Inject unique Welch averaging noise at N_obs
        k_seg = int(np.random.randint(k_welch_range[0], k_welch_range[1] + 1))
        noise_mult = np.random.gamma(shape=k_seg, scale=1.0 / k_seg, size=n_obs).astype(np.float32)
        linear_noisy_obs = np.exp(log_clean_obs) * noise_mult
        log_noisy_obs = np.log(np.maximum(linear_noisy_obs, 1e-12))

        # 4. Interpolate back to canonical grid (250 bins)
        recon_interp = interp1d(f_obs, log_noisy_obs, kind="linear", fill_value="extrapolate")
        log_noisy = recon_interp(f_canonical).astype(np.float32)

        log_clean_arr[i] = log_clean
        log_noisy_arr[i] = log_noisy

        # Save physical observation parameters alongside ground-truth peaks
        meta["n_obs"] = n_obs
        meta["k_seg"] = k_seg
        metadata_list.append(meta)

    # Save to standalone compressed NPZ
    np.savez_compressed(
        filepath,
        f=f_canonical,
        log_noisy=log_noisy_arr,
        log_clean=log_clean_arr,
        metadata=np.array(metadata_list, dtype=object),
    )

    file_size_kb = os.path.getsize(filepath) / 1024
    print(f"Benchmark dataset saved: '{filepath}' ({file_size_kb:.1f} KB, {n_samples} samples).")


if __name__ == "__main__":
    generate_eeg_benchmark_dataset()