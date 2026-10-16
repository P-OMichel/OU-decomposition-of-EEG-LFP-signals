"""
Synthetic EEG PSD generator optimized to match FOOOF's generative assumptions:
- Log-additive aperiodic + periodic decomposition
- Pure Gaussian peak profiles
- Decoupled knee and delta peak frequencies
- Physiologically realistic noise-to-signal ratios
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from Functions.extract_psd_bumps import extract_intervals_method1_detrended_seed


# =====================================================================
# 1. FOOOF-COMPATIBLE PEAK EVALUATION (GAUSSIAN IN LOG SPACE)
# =====================================================================
def evaluate_gaussian_log_peak(f, f0, height_log10, width_std):
    """
    Evaluates a Gaussian peak in log10 space matching FOOOF formulation:
    G(f) = a * exp(-((f - c)^2) / (2 * sigma^2))
    where:
      - f0: center frequency (c)
      - height_log10: peak height above aperiodic background in log10 units (a)
      - width_std: standard deviation width in Hz (sigma)
    """
    return height_log10 * np.exp(-((f - f0) ** 2) / (2.0 * width_std**2))


# =====================================================================
# 2. SAMPLING LOGIC & SYNTHESIS
# =====================================================================
def sample_eeg_peak_frequencies(f_min=0.5, f_max=45.0, min_gap=2.5, max_tries=1000):
    """
    Samples realistic, well-separated oscillatory rhythms:
      - Delta (2.0 - 4.0 Hz): p = 0.80 (kept above knee)
      - Alpha (8.5 - 12.5 Hz): p = 0.95 (dominant posterior)
      - Theta (5.0 - 7.5 Hz): p = 0.30
      - Beta  (15.0 - 28.0 Hz): p = 0.35
    """
    for _ in range(max_tries):
        peaks_to_generate = []

        # 1. Delta candidate (placed above 2.0 Hz so knee doesn't swallow it)
        if np.random.rand() < 0.80:
            f0_delta = np.random.uniform(2.0, 3.8)
            peaks_to_generate.append({"band": "delta", "f0": f0_delta})

        # 2. Theta candidate
        if np.random.rand() < 0.30:
            f0_theta = np.random.uniform(5.0, 7.5)
            peaks_to_generate.append({"band": "theta", "f0": f0_theta})

        # 3. Alpha candidate
        if np.random.rand() < 0.95:
            f0_alpha = np.random.uniform(8.5, 12.5)
            peaks_to_generate.append({"band": "alpha", "f0": f0_alpha})

        # 4. Beta candidate
        if np.random.rand() < 0.35:
            f0_beta = np.random.uniform(16.0, 26.0)
            peaks_to_generate.append({"band": "beta", "f0": f0_beta})

        f0_vals = np.array([p["f0"] for p in peaks_to_generate])
        f0_vals.sort()
        if len(f0_vals) <= 1 or np.all(np.diff(f0_vals) >= min_gap):
            return peaks_to_generate

    return [{"band": "delta", "f0": 2.5}, {"band": "alpha", "f0": 10.2}]


def generate_single_eeg_psd_sample(f_canonical, f_min=0.5, f_max=45.0, min_peak_gap=2.5):
    """
    Synthesizes clean log10 PSD following FOOOF's generative model:
    log10(PSD) = Offset - log10(k + f^chi) + sum(Gaussians)
    """
    # 1. Aperiodic component in log10 space
    chi = np.random.uniform(1.5, 2.5)
    offset = np.random.uniform(1.5, 3.0)
    # Low knee frequency (0.8 - 1.8 Hz) prevents conflict with alpha and beta
    f_knee = np.random.uniform(0.8, 1.8)
    k_knee = f_knee ** chi

    # log10 background
    log10_aperiodic = offset - np.log10(k_knee + (np.maximum(f_canonical, 0.1)) ** chi)

    # 2. Oscillatory peaks added directly in log10 space
    candidate_peaks = sample_eeg_peak_frequencies(f_min=f_min, f_max=f_max, min_gap=min_peak_gap)
    log10_oscillatory = np.zeros_like(f_canonical)
    peaks_meta = []

    for peak_info in candidate_peaks:
        f0 = peak_info["f0"]
        band = peak_info["band"]

        if band == "delta":
            width_std = np.random.uniform(0.6, 1.2)  # In Hz
            height_log10 = np.random.uniform(0.40, 0.90)  # Log10 amplitude
        elif band == "alpha":
            width_std = np.random.uniform(0.8, 1.8)
            height_log10 = np.random.uniform(0.50, 1.10)
        elif band == "theta":
            width_std = np.random.uniform(0.6, 1.2)
            height_log10 = np.random.uniform(0.35, 0.70)
        else:  # Beta
            width_std = np.random.uniform(1.2, 2.5)
            height_log10 = np.random.uniform(0.30, 0.60)

        peak_vals = evaluate_gaussian_log_peak(f_canonical, f0=f0, height_log10=height_log10, width_std=width_std)
        log10_oscillatory += peak_vals
        peaks_meta.append({
            "band": band,
            "f0": f0,
            "amp": height_log10,
            "width": width_std,
            "profile": "gaussian",
        })

    # Combined clean spectrum in log10 space
    log10_clean = log10_aperiodic + log10_oscillatory

    # Convert to natural log (np.log) to preserve compatibility with your UNet training
    log_clean = (log10_clean * np.log(10.0)).astype(np.float32)

    meta = {
        "peaks": peaks_meta,
        "chi": chi,
        "offset": offset,
        "k_knee": k_knee,
        "f_knee": f_knee,
    }
    return log_clean, meta


# =====================================================================
# 3. DATASET GENERATOR
# =====================================================================
def generate_and_save_eeg_dataset(
    filepath="eeg_psd_benchmark.npz",
    n_samples=100,
    n_freqs=250,
    f_min=0.5,
    f_max=45.0,
    min_peak_gap=2.5,
    n_obs_range=(120, 350),     # Minimum observation points
    k_welch_range=(20, 50),     # Higher Welch segments (cleaner empirical PSD)
    lam=1e2,
    max_high_ratio=0.01,
    min_boundary_ratio=0.5,
):
    # Enforce float64 linspace to avoid step-size roundoff errors
    f_canonical = np.linspace(f_min, f_max, n_freqs, dtype=np.float64)

    log_clean_arr = np.zeros((n_samples, n_freqs), dtype=np.float32)
    log_noisy_arr = np.zeros((n_samples, n_freqs), dtype=np.float32)
    masks_arr = np.zeros((n_samples, 2, n_freqs), dtype=np.float32)
    metadata_list = []

    print(f"Generating {n_samples} FOOOF-compatible EEG samples...")

    for i in range(n_samples):
        # 1. Synthesize clean spectrum
        log_clean, meta = generate_single_eeg_psd_sample(
            f_canonical, f_min=f_min, f_max=f_max, min_peak_gap=min_peak_gap
        )
        f0_list = [p["f0"] for p in meta.get("peaks", [])]

        # 2. Extract Interval Boundaries and compute identical 2D masks
        intervals = extract_intervals_method1_detrended_seed(
            f_canonical, log_clean, f0_list, lam=lam,
            max_high_ratio=max_high_ratio, min_boundary_ratio=min_boundary_ratio
        )

        peak_mask = np.zeros(n_freqs, dtype=np.float32)
        interval_mask = np.zeros(n_freqs, dtype=np.float32)

        for info in intervals:
            f0 = info["f0"]
            f_left = info["peak_edges"][0] if info["peak_edges"][0] is not None else info["segment_bounds"][0]
            f_right = info["peak_edges"][1] if info["peak_edges"][1] is not None else info["segment_bounds"][1]

            peak_mask[np.argmin(np.abs(f_canonical - f0))] = 1.0
            interval_mask[(f_canonical >= f_left) & (f_canonical <= f_right)] = 1.0

        # 3. Simulate Welch observation noise
        n_obs = int(np.random.randint(n_obs_range[0], n_obs_range[1] + 1))
        f_obs = np.linspace(f_min, f_max, n_obs, dtype=np.float64)

        clean_interp = interp1d(f_canonical, log_clean, kind="linear", fill_value="extrapolate")
        log_clean_obs = clean_interp(f_obs)

        # Multiplicative gamma noise for Welch periodograms
        k_seg = int(np.random.randint(k_welch_range[0], k_welch_range[1] + 1))
        noise_mult = np.random.gamma(shape=k_seg, scale=1.0 / k_seg, size=n_obs).astype(np.float64)
        linear_noisy_obs = np.exp(log_clean_obs) * noise_mult
        log_noisy_obs = np.log(np.maximum(linear_noisy_obs, 1e-12))

        # 4. Interpolate back to canonical grid
        recon_interp = interp1d(f_obs, log_noisy_obs, kind="linear", fill_value="extrapolate")
        log_noisy = recon_interp(f_canonical).astype(np.float32)

        # Store arrays
        log_clean_arr[i] = log_clean
        log_noisy_arr[i] = log_noisy
        masks_arr[i, 0] = peak_mask
        masks_arr[i, 1] = interval_mask

        meta["n_obs"] = n_obs
        meta["k_seg"] = k_seg
        metadata_list.append(meta)

    np.savez_compressed(
        filepath,
        f=f_canonical.astype(np.float32),
        log_noisy=log_noisy_arr,
        log_clean=log_clean_arr,
        masks=masks_arr,
        metadata=np.array(metadata_list, dtype=object),
    )
    file_size_kb = os.path.getsize(filepath) / 1024
    print(f"Saved: '{filepath}' ({file_size_kb:.1f} KB, {n_samples} samples).")


if __name__ == "__main__":
    generate_and_save_eeg_dataset(
        filepath="eeg_psd_benchmark_fooof_advantage.npz",
        n_samples=100,
        n_freqs=250,
        f_min=0.5,
        f_max=45.0,
        min_peak_gap=2.5,
    )