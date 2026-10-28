"""
Synthetic EEG PSD generator aligned to the canonical training grid [0.1, 50.0] Hz (250 bins).
"""

import os
import numpy as np
from scipy.interpolate import interp1d
from Functions.extract_psd_bumps import extract_intervals_method1_detrended_seed


def evaluate_gaussian_log_peak(f, f0, height_log10, width_std):
    """Evaluates a Gaussian peak in log10 space matching FOOOF formulation."""
    return height_log10 * np.exp(-((f - f0) ** 2) / (2.0 * width_std**2))


def sample_eeg_peak_frequencies(f_min=1.0, f_max=45.0, min_gap=2.5, max_tries=1000):
    """Samples realistic, well-separated oscillatory rhythms."""
    for _ in range(max_tries):
        peaks_to_generate = []

        if np.random.rand() < 0.80:
            f0_delta = np.random.uniform(2.0, 3.8)
            peaks_to_generate.append({"band": "delta", "f0": f0_delta})

        if np.random.rand() < 0.30:
            f0_theta = np.random.uniform(5.0, 7.5)
            peaks_to_generate.append({"band": "theta", "f0": f0_theta})

        if np.random.rand() < 0.95:
            f0_alpha = np.random.uniform(8.5, 12.5)
            peaks_to_generate.append({"band": "alpha", "f0": f0_alpha})

        if np.random.rand() < 0.35:
            f0_beta = np.random.uniform(16.0, 26.0)
            peaks_to_generate.append({"band": "beta", "f0": f0_beta})

        f0_vals = np.array([p["f0"] for p in peaks_to_generate])
        f0_vals.sort()
        if len(f0_vals) <= 1 or np.all(np.diff(f0_vals) >= min_gap):
            return peaks_to_generate

    return [{"band": "delta", "f0": 2.5}, {"band": "alpha", "f0": 10.2}]


# def generate_single_eeg_psd_sample(f_canonical, f_min=1.0, f_max=45.0, min_peak_gap=2.5):
#     """Synthesizes clean log-e PSD matching FOOOF's generative assumptions."""
#     chi = np.random.uniform(1.5, 2.5)
#     offset = np.random.uniform(1.5, 3.0)
#     f_knee = np.random.uniform(0.8, 1.8)
#     k_knee = f_knee ** chi

#     log10_aperiodic = offset - np.log10(k_knee + (np.maximum(f_canonical, 0.05)) ** chi)

#     candidate_peaks = sample_eeg_peak_frequencies(f_min=f_min, f_max=f_max, min_gap=min_peak_gap)
#     log10_oscillatory = np.zeros_like(f_canonical)
#     peaks_meta = []

#     for peak_info in candidate_peaks:
#         f0 = peak_info["f0"]
#         band = peak_info["band"]

#         if band == "delta":
#             width_std = np.random.uniform(0.6, 1.2)
#             height_log10 = np.random.uniform(0.40, 0.90)
#         elif band == "alpha":
#             width_std = np.random.uniform(0.8, 1.8)
#             height_log10 = np.random.uniform(0.50, 1.10)
#         elif band == "theta":
#             width_std = np.random.uniform(0.6, 1.2)
#             height_log10 = np.random.uniform(0.35, 0.70)
#         else:  # Beta
#             width_std = np.random.uniform(1.2, 2.5)
#             height_log10 = np.random.uniform(0.30, 0.60)

#         peak_vals = evaluate_gaussian_log_peak(f_canonical, f0=f0, height_log10=height_log10, width_std=width_std)
#         log10_oscillatory += peak_vals
#         peaks_meta.append({
#             "band": band,
#             "f0": f0,
#             "amp": height_log10,
#             "width": width_std,
#             "profile": "gaussian",
#         })

#     log10_clean = log10_aperiodic + log10_oscillatory
#     log_clean = (log10_clean * np.log(10.0)).astype(np.float32)

#     meta = {
#         "peaks": peaks_meta,
#         "chi": chi,
#         "offset": offset,
#         "k_knee": k_knee,
#         "f_knee": f_knee,
#     }
#     return log_clean, meta


def generate_single_eeg_psd_sample(f_canonical, f_min=1.0, f_max=45.0, min_peak_gap=2.5):
    """
    Synthesizes clean log-e PSD with linear additive power dynamics (matching Dataset 1),
    eliminating the negative center-frequency bias caused by log-additive slope-drag.
    """
    # 1. Background (Aperiodic) Component
    chi = np.random.uniform(1.5, 2.5)
    offset = np.random.uniform(1.5, 3.0)
    f_knee = np.random.uniform(0.8, 1.8)
    k_knee = f_knee ** chi

    # Aperiodic curve in log10 space, then converted to physical linear power
    log10_aperiodic = offset - np.log10(k_knee + (np.maximum(f_canonical, 0.05)) ** chi)
    aperiodic_linear = 10.0 ** log10_aperiodic

    # 2. Oscillatory Peaks (Additive in Linear Power)
    candidate_peaks = sample_eeg_peak_frequencies(f_min=f_min, f_max=f_max, min_gap=min_peak_gap)
    oscillatory_linear = np.zeros_like(f_canonical, dtype=np.float64)
    peaks_meta = []

    for peak_info in candidate_peaks:
        f0 = peak_info["f0"]
        band = peak_info["band"]

        if band == "delta":
            width_std = np.random.uniform(0.6, 1.2)
            height_log10 = np.random.uniform(0.40, 0.90)
        elif band == "alpha":
            width_std = np.random.uniform(0.8, 1.8)
            height_log10 = np.random.uniform(0.50, 1.10)
        elif band == "theta":
            width_std = np.random.uniform(0.6, 1.2)
            height_log10 = np.random.uniform(0.35, 0.70)
        else:  # Beta
            width_std = np.random.uniform(1.2, 2.5)
            height_log10 = np.random.uniform(0.30, 0.60)

        # Evaluate Gaussian peak profile normalized in [0, 1]
        gaussian_shape = np.exp(-((f_canonical - f0) ** 2) / (2.0 * width_std**2))

        # Convert peak height to linear excess power added atop the background
        # (height_log10 represents excess height above baseline in log10 space)
        peak_amp_linear = (10.0 ** height_log10 - 1.0) * np.interp(f0, f_canonical, aperiodic_linear)
        peak_linear_signal = peak_amp_linear * gaussian_shape

        oscillatory_linear += peak_linear_signal

        peaks_meta.append({
            "band": band,
            "f0": f0,
            "amp": height_log10,
            "width": width_std,
            "profile": "gaussian",
        })

    # 3. Combine Linearly and Convert to Natural Log (log-e)
    psd_clean_linear = aperiodic_linear + oscillatory_linear
    log_clean = np.log(np.maximum(psd_clean_linear, 1e-12)).astype(np.float32)

    meta = {
        "peaks": peaks_meta,
        "chi": chi,
        "offset": offset,
        "k_knee": k_knee,
        "f_knee": f_knee,
    }
    return log_clean, meta


def generate_and_save_multinoise_dataset(
    filepath="eeg_psd_multinoise_benchmark_aligned.npz",
    n_samples=250,
    n_freqs=250,
    f_min=0.1,      # Exactly matched to Dataset 1
    f_max=50.0,     # Exactly matched to Dataset 1
    min_peak_gap=2.5,
    n_obs_range=(120, 350),
    welch_k_levels=(40, 20, 10, 5),
    lam=1e2,
    max_high_ratio=0.01,
    min_boundary_ratio=0.5,
):
    f_canonical = np.linspace(f_min, f_max, n_freqs, dtype=np.float64)
    n_levels = len(welch_k_levels)

    log_clean_arr = np.zeros((n_samples, n_freqs), dtype=np.float32)
    log_noisy_arr = np.zeros((n_samples, n_levels, n_freqs), dtype=np.float32)
    masks_arr = np.zeros((n_samples, 2, n_freqs), dtype=np.float32)
    metadata_list = []

    print(f"Generating {n_samples} samples across {n_levels} noise levels (K={welch_k_levels})...")
    print(f"Canonical Grid: {n_freqs} bins in [{f_min}, {f_max}] Hz.")

    for i in range(n_samples):
        # 1. Synthesize ground-truth clean curve
        log_clean, meta = generate_single_eeg_psd_sample(
            f_canonical, f_min=1.0, f_max=45.0, min_peak_gap=min_peak_gap
        )
        f0_list = [p["f0"] for p in meta.get("peaks", [])]

        # 2. Extract Interval Boundaries & create target masks
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

        # 3. Interpolate clean PSD to physical observation grid
        n_obs = int(np.random.randint(n_obs_range[0], n_obs_range[1] + 1))
        f_obs = np.linspace(f_min, f_max, n_obs, dtype=np.float64)

        clean_interp = interp1d(f_canonical, log_clean, kind="linear", fill_value="extrapolate")
        log_clean_obs = clean_interp(f_obs)

        # 4. Generate the 4 noisy variants from the same analytical curve
        for lvl_idx, k_seg in enumerate(welch_k_levels):
            noise_mult = np.random.gamma(shape=k_seg, scale=1.0 / k_seg, size=n_obs).astype(np.float64)
            linear_noisy_obs = np.exp(log_clean_obs) * noise_mult
            log_noisy_obs = np.log(np.maximum(linear_noisy_obs, 1e-12))

            recon_interp = interp1d(f_obs, log_noisy_obs, kind="linear", fill_value="extrapolate")
            log_noisy_arr[i, lvl_idx] = recon_interp(f_canonical).astype(np.float32)

        log_clean_arr[i] = log_clean
        masks_arr[i, 0] = peak_mask
        masks_arr[i, 1] = interval_mask

        meta["n_obs"] = n_obs
        metadata_list.append(meta)

    # Save f directly as float64 to prevent any roundoff error for FOOOF
    np.savez_compressed(
        filepath,
        f=f_canonical.astype(np.float64),
        noise_levels_k=np.array(welch_k_levels, dtype=np.int32),
        log_noisy=log_noisy_arr,
        log_clean=log_clean_arr,
        masks=masks_arr,
        metadata=np.array(metadata_list, dtype=object),
    )
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Saved: '{filepath}' ({file_size_mb:.2f} MB, {n_samples} samples x {n_levels} noise levels).")


if __name__ == "__main__":
    generate_and_save_multinoise_dataset(
        filepath="eeg_psd_multinoise_benchmark_aligned.npz",
        n_samples=250,
        n_freqs=250,
        f_min=0.1,
        f_max=50.0,
        welch_k_levels=(40, 20, 10, 5),
    )