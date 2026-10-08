import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from Functions.extract_psd_bumps import extract_intervals_method1_detrended_seed

# =====================================================================
# 1. PEAK PROFILE FUNCTIONS (FULL ASYMPTOTICS PRESERVED)
# =====================================================================
def evaluate_peak_profile(f, f0, amp, width, profile_type="gaussian"):
    """
    Evaluates full asymptotic peak profiles over frequency array f:
      - 'gaussian': Standard bell profile
      - 'lorentzian': Damped harmonic oscillator with heavier algebraic tails
      - 'pseudo_voigt': Linear combination of Gaussian and Lorentzian
    """
    if profile_type == "gaussian":
        # width parameter acts as standard deviation sigma
        return amp * np.exp(-((f - f0)**2) / (2.0 * width**2))

    elif profile_type == "lorentzian":
        # width parameter acts as FWHM gamma
        gamma = width * 2.355  # matched FWHM scale to comparable gaussian
        return amp * (0.5 * gamma)**2 / ((f - f0)**2 + (0.5 * gamma)**2)

    elif profile_type == "pseudo_voigt":
        # Voigt approximation: eta * Lorentzian + (1 - eta) * Gaussian
        eta = np.random.uniform(0.3, 0.7)
        gamma = width * 2.355
        lorentz = amp * (0.5 * gamma)**2 / ((f - f0)**2 + (0.5 * gamma)**2)
        gauss = amp * np.exp(-((f - f0)**2) / (2.0 * width**2))
        return eta * lorentz + (1.0 - eta) * gauss

    else:
        raise ValueError(f"Unknown peak profile type: {profile_type}")


# =====================================================================
# 2. SAMPLING LOGIC & SYNTHESIS
# =====================================================================
def sample_eeg_peak_frequencies(f_min=0.5, f_max=45.0, min_gap=2.0, max_tries=1000):
    """
    Samples EEG peak candidates across physiological rhythm bands:
      - Delta (0.5 - 4.0 Hz): MANDATORY (p = 1.0)
      - Alpha (8.0 - 13.0 Hz): Dominant posterior rhythm (p = 0.90)
      - Theta (4.0 - 8.0 Hz): Low probability (p = 0.20)
      - Beta  (14.0 - 30.0 Hz): Low probability (p = 0.25)
    Ensures minimum separation of `min_gap` Hz between any co-occurring peaks.
    """
    for _ in range(max_tries):
        peaks_to_generate = []

        # 1. Delta candidate (MANDATORY: p = 1.0)
        f0_delta = np.random.uniform(1.2, 3.5)
        peaks_to_generate.append({"band": "delta", "f0": f0_delta})

        # 2. Theta candidate (p = 0.20)
        if np.random.rand() < 0.20:
            f0_theta = np.random.uniform(4.5, 7.5)
            peaks_to_generate.append({"band": "theta", "f0": f0_theta})

        # 3. Alpha candidate (p = 0.90)
        if np.random.rand() < 0.90:
            f0_alpha = np.random.uniform(8.5, 12.5)
            peaks_to_generate.append({"band": "alpha", "f0": f0_alpha})

        # 4. Beta candidate (p = 0.25)
        if np.random.rand() < 0.25:
            f0_beta = np.random.uniform(15.0, 28.0)
            peaks_to_generate.append({"band": "beta", "f0": f0_beta})

        f0_vals = np.array([p["f0"] for p in peaks_to_generate])
        f0_vals.sort()
        if len(f0_vals) == 1 or np.all(np.diff(f0_vals) >= min_gap):
            return peaks_to_generate

    return [{"band": "delta", "f0": 2.2}, {"band": "alpha", "f0": 10.5}]


def generate_single_eeg_psd_sample(f_canonical, f_min=0.5, f_max=45.0, min_peak_gap=2.0):
    """
    Synthesizes an EEG-like clean PSD:
      - Always incorporates a knee parameter k = f_k^chi
      - Steeper decay exponent chi in [1.8, 3.2]
      - Multi-family peak curves (Gaussian, Lorentzian, Pseudo-Voigt) with full asymptotics
    """
    # 1. Aperiodic component: S_ap(f) = 10^b / (k + f^chi)
    chi = np.random.uniform(1.8, 3.2)
    offset = np.random.uniform(1.0, 3.0)
    f_knee = np.random.uniform(2.0, 10.0)
    k_knee = f_knee ** chi

    aperiodic_linear = (10.0**offset) / (k_knee + (np.maximum(f_canonical, 0.1))**chi)
    aperiodic_linear += 1e-4

    # 2. Oscillatory peaks with varied shapes and full asymptotic tails
    candidate_peaks = sample_eeg_peak_frequencies(f_min=f_min, f_max=f_max, min_gap=min_peak_gap)
    oscillatory_linear = np.zeros_like(f_canonical)
    peaks_meta = []

    profile_choices = ["gaussian", "lorentzian", "pseudo_voigt"]

    for peak_info in candidate_peaks:
        f0 = peak_info["f0"]
        band = peak_info["band"]
        profile = np.random.choice(profile_choices, p=[0.50, 0.30, 0.20])

        if band == "delta":
            width = np.random.uniform(0.4, 0.8)
            amp = np.random.uniform(1.5, 4.5)
        elif band == "alpha":
            width = np.random.uniform(0.6, 1.4)
            amp = np.random.uniform(2.0, 6.0)
        elif band == "theta":
            width = np.random.uniform(0.5, 0.9)
            amp = np.random.uniform(0.2, 0.8)
        else:  # Beta
            width = np.random.uniform(1.0, 2.2)
            amp = np.random.uniform(0.1, 0.6)

        # Evaluated across the entire grid (no truncation)
        peak_vals = evaluate_peak_profile(f_canonical, f0=f0, amp=amp, width=width, profile_type=profile)
        oscillatory_linear += peak_vals
        peaks_meta.append({"band": band, "f0": f0, "amp": amp, "width": width, "profile": profile})

    clean_linear = aperiodic_linear + oscillatory_linear
    log_clean = np.log(np.maximum(clean_linear, 1e-12)).astype(np.float32)

    meta = {
        "peaks": peaks_meta,
        "chi": chi,
        "offset": offset,
        "k_knee": k_knee,
        "f_knee": f_knee,
    }
    return log_clean, meta


# =====================================================================
# 3. SINGLE DATASET GENERATOR (100 SAMPLES, NO SPLITS)
# =====================================================================
def generate_and_save_eeg_dataset(
    filepath="eeg_psd_benchmark.npz",
    n_samples=100,
    n_freqs=250,
    f_min=0.5,
    f_max=45.0,
    min_peak_gap=2.0,
    n_obs_range=(80, 350),
    k_welch_range=(8, 40),
    lam=1e2,
    max_high_ratio=0.01,
    min_boundary_ratio=0.5,
):
    f_canonical = np.linspace(f_min, f_max, n_freqs, dtype=np.float32)

    log_clean_arr = np.zeros((n_samples, n_freqs), dtype=np.float32)
    log_noisy_arr = np.zeros((n_samples, n_freqs), dtype=np.float32)
    masks_arr = np.zeros((n_samples, 2, n_freqs), dtype=np.float32)
    metadata_list = []

    print(f"Generating {n_samples} EEG samples (mixed peak profiles, full asymptotics)...")

    for i in range(n_samples):
        # 1. Synthesize EEG ground-truth spectrum
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

        # 3. Simulate variable physical observation length N_obs
        n_obs = int(np.random.randint(n_obs_range[0], n_obs_range[1] + 1))
        f_obs = np.linspace(f_min, f_max, n_obs, dtype=np.float32)

        clean_interp = interp1d(f_canonical, log_clean, kind="linear", fill_value="extrapolate")
        log_clean_obs = clean_interp(f_obs)

        # 4. Inject Welch Gamma noise at N_obs
        k_seg = int(np.random.randint(k_welch_range[0], k_welch_range[1] + 1))
        noise_mult = np.random.gamma(shape=k_seg, scale=1.0 / k_seg, size=n_obs).astype(np.float32)
        linear_noisy_obs = np.exp(log_clean_obs) * noise_mult
        log_noisy_obs = np.log(np.maximum(linear_noisy_obs, 1e-12))

        # 5. Interpolate back to canonical grid (250 bins)
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
        f=f_canonical,
        log_noisy=log_noisy_arr,
        log_clean=log_clean_arr,
        masks=masks_arr,
        metadata=np.array(metadata_list, dtype=object),
    )
    file_size_kb = os.path.getsize(filepath) / 1024
    print(f"Saved: '{filepath}' ({file_size_kb:.1f} KB, {n_samples} samples).")

# =====================================================================
# 3. VISUALIZATION UTILITY (ADAPTED FOR SINGLE DATASET)
# =====================================================================
def plot_psd_mask_sample(f, log_noisy, log_clean, center_mask, interval_mask, sample_idx=0):
    """
    Plots a single sample:
    1. Log PSD (Noisy input + Clean target + Center Mask + Interval Shading)
    2. Linear PSD representation
    3. Dual Mask Channels (Channel 0: Centers, Channel 1: Intervals)
    """
    psd_noisy = np.exp(log_noisy)
    psd_clean = np.exp(log_clean)

    peak_indices = np.where(center_mask > 0.5)[0]
    peak_freqs = f[peak_indices]
    n_peaks = len(peak_freqs)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(11, 10), sharex=True, 
        gridspec_kw={'height_ratios': [2, 2, 1.2]}
    )

    # -----------------------------------------------------------------
    # SUBPLOT 1: LOG PSD
    # -----------------------------------------------------------------
    ax1.plot(f, log_noisy, color="gray", alpha=0.4, lw=1.2, label="Noisy Input Log PSD")
    ax1.plot(f, log_clean, color="blue", lw=1.8, label="Clean Log PSD Target")

    for idx, f0 in enumerate(peak_freqs):
        ax1.axvline(f0, color="red", linestyle="--", alpha=0.8, 
                    label="Peak Center (Mask Ch 0)" if idx == 0 else "")

    ax1.fill_between(f, log_noisy.min(), log_noisy.max(), where=(interval_mask > 0.5), 
                     color="orange", alpha=0.25, label="Interval Mask (Ch 1)")

    ax1.set_ylabel("Log PSD")
    ax1.set_title(f"Sample #{sample_idx} — EEG Log PSD with Overlays (Peaks: {n_peaks})")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc="upper right")

    # -----------------------------------------------------------------
    # SUBPLOT 2: LINEAR PSD
    # -----------------------------------------------------------------
    ax2.plot(f, psd_noisy, color="gray", alpha=0.4, lw=1.2, label="Noisy Input Linear PSD")
    ax2.plot(f, psd_clean, color="blue", lw=1.8, label="Clean Linear PSD Target")

    for idx, f0 in enumerate(peak_freqs):
        ax2.axvline(f0, color="red", linestyle="--", alpha=0.8)

    ax2.fill_between(f, 0, psd_noisy.max(), where=(interval_mask > 0.5), 
                     color="orange", alpha=0.25)

    ax2.set_ylabel("Linear Power")
    ax2.set_title("Linear PSD Domain Overlay")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # -----------------------------------------------------------------
    # SUBPLOT 3: DUAL MASK CHANNELS
    # -----------------------------------------------------------------
    ax3.plot(f, center_mask, color="red", lw=1.5, label="Ch 0: Peak Center Heatmap")
    ax3.plot(f, interval_mask, color="orange", lw=1.5, linestyle="-", label="Ch 1: Interval Range Mask")
    
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Mask Value")
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_title("Target Mask Channels")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


# =====================================================================
# 4. EXECUTION / VISUALIZATION DEMO
# =====================================================================
if __name__ == "__main__":
    DATASET_PATH = "eeg_psd_benchmark.npz"

    # Generate the single dataset (run once)
    generate_and_save_eeg_dataset(
        filepath=DATASET_PATH,
        n_samples=100,
        n_freqs=250,
        f_min=0.5,
        f_max=45.0,
        min_peak_gap=2.0,
    )
