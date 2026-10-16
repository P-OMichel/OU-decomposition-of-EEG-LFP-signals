
import os
import numpy as np
import matplotlib.pyplot as plt


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

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), sharex=True, gridspec_kw={'height_ratios': [2, 2, 1.2]})

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
    DATASET_PATH = "eeg_psd_benchmark_fooof_advantage.npz" # eeg_psd_benchmark.npz


    # Step 2: Load and plot directly from the single flat arrays
    if os.path.exists(DATASET_PATH):
        data = np.load(DATASET_PATH, allow_pickle=True)
        f = data["f"]
        log_noisy = data["log_noisy"]
        log_clean = data["log_clean"]
        masks = data["masks"]

        print(f"\nLoaded '{DATASET_PATH}' successfully:")
        print(f" - Samples     : {len(log_noisy)}")
        print(f" - Grid Range  : [{f[0]:.1f} Hz, {f[-1]:.1f} Hz] ({len(f)} bins)")
        print(f" - Masks Shape : {masks.shape}")

        # Plot first 5 samples
        for i in range(5):
            plot_psd_mask_sample(
                f=f,
                log_noisy=log_noisy[i],
                log_clean=log_clean[i],
                center_mask=masks[i, 0],
                interval_mask=masks[i, 1],
                sample_idx=i,
            )