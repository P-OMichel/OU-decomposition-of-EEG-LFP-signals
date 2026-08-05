import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# VISUALIZATION UTILITY FOR MASK DATASET / MODEL INFERENCE
# =====================================================================
def plot_psd_mask_sample(f, log_noisy, log_clean, center_mask, interval_mask, sample_idx=0):
    """
    Plots a sample showing:
    1. Log PSD (Noisy input + Clean target + Center Mask + Interval Shading)
    2. Linear PSD representation
    3. Dual Mask Channels (Channel 0: Centers, Channel 1: Intervals)
    """
    # Linear PSDs for linear domain visualization
    psd_noisy = np.exp(log_noisy)
    psd_clean = np.exp(log_clean)

    # Extract detected peak positions & overall interval boundaries directly from masks
    peak_indices = np.where(center_mask > 0.5)[0]
    peak_freqs = f[peak_indices]
    n_peaks = len(peak_freqs)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), sharex=True, 
                                        gridspec_kw={'height_ratios': [2, 2, 1.2]})

    # -----------------------------------------------------------------
    # SUBPLOT 1: LOG PSD FIT WITH MASK OVERLAYS
    # -----------------------------------------------------------------
    ax1.plot(f, log_noisy, color="gray", alpha=0.4, lw=1.2, label="Noisy Input Log PSD")
    ax1.plot(f, log_clean, color="blue", lw=1.8, label="Clean Log PSD Target")

    # Draw Peak Center Lines (from Channel 0)
    for idx, f0 in enumerate(peak_freqs):
        ax1.axvline(f0, color="red", linestyle="--", alpha=0.8, 
                    label="Peak Center (Mask Ch 0)" if idx == 0 else "")

    # Highlight Frequency Intervals (from Channel 1)
    # Using np.ma.masked_where to shade contiguous active regions
    shaded_interval = np.ma.masked_where(interval_mask < 0.5, log_clean)
    ax1.fill_between(f, log_noisy.min(), log_noisy.max(), where=(interval_mask > 0.5), 
                     color="orange", alpha=0.25, label="Interval Mask (Ch 1)")

    ax1.set_ylabel("Log PSD")
    ax1.set_title(f"Sample #{sample_idx} — Log PSD with Mask Overlays (Detected Peaks: {n_peaks})")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Clean up duplicate legend entries
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc="upper right")

    # -----------------------------------------------------------------
    # SUBPLOT 2: LINEAR PSD FIT WITH MASK OVERLAYS
    # -----------------------------------------------------------------
    ax2.plot(f, psd_noisy, color="gray", alpha=0.4, lw=1.2, label="Noisy Input Linear PSD")
    ax2.plot(f, psd_clean, color="blue", lw=1.8, label="Clean Linear PSD Target")

    for idx, f0 in enumerate(peak_freqs):
        ax2.axvline(f0, color="red", linestyle="--", alpha=0.8)

    ax2.fill_between(f, 0, psd_noisy.max(), where=(interval_mask > 0.5), 
                     color="orange", alpha=0.25)

    ax2.set_ylabel("Linear Power")
    ax2.set_title(f"Linear PSD Domain Overlay")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # -----------------------------------------------------------------
    # SUBPLOT 3: DUAL MASK CHANNELS VISUALIZATION
    # -----------------------------------------------------------------
    ax3.plot(f, center_mask, color="red", lw=1.5, label="Ch 0: Peak Center Heatmap")
    ax3.plot(f, interval_mask, color="orange", lw=1.5, linestyle="-", label="Ch 1: Interval Range Mask")
    
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Mask Value")
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_title("Target Mask Channels (Network Target Outputs)")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


# =====================================================================
# DEMO EXECUTION (Inspects First Few Dataset Samples)
# =====================================================================
if __name__ == "__main__":
    DATASET_PATH = "psd_dataset_masks.npz"

    # 1. Load the Mask Dataset
    try:
        data = np.load(DATASET_PATH, allow_pickle=True)
        f = data["f"]
        log_noisy_train = data["log_noisy_train"]
        log_clean_train = data["log_clean_train"]
        masks_train = data["masks_train"]  # Shape: (N_samples, 2, n_freqs)
        
        print(f"Loaded '{DATASET_PATH}' successfully!")
        print(f" - Frequency Grid Range : [{f[0]:.1f} Hz, {f[-1]:.1f} Hz] ({len(f)} bins)")
        print(f" - Mask Array Shape     : {masks_train.shape}")
        
        # 2. Plot the first 3 samples
        for i in range(12,13):
            plot_psd_mask_sample(
                f=f,
                log_noisy=log_noisy_train[i],
                log_clean=log_clean_train[i],
                center_mask=masks_train[i, 0],   # Channel 0: Peak Center Mask
                interval_mask=masks_train[i, 1], # Channel 1: Interval Range Mask
                sample_idx=i
            )

    except FileNotFoundError:
        print(f"File '{DATASET_PATH}' not found. Please run the mask dataset generator script first!")