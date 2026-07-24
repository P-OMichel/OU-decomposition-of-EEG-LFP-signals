import os
import matplotlib.pyplot as plt
import numpy as np


def plot_dataset_examples(
    filepath="psd_dataset_splits.npz", split="train", num_plots=3
):
    """Loads a specific dataset split (train, val, or test) from the saved archive

    and plots random samples comparing noisy input against clean ground truth.

    Parameters
    ----------
    filepath : str
        Path to the .npz archive containing dataset splits.
    split : str
        Which split to sample from ('train', 'val', or 'test').
    num_plots : int
        Number of random sample curves to display.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file '{filepath}' not found!")

    # 1. Load archive
    data = np.load(filepath, allow_pickle=True)
    f = data["f"]

    noisy_key = f"log_noisy_{split}"
    clean_key = f"log_clean_{split}"
    meta_key = f"metadata_{split}"

    if noisy_key not in data:
        raise KeyError(
            f"Split '{split}' not found in archive. Available keys: {list(data.keys())}"
        )

    log_noisy_arr = data[noisy_key]
    log_clean_arr = data[clean_key]
    metadata_arr = data[meta_key]

    total_samples = len(log_noisy_arr)
    print(
        f"Loaded '{split}' split containing {total_samples} samples across {len(f)} frequency bins."
    )

    # 2. Sample random indices
    random_indices = np.random.choice(
        total_samples, size=num_plots, replace=False
    )

    # 3. Create plots
    fig, axes = plt.subplots(
        num_plots, 1, figsize=(11, 3.8 * num_plots), sharex=True
    )
    if num_plots == 1:
        axes = [axes]

    for ax, idx in zip(axes, random_indices):
        log_noisy = log_noisy_arr[idx]
        log_clean = log_clean_arr[idx]
        meta = metadata_arr[idx]

        # Plot curves
        ax.plot(
            f,
            log_noisy,
            color="gray",
            alpha=0.5,
            linewidth=1.0,
            label="Noisy Input",
        )
        ax.plot(
            f,
            log_clean,
            color="crimson",
            linewidth=2.0,
            label="Ground Truth Clean PSD",
        )

        # Draw vertical lines for each peak frequency f0
        for p_info in meta["peaks"]:
            f0 = p_info["f0"]
            shape_type = p_info["shape"]
            ax.axvline(
                x=f0,
                color="navy",
                linestyle="--",
                alpha=0.6,
                label=f"f0={f0:.1f}Hz ({shape_type})",
            )

        ax.set_ylabel("Log Power (a.u.)")
        knee_str = "Yes" if meta["has_knee"] else "No"
        ax.set_title(
            f"Split: '{split}' | Sample #{idx} | Peaks: {meta['n_peaks']} | "
            f"Noise Std: {meta['noise_level']:.2f} | 1/f Slope χ: {meta['chi']:.2f} | Knee: {knee_str}"
        )
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()


# =====================================================================
# EXAMPLE USAGE
# =====================================================================
if __name__ == "__main__":
    DATASET_PATH = "psd_dataset_splits.npz"

    # Display 3 random samples from the training set
    plot_dataset_examples(filepath=DATASET_PATH, split="train", num_plots=3)

    # You can also inspect the test set:
    # plot_dataset_examples(filepath=DATASET_PATH, split="test", num_plots=3)d