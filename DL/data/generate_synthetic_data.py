import os
import numpy as np
import torch
from torch.utils.data import Dataset
from Functions.extract_psd_bumps import extract_intervals_method1_detrended_seed


# =====================================================================
# 1. CORE SAMPLE GENERATOR
# =====================================================================
def generate_single_psd_sample(f, n_freqs, f_max):
    """Generates a single (log_noisy, log_clean, metadata) tuple."""
    # A. Background Component (1/f^chi with optional Knee)
    chi = np.random.uniform(0.8, 3.0)
    b_amp = np.random.uniform(0.5, 3.0)
    has_knee = np.random.choice([True, False], p=[0.4, 0.6])

    if has_knee:
        f_knee = np.random.uniform(2.0, 25.0)
        bg = b_amp / (f_knee**chi + f**chi)
    else:
        bg = b_amp / (f**chi)

    psd_clean = bg.copy()

    # B. Peaks Injection (0 to 4 peaks, mixed models)
    n_peaks = np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.3, 0.3, 0.2, 0.1])
    peak_params = []
    peak_shape_names = [
        "pseudo_voigt",
        "ornstein_uhlenbeck",
        "pearson_iv",
        "student_t",
    ]

    for _ in range(n_peaks):
        shape_type = np.random.choice(peak_shape_names)
        f0 = np.random.uniform(0.5, f_max - 10.0)  # Includes Delta band
        A = np.random.uniform(0.3, 4.0)

        # Dynamic gamma scaling for low frequencies  # NOTE: Old version
        if f0 < 4.0:
            gamma = np.random.uniform(0.1, max(0.2, 0.6 * f0))
        else:
            is_sharp = np.random.choice([True, False], p=[0.5, 0.5])
            gamma = (
                np.random.uniform(0.2, 1.2)
                if is_sharp
                else np.random.uniform(1.5, 5.0)
           )



        if shape_type == "pseudo_voigt":
            eta = np.random.uniform(0.0, 1.0)
            sigma = gamma / np.sqrt(2 * np.log(2))
            f_pos, f_neg = f - f0, f + f0
            L_pos = (1.0 / np.pi) * (gamma / (f_pos**2 + gamma**2))
            L_neg = (1.0 / np.pi) * (gamma / (f_neg**2 + gamma**2))
            G_pos = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * (f_pos / sigma) ** 2
            )
            G_neg = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * (f_neg / sigma) ** 2
            )
            peak_signal = A**2 * (
                eta * L_pos + (1 - eta) * G_pos + eta * L_neg + (1 - eta) * G_neg
            )
            p_dict = {
                "shape": shape_type,
                "f0": f0,
                "A": A,
                "gamma": gamma,
                "eta": eta,
            }

        elif shape_type == "ornstein_uhlenbeck":
            lbda = max(0.1, gamma)
            Omega, omega0 = 2.0 * np.pi * f, 2.0 * np.pi * f0
            peak_signal = (A**2 / np.pi) * (
                1.0 / (lbda**2 + (Omega - omega0) ** 2)
                + 1.0 / (lbda**2 + (Omega + omega0) ** 2)
            )
            p_dict = {"shape": shape_type, "f0": f0, "A": A, "lbda": lbda}

        elif shape_type == "pearson_iv":
            m = np.random.uniform(1.1, 4.0)
            nu_p = np.random.uniform(-2.0, 2.0)
            y_pos, y_neg = (f - f0) / gamma, (f + f0) / gamma
            term_pos = (1.0 + y_pos**2) ** (-m) * np.exp(
                -nu_p * np.arctan(y_pos)
            )
            term_neg = (1.0 + y_neg**2) ** (-m) * np.exp(
                -nu_p * np.arctan(y_neg)
            )
            peak_signal = A**2 * (term_pos + term_neg)
            p_dict = {
                "shape": shape_type,
                "f0": f0,
                "A": A,
                "gamma": gamma,
                "m": m,
                "nu_p": nu_p,
            }

        elif shape_type == "student_t":
            alpha = np.random.uniform(1.2, 4.0)
            term_pos = 1.0 + ((f - f0) / gamma) ** 2
            term_neg = 1.0 + ((f + f0) / gamma) ** 2
            peak_signal = (A**2 / np.pi) * (
                (1.0 / term_pos ** (alpha / 2.0))
                + (1.0 / term_neg ** (alpha / 2.0))
            )
            p_dict = {
                "shape": shape_type,
                "f0": f0,
                "A": A,
                "gamma": gamma,
                "alpha": alpha,
            }

        # Optional post-peak decay tail
        has_post_decay = np.random.choice([True, False], p=[0.3, 0.7])
        if has_post_decay and f0 > 2.0:
            beta_decay = np.random.uniform(1.0, 3.5)
            f_safe = np.maximum(f, 1e-6)
            post_decay = (
                (A**2 * 0.3)
                * ((f_safe / f0) ** (-beta_decay))
                * np.maximum(0.0, 1.0 - (f0 / f_safe))
            )
            peak_signal += np.where(f > f0, post_decay, 0.0)
            p_dict["beta_decay"] = beta_decay

        psd_clean += peak_signal
        peak_params.append(p_dict)

    # C. Noise Injection
    noise_level = np.random.uniform(0.05, 0.50)
    log_noise = np.random.normal(loc=0.0, scale=noise_level, size=n_freqs)

    log_clean = np.log(psd_clean + 1e-12)
    log_noisy = log_clean + log_noise

    meta = {
        "n_peaks": n_peaks,
        "b_amp": float(b_amp),
        "chi": float(chi),
        "has_knee": has_knee,
        "noise_level": float(noise_level),
        "peaks": peak_params,
    }

    return log_noisy, log_clean, meta

def generate_single_psd_sample_high_amplitude(f, n_freqs, f_max):
    """Generates a single (log_noisy, log_clean, metadata) tuple."""
    # A. Background Component (1/f^chi with optional Knee)
    chi = np.random.uniform(0.8, 3.0)
    b_amp = np.random.uniform(0.5, 3.0)
    has_knee = np.random.choice([True, False], p=[0.4, 0.6])

    if has_knee:
        f_knee = np.random.uniform(2.0, 25.0)
        bg = b_amp / (f_knee**chi + f**chi)
    else:
        bg = b_amp / (f**chi)

    psd_clean = bg.copy()

    # B. Peaks Injection (0 to 4 peaks, mixed models)
    n_peaks = np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.3, 0.3, 0.2, 0.1])
    peak_params = []
    peak_shape_names = [
        "pseudo_voigt",
        "ornstein_uhlenbeck",
        "pearson_iv",
        "student_t",
    ]

    for _ in range(n_peaks):
        shape_type = np.random.choice(peak_shape_names)
        f0 = np.random.uniform(0.5, f_max - 10.0)  # Includes Delta band
        A = np.random.uniform(0.3, 4.0)

        # Modified peak parameter generation block: # NOTE: peak for low frequency can have enhanced amplitude
        if f0 < 4.0:
            # Allow larger amplitudes for slow-wave delta peaks (anesthesia/sleep)
            A = np.random.uniform(1.0, 8.0)
            gamma = np.random.uniform(0.3, max(0.4, 0.8 * f0))
        else:
            A = np.random.uniform(0.3, 4.0)
            is_sharp = np.random.choice([True, False], p=[0.5, 0.5])
            gamma = (
                np.random.uniform(0.2, 1.2) if is_sharp else np.random.uniform(1.5, 5.0)
            )

        if shape_type == "pseudo_voigt":
            eta = np.random.uniform(0.0, 1.0)
            sigma = gamma / np.sqrt(2 * np.log(2))
            f_pos, f_neg = f - f0, f + f0
            L_pos = (1.0 / np.pi) * (gamma / (f_pos**2 + gamma**2))
            L_neg = (1.0 / np.pi) * (gamma / (f_neg**2 + gamma**2))
            G_pos = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * (f_pos / sigma) ** 2
            )
            G_neg = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * (f_neg / sigma) ** 2
            )
            peak_signal = A**2 * (
                eta * L_pos + (1 - eta) * G_pos + eta * L_neg + (1 - eta) * G_neg
            )
            p_dict = {
                "shape": shape_type,
                "f0": f0,
                "A": A,
                "gamma": gamma,
                "eta": eta,
            }

        elif shape_type == "ornstein_uhlenbeck":
            lbda = max(0.1, gamma)
            Omega, omega0 = 2.0 * np.pi * f, 2.0 * np.pi * f0
            peak_signal = (A**2 / np.pi) * (
                1.0 / (lbda**2 + (Omega - omega0) ** 2)
                + 1.0 / (lbda**2 + (Omega + omega0) ** 2)
            )
            p_dict = {"shape": shape_type, "f0": f0, "A": A, "lbda": lbda}

        elif shape_type == "pearson_iv":
            m = np.random.uniform(1.1, 4.0)
            nu_p = np.random.uniform(-2.0, 2.0)
            y_pos, y_neg = (f - f0) / gamma, (f + f0) / gamma
            term_pos = (1.0 + y_pos**2) ** (-m) * np.exp(
                -nu_p * np.arctan(y_pos)
            )
            term_neg = (1.0 + y_neg**2) ** (-m) * np.exp(
                -nu_p * np.arctan(y_neg)
            )
            peak_signal = A**2 * (term_pos + term_neg)
            p_dict = {
                "shape": shape_type,
                "f0": f0,
                "A": A,
                "gamma": gamma,
                "m": m,
                "nu_p": nu_p,
            }

        elif shape_type == "student_t":
            alpha = np.random.uniform(1.2, 4.0)
            term_pos = 1.0 + ((f - f0) / gamma) ** 2
            term_neg = 1.0 + ((f + f0) / gamma) ** 2
            peak_signal = (A**2 / np.pi) * (
                (1.0 / term_pos ** (alpha / 2.0))
                + (1.0 / term_neg ** (alpha / 2.0))
            )
            p_dict = {
                "shape": shape_type,
                "f0": f0,
                "A": A,
                "gamma": gamma,
                "alpha": alpha,
            }

        # Optional post-peak decay tail
        has_post_decay = np.random.choice([True, False], p=[0.3, 0.7])
        if has_post_decay and f0 > 2.0:
            beta_decay = np.random.uniform(1.0, 3.5)
            f_safe = np.maximum(f, 1e-6)
            post_decay = (
                (A**2 * 0.3)
                * ((f_safe / f0) ** (-beta_decay))
                * np.maximum(0.0, 1.0 - (f0 / f_safe))
            )
            peak_signal += np.where(f > f0, post_decay, 0.0)
            p_dict["beta_decay"] = beta_decay

        psd_clean += peak_signal
        peak_params.append(p_dict)

    # C. Noise Injection
    noise_level = np.random.uniform(0.05, 0.50)
    log_noise = np.random.normal(loc=0.0, scale=noise_level, size=n_freqs)

    log_clean = np.log(psd_clean + 1e-12)
    log_noisy = log_clean + log_noise

    meta = {
        "n_peaks": n_peaks,
        "b_amp": float(b_amp),
        "chi": float(chi),
        "has_knee": has_knee,
        "noise_level": float(noise_level),
        "peaks": peak_params,
    }

    return log_noisy, log_clean, meta

# =====================================================================
# 2. SPLIT DATASET GENERATOR & SAVER
# =====================================================================
def generate_and_save_split_dataset(
    filepath="psd_dataset_splits.npz",
    n_train=20000,
    n_val=2500,
    n_test=2500,
    n_freqs=500,
    f_max=100.0,
):
    """Generates train, val, and test splits and saves them into one archive."""
    f = np.linspace(0.1, f_max, n_freqs)
    splits = {"train": n_train, "val": n_val, "test": n_test}
    data_dict = {"f": f}

    print("Generating train/val/test dataset...")

    for split_name, count in splits.items():
        print(f" -> Generating {split_name} set ({count} samples)...")
        log_clean_arr = np.zeros((count, n_freqs))
        log_noisy_arr = np.zeros((count, n_freqs))
        meta_list = []

        for i in range(count):
            # log_noisy, log_clean, meta = generate_single_psd_sample(
            #     f, n_freqs, f_max
            # )
            log_noisy, log_clean, meta = generate_single_psd_sample_high_amplitude(
                f, n_freqs, f_max
            )
            log_noisy_arr[i] = log_noisy
            log_clean_arr[i] = log_clean
            meta_list.append(meta)

        data_dict[f"log_noisy_{split_name}"] = log_noisy_arr
        data_dict[f"log_clean_{split_name}"] = log_clean_arr
        data_dict[f"metadata_{split_name}"] = np.array(
            meta_list, dtype=object
        )

    # Save to compressed file
    np.savez_compressed(filepath, **data_dict)
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Done! Dataset saved to '{filepath}' ({file_size_mb:.2f} MB)")


# =====================================================================
# 3. PYTORCH DATASET LOADER
# =====================================================================
class PSDSplitDataset(Dataset):
    """PyTorch Dataset wrapper for loading specific splits (train/val/test)."""

    def __init__(self, filepath="psd_dataset_splits.npz", split="train"):
        assert split in [
            "train",
            "val",
            "test",
        ], "Split must be 'train', 'val', or 'test'"

        data = np.load(filepath, allow_pickle=True)
        self.f = data["f"]
        self.log_noisy = torch.tensor(
            data[f"log_noisy_{split}"], dtype=torch.float32
        ).unsqueeze(1)
        self.log_clean = torch.tensor(
            data[f"log_clean_{split}"], dtype=torch.float32
        ).unsqueeze(1)
        self.metadata = data[f"metadata_{split}"]

    def __len__(self):
        return len(self.log_noisy)

    def __getitem__(self, idx):
        # Returns shape (1, n_freqs) for 1D CNNs
        return self.log_noisy[idx], self.log_clean[idx]

# # =====================================================================
# # EXECUTION DEMO
# # =====================================================================
# if __name__ == "__main__":
#     print('Generate data')
#     DATASET_PATH = "psd_dataset_splits_high_amplitude.npz"

#     # 1. Generate 20,000 train, 2,500 val, 2,500 test samples
#     generate_and_save_split_dataset(
#         filepath=DATASET_PATH,
#         n_train=20000,
#         n_val=2500,
#         n_test=2500,
#         n_freqs=250,
#         f_max=50,
#     )

#     # 2. Example PyTorch loading
#     train_dataset = PSDSplitDataset(filepath=DATASET_PATH, split="train")
#     val_dataset = PSDSplitDataset(filepath=DATASET_PATH, split="val")
#     test_dataset = PSDSplitDataset(filepath=DATASET_PATH, split="test")

#     print(f"\nPyTorch Datasets Ready:")
#     print(f" - Train samples: {len(train_dataset)}")
#     print(f" - Val samples:   {len(val_dataset)}")
#     print(f" - Test samples:  {len(test_dataset)}")


def generate_single_psd_sample_high_amplitude(f, n_freqs, f_max):
    """Generates a single (log_noisy, log_clean, metadata) tuple."""
    chi = np.random.uniform(0.8, 3.0)
    b_amp = np.random.uniform(0.5, 3.0)
    has_knee = np.random.choice([True, False], p=[0.4, 0.6])

    if has_knee:
        f_knee = np.random.uniform(2.0, 25.0)
        bg = b_amp / (f_knee**chi + f**chi)
    else:
        bg = b_amp / (f**chi)

    psd_clean = bg.copy()

    n_peaks = np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.3, 0.3, 0.2, 0.1])
    peak_params = []
    peak_shape_names = ["pseudo_voigt", "ornstein_uhlenbeck", "pearson_iv", "student_t"]

    for _ in range(n_peaks):
        shape_type = np.random.choice(peak_shape_names)
        f0 = np.random.uniform(0.5, f_max - 10.0)

        if f0 < 4.0:
            A = np.random.uniform(1.0, 8.0)
            gamma = np.random.uniform(0.3, max(0.4, 0.8 * f0))
        else:
            A = np.random.uniform(0.3, 4.0)
            is_sharp = np.random.choice([True, False], p=[0.5, 0.5])
            gamma = np.random.uniform(0.2, 1.2) if is_sharp else np.random.uniform(1.5, 5.0)

        if shape_type == "pseudo_voigt":
            eta = np.random.uniform(0.0, 1.0)
            sigma = gamma / np.sqrt(2 * np.log(2))
            f_pos, f_neg = f - f0, f + f0
            L_pos = (1.0 / np.pi) * (gamma / (f_pos**2 + gamma**2))
            L_neg = (1.0 / np.pi) * (gamma / (f_neg**2 + gamma**2))
            G_pos = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (f_pos / sigma) ** 2)
            G_neg = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (f_neg / sigma) ** 2)
            peak_signal = A**2 * (eta * L_pos + (1 - eta) * G_pos + eta * L_neg + (1 - eta) * G_neg)
            p_dict = {"shape": shape_type, "f0": f0, "A": A, "gamma": gamma, "eta": eta}

        elif shape_type == "ornstein_uhlenbeck":
            lbda = max(0.1, gamma)
            Omega, omega0 = 2.0 * np.pi * f, 2.0 * np.pi * f0
            peak_signal = (A**2 / np.pi) * (
                1.0 / (lbda**2 + (Omega - omega0) ** 2) + 1.0 / (lbda**2 + (Omega + omega0) ** 2)
            )
            p_dict = {"shape": shape_type, "f0": f0, "A": A, "lbda": lbda}

        elif shape_type == "pearson_iv":
            m = np.random.uniform(1.1, 4.0)
            nu_p = np.random.uniform(-2.0, 2.0)
            y_pos, y_neg = (f - f0) / gamma, (f + f0) / gamma
            term_pos = (1.0 + y_pos**2) ** (-m) * np.exp(-nu_p * np.arctan(y_pos))
            term_neg = (1.0 + y_neg**2) ** (-m) * np.exp(-nu_p * np.arctan(y_neg))
            peak_signal = A**2 * (term_pos + term_neg)
            p_dict = {"shape": shape_type, "f0": f0, "A": A, "gamma": gamma, "m": m, "nu_p": nu_p}

        elif shape_type == "student_t":
            alpha = np.random.uniform(1.2, 4.0)
            term_pos = 1.0 + ((f - f0) / gamma) ** 2
            term_neg = 1.0 + ((f + f0) / gamma) ** 2
            peak_signal = (A**2 / np.pi) * ((1.0 / term_pos ** (alpha / 2.0)) + (1.0 / term_neg ** (alpha / 2.0)))
            p_dict = {"shape": shape_type, "f0": f0, "A": A, "gamma": gamma, "alpha": alpha}

        has_post_decay = np.random.choice([True, False], p=[0.3, 0.7])
        if has_post_decay and f0 > 2.0:
            beta_decay = np.random.uniform(1.0, 3.5)
            f_safe = np.maximum(f, 1e-6)
            post_decay = (A**2 * 0.3) * ((f_safe / f0) ** (-beta_decay)) * np.maximum(0.0, 1.0 - (f0 / f_safe))
            peak_signal += np.where(f > f0, post_decay, 0.0)
            p_dict["beta_decay"] = beta_decay

        psd_clean += peak_signal
        peak_params.append(p_dict)

    noise_level = np.random.uniform(0.05, 0.50)
    log_noise = np.random.normal(loc=0.0, scale=noise_level, size=n_freqs)

    log_clean = np.log(psd_clean + 1e-12)
    log_noisy = log_clean + log_noise

    meta = {
        "n_peaks": n_peaks,
        "b_amp": float(b_amp),
        "chi": float(chi),
        "has_knee": has_knee,
        "noise_level": float(noise_level),
        "peaks": peak_params,
    }

    return log_noisy, log_clean, meta


# NOTE: new dataset with intervals of freqeuncy for the peaks

# =====================================================================
# 3. SPLIT DATASET GENERATOR & SAVER (SORTED & NORMALIZED TARGETS)
# =====================================================================
def generate_and_save_split_dataset_with_intervals(
    filepath="psd_dataset_intervals_ha.npz",
    n_train=20000,
    n_val=2500,
    n_test=2500,
    n_freqs=250,
    f_max=50.0,
    max_peaks=4,
    lam=1e2,
    max_high_ratio=0.01,
    min_boundary_ratio=0.5,
):
    """Generates train, val, and test splits with precomputed Method 1 intervals saved in NPZ."""
    f = np.linspace(0.1, f_max, n_freqs)
    splits = {"train": n_train, "val": n_val, "test": n_test}
    data_dict = {"f": f}

    print("Generating train/val/test dataset with precomputed Method 1 intervals...")

    for split_name, count in splits.items():
        print(f" -> Processing {split_name} set ({count} samples)...")
        log_clean_arr = np.zeros((count, n_freqs), dtype=np.float32)
        log_noisy_arr = np.zeros((count, n_freqs), dtype=np.float32)
        targets_arr = np.zeros((count, max_peaks, 3), dtype=np.float32)  # Normalized [f0, f_left, f_right] in [0, 1]
        masks_arr = np.zeros((count, max_peaks), dtype=np.float32)        # Mask
        peak_counts_arr = np.zeros(count, dtype=np.float32)              # Number of peaks
        meta_list = []

        for i in range(count):
            log_noisy, log_clean, meta = generate_single_psd_sample_high_amplitude(f, n_freqs, f_max)
            
            # -------------------------------------------------------------
            # FIX 1: SORT PEAKS BY FREQUENCY BEFORE INTERVAL EXTRACTION
            # -------------------------------------------------------------
            sorted_peaks = sorted(meta.get("peaks", []), key=lambda p: p["f0"])
            f0_list = [p["f0"] for p in sorted_peaks]

            # Extract Method 1 Interval Boundaries on log_clean ground truth
            intervals = extract_intervals_method1_detrended_seed(
                f, log_clean, f0_list, lam=lam, max_high_ratio=max_high_ratio, min_boundary_ratio=min_boundary_ratio
            )

            # Ensure extracted intervals are sorted strictly by f0 ascendingly
            intervals = sorted(intervals, key=lambda info: info["f0"])

            n_p = min(len(intervals), max_peaks)
            for k in range(n_p):
                info = intervals[k]
                f0 = info["f0"]
                f_left = info["peak_edges"][0] if info["peak_edges"][0] is not None else info["segment_bounds"][0]
                f_right = info["peak_edges"][1] if info["peak_edges"][1] is not None else info["segment_bounds"][1]

                # ---------------------------------------------------------
                # FIX 2: NORMALIZE TARGET COORDINATES TO [0, 1] RANGE
                # ---------------------------------------------------------
                targets_arr[i, k] = [f0 / f_max, f_left / f_max, f_right / f_max]
                masks_arr[i, k] = 1.0

            log_noisy_arr[i] = log_noisy
            log_clean_arr[i] = log_clean
            peak_counts_arr[i] = n_p
            meta_list.append(meta)

        data_dict[f"log_noisy_{split_name}"] = log_noisy_arr
        data_dict[f"log_clean_{split_name}"] = log_clean_arr
        data_dict[f"interval_targets_{split_name}"] = targets_arr
        data_dict[f"interval_masks_{split_name}"] = masks_arr
        data_dict[f"peak_counts_{split_name}"] = peak_counts_arr
        data_dict[f"metadata_{split_name}"] = np.array(meta_list, dtype=object)

    # Save compressed archive
    np.savez_compressed(filepath, **data_dict)
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Done! Complete Dataset saved to '{filepath}' ({file_size_mb:.2f} MB)")


# =====================================================================
# 4. FAST DIRECT PYTORCH DATASET LOADER
# =====================================================================
class PSDIntervalDataset(Dataset):
    """
    Direct PyTorch Dataset loader reading precomputed intervals straight from NPZ.
    Returns:
      - log_noisy: (1, n_freqs)
      - log_clean: (1, n_freqs)
      - interval_targets: (max_peaks, 3) -> Normalized [f0, f_left, f_right] in [0, 1]
      - interval_masks: (max_peaks,) -> 1.0 if peak present, else 0.0
      - peak_counts: scalar tensor
    """

    def __init__(self, filepath="psd_dataset_splits_high_amplitude.npz", split="train"):
        assert split in ["train", "val", "test"], "Split must be 'train', 'val', or 'test'"

        data = np.load(filepath, allow_pickle=True)
        self.f = data["f"]
        self.log_noisy = torch.tensor(data[f"log_noisy_{split}"], dtype=torch.float32).unsqueeze(1)
        self.log_clean = torch.tensor(data[f"log_clean_{split}"], dtype=torch.float32).unsqueeze(1)
        self.targets = torch.tensor(data[f"interval_targets_{split}"], dtype=torch.float32)
        self.masks = torch.tensor(data[f"interval_masks_{split}"], dtype=torch.float32)
        self.peak_counts = torch.tensor(data[f"peak_counts_{split}"], dtype=torch.float32)
        self.metadata = data[f"metadata_{split}"]

    def __len__(self):
        return len(self.log_noisy)

    def __getitem__(self, idx):
        return (
            self.log_noisy[idx],
            self.log_clean[idx],
            self.targets[idx],
            self.masks[idx],
            self.peak_counts[idx],
        )


# # =====================================================================
# # EXECUTION DEMO
# # =====================================================================
# if __name__ == "__main__":
#     DATASET_PATH = "psd_dataset_intervals_ha.npz"

#     # 1. Generate dataset with pre-calculated Method 1 intervals (Sorted & Normalized)
#     generate_and_save_split_dataset_with_intervals(
#         filepath=DATASET_PATH,
#         n_train=20000,
#         n_val=2500,
#         n_test=2500,
#         n_freqs=250,
#         f_max=50.0,
#     )

#     # 2. Fast PyTorch Datasets initialization directly from NPZ
#     train_dataset = PSDIntervalDataset(filepath=DATASET_PATH, split="train")
#     val_dataset = PSDIntervalDataset(filepath=DATASET_PATH, split="val")
#     test_dataset = PSDIntervalDataset(filepath=DATASET_PATH, split="test")

#     print(f"\nPyTorch Multi-Task Datasets Ready:")
#     print(f" - Train samples: {len(train_dataset)}")
#     print(f" - Val samples:   {len(val_dataset)}")
#     print(f" - Test samples:  {len(test_dataset)}")

#     # Sample check
#     noisy_x, clean_y, targets, masks, n_p = train_dataset[0]
#     print(f"\nSample #0 Check:")
#     print(f" - Noisy Input Shape   : {noisy_x.shape}")
#     print(f" - Target Intervals (Normalized [0, 1]): \n{targets.numpy()}")
#     print(f" - Mask Array          : {masks.numpy()}")
#     print(f" - Peak Count          : {n_p.item()}")



# ============================================================================================
# Mask dataset
# ===========================================================================================
def generate_and_save_mask_dataset(
    filepath="psd_dataset_masks.npz",
    n_train=20000,
    n_val=2500,
    n_test=2500,
    n_freqs=250,
    f_max=50.0,
    lam=1e2,
    max_high_ratio=0.01,
    min_boundary_ratio=0.5,
):
    f = np.linspace(0.1, f_max, n_freqs)
    df = f[1] - f[0]  # Frequency bin resolution (~0.2 Hz)
    splits = {"train": n_train, "val": n_val, "test": n_test}
    data_dict = {"f": f}

    print("Generating train/val/test dataset with 2D Mask targets...")

    for split_name, count in splits.items():
        print(f" -> Generating {split_name} set ({count} samples)...")
        log_clean_arr = np.zeros((count, n_freqs), dtype=np.float32)
        log_noisy_arr = np.zeros((count, n_freqs), dtype=np.float32)
        masks_arr = np.zeros((count, 2, n_freqs), dtype=np.float32)  # 2 Channels: [Peak Centers, Intervals]

        for i in range(count):
            log_noisy, log_clean, meta = generate_single_psd_sample_high_amplitude(f, n_freqs, f_max)
            f0_list = [p["f0"] for p in meta.get("peaks", [])]

            # Extract Method 1 Interval Boundaries
            intervals = extract_intervals_method1_detrended_seed(
                f, log_clean, f0_list, lam=lam, max_high_ratio=max_high_ratio, min_boundary_ratio=min_boundary_ratio
            )

            # Build 2D Target Masks
            peak_mask = np.zeros(n_freqs, dtype=np.float32)
            interval_mask = np.zeros(n_freqs, dtype=np.float32)

            for info in intervals:
                f0 = info["f0"]
                f_left = info["peak_edges"][0] if info["peak_edges"][0] is not None else info["segment_bounds"][0]
                f_right = info["peak_edges"][1] if info["peak_edges"][1] is not None else info["segment_bounds"][1]

                # Channel 0: Peak Center Mask (Find closest bin index)
                idx_f0 = np.argmin(np.abs(f - f0))
                peak_mask[idx_f0] = 1.0

                # Channel 1: Interval Mask (Set 1.0 for all bins inside [f_left, f_right])
                idx_in_interval = np.where((f >= f_left) & (f <= f_right))[0]
                interval_mask[idx_in_interval] = 1.0

            log_noisy_arr[i] = log_noisy
            log_clean_arr[i] = log_clean
            masks_arr[i, 0] = peak_mask
            masks_arr[i, 1] = interval_mask

        data_dict[f"log_noisy_{split_name}"] = log_noisy_arr
        data_dict[f"log_clean_{split_name}"] = log_clean_arr
        data_dict[f"masks_{split_name}"] = masks_arr

    np.savez_compressed(filepath, **data_dict)
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Done! Mask Dataset saved to '{filepath}' ({file_size_mb:.2f} MB)")


class PSDMaskDataset(Dataset):
    """PyTorch Dataset returning (log_noisy, log_clean, target_masks)."""
    def __init__(self, filepath="psd_dataset_masks.npz", split="train"):
        assert split in ["train", "val", "test"]
        data = np.load(filepath, allow_pickle=True)
        self.f = data["f"]
        self.log_noisy = torch.tensor(data[f"log_noisy_{split}"], dtype=torch.float32).unsqueeze(1)
        self.log_clean = torch.tensor(data[f"log_clean_{split}"], dtype=torch.float32).unsqueeze(1)
        self.target_masks = torch.tensor(data[f"masks_{split}"], dtype=torch.float32)  # Shape: (N, 2, n_freqs)

    def __len__(self):
        return len(self.log_noisy)

    def __getitem__(self, idx):
        return self.log_noisy[idx], self.log_clean[idx], self.target_masks[idx]


# =====================================================================
# EXECUTION DEMO
# =====================================================================
if __name__ == "__main__":
    DATASET_PATH = "psd_dataset_mask.npz"

    # 1. Generate dataset with pre-calculated Method 1 intervals (Sorted & Normalized)
    generate_and_save_mask_dataset(
        filepath=DATASET_PATH,
        n_train=20000,
        n_val=2500,
        n_test=2500,
        n_freqs=250,
        f_max=50.0,
    )

    # 2. Fast PyTorch Datasets initialization directly from NPZ
    train_dataset = PSDMaskDataset(filepath=DATASET_PATH, split="train")
    val_dataset = PSDMaskDataset(filepath=DATASET_PATH, split="val")
    test_dataset = PSDMaskDataset(filepath=DATASET_PATH, split="test")

    print(f"\nPyTorch Multi-Task Datasets Ready:")
    print(f" - Train samples: {len(train_dataset)}")
    print(f" - Val samples:   {len(val_dataset)}")
    print(f" - Test samples:  {len(test_dataset)}")
