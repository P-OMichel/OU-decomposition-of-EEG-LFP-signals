import os
import numpy as np
import torch
from torch.utils.data import Dataset


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


# =====================================================================
# EXECUTION DEMO
# =====================================================================
if __name__ == "__main__":
    print('Generate data')
    DATASET_PATH = "psd_dataset_splits_high_amplitude.npz"

    # 1. Generate 20,000 train, 2,500 val, 2,500 test samples
    generate_and_save_split_dataset(
        filepath=DATASET_PATH,
        n_train=20000,
        n_val=2500,
        n_test=2500,
        n_freqs=250,
        f_max=50,
    )

    # 2. Example PyTorch loading
    train_dataset = PSDSplitDataset(filepath=DATASET_PATH, split="train")
    val_dataset = PSDSplitDataset(filepath=DATASET_PATH, split="val")
    test_dataset = PSDSplitDataset(filepath=DATASET_PATH, split="test")

    print(f"\nPyTorch Datasets Ready:")
    print(f" - Train samples: {len(train_dataset)}")
    print(f" - Val samples:   {len(val_dataset)}")
    print(f" - Test samples:  {len(test_dataset)}")