import os
import numpy as np
from scipy.interpolate import interp1d
import torch
from torch.utils.data import Dataset
from Functions.extract_psd_bumps import extract_intervals_method1_detrended_seed


def sample_peak_frequencies(n_peaks: int, f_min: float, f_max: float, min_gap: float = 2.0, max_tries: int = 1000):
    """
    Samples peak center frequencies ensuring at least `min_gap` Hz separation.
    """
    if n_peaks <= 0:
        return []
    
    # Margin away from edges to ensure full peak shape fits inside grid
    low = f_min + 1.0
    high = f_max - 1.0

    for _ in range(max_tries):
        candidates = np.random.uniform(low, high, size=n_peaks)
        candidates.sort()
        if n_peaks == 1 or np.all(np.diff(candidates) >= min_gap):
            return candidates.tolist()

    # Fallback deterministic spacing if rejection sampling times out
    return np.linspace(low, high, n_peaks).tolist()


def generate_single_psd_sample_high_amplitude(f: np.ndarray, f_min: float, f_max: float, min_peak_gap: float = 2.0):
    """
    Synthesizes a realistic clean continuous PSD consisting of a 1/f^alpha 
    broadband background and a variable number of Lorentzian/Gaussian peaks.
    """
    # 1. Colored background: S_bg(f) = A_bg / (f^alpha + f_c^alpha) + floor
    alpha = np.random.uniform(0.8, 2.2)
    a_bg = np.random.uniform(1.0, 10.0)
    f_c = np.random.uniform(0.5, 3.0)
    floor_level = np.random.uniform(1e-4, 1e-2)
    
    bg = a_bg / (np.maximum(f, 0.05)**alpha + f_c**alpha) + floor_level

    # 2. Resonant peaks with >= 2.0 Hz gap
    n_peaks = np.random.choice([0, 1, 2, 3, 4], p=[0.05, 0.35, 0.35, 0.15, 0.10])
    f0_list = sample_peak_frequencies(n_peaks, f_min, f_max, min_gap=min_peak_gap)

    peak_curve = np.zeros_like(f)
    peaks_meta = []

    for f0 in f0_list:
        amplitude = np.random.uniform(5.0, 50.0)
        gamma = np.random.uniform(0.3, 1.2)  # Full-width at half-maximum proxy
        
        # Lorentzian resonance
        lorentz = amplitude * (0.5 * gamma)**2 / ((f - f0)**2 + (0.5 * gamma)**2)
        peak_curve += lorentz
        peaks_meta.append({"f0": f0, "amplitude": amplitude, "gamma": gamma})

    linear_clean = bg + peak_curve
    log_clean = np.log(np.maximum(linear_clean, 1e-12)).astype(np.float32)

    meta = {"peaks": peaks_meta, "alpha": alpha, "a_bg": a_bg}
    return log_clean, meta


def generate_and_save_mask_dataset(
    filepath="psd_dataset_masks.npz",
    n_train=20000,
    n_val=2500,
    n_test=2500,
    n_freqs=250,
    f_min=0.1,
    f_max=50.0,
    min_peak_gap=2.0,
    n_obs_range=(80, 350),
    k_welch_range=(4, 32),
    lam=1e2,
    max_high_ratio=0.01,
    min_boundary_ratio=0.5,
):
    f_canonical = np.linspace(f_min, f_max, n_freqs, dtype=np.float32)
    splits = {"train": n_train, "val": n_val, "test": n_test}
    data_dict = {"f": f_canonical}

    print(f"Generating dataset. Peak gap >= {min_peak_gap} Hz. Canonical resolution: {n_freqs} bins.")

    for split_name, count in splits.items():
        print(f" -> Generating {split_name} set ({count} samples)...")
        log_clean_arr = np.zeros((count, n_freqs), dtype=np.float32)
        log_noisy_arr = np.zeros((count, n_freqs), dtype=np.float32)
        masks_arr = np.zeros((count, 2, n_freqs), dtype=np.float32)

        for i in range(count):
            # 1. Synthesize unique clean continuous PSD curve
            log_clean, meta = generate_single_psd_sample_high_amplitude(
                f_canonical, f_min, f_max, min_peak_gap=min_peak_gap
            )
            f0_list = [p["f0"] for p in meta.get("peaks", [])]

            # 2. Extract interval boundaries and construct targets on canonical grid
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

            # Sample clean shape onto the observation grid
            clean_interp = interp1d(f_canonical, log_clean, kind="linear", fill_value="extrapolate")
            log_clean_obs = clean_interp(f_obs)

            # 4. Inject Welch averaging noise at N_obs
            k_seg = int(np.random.randint(k_welch_range[0], k_welch_range[1] + 1))
            noise_mult = np.random.gamma(shape=k_seg, scale=1.0 / k_seg, size=n_obs).astype(np.float32)
            linear_noisy_obs = np.exp(log_clean_obs) * noise_mult
            log_noisy_obs = np.log(np.maximum(linear_noisy_obs, 1e-12))

            # 5. Interpolate back to canonical network input size (250 bins)
            recon_interp = interp1d(f_obs, log_noisy_obs, kind="linear", fill_value="extrapolate")
            log_noisy = recon_interp(f_canonical).astype(np.float32)

            # Store sample
            log_clean_arr[i] = log_clean
            log_noisy_arr[i] = log_noisy
            masks_arr[i, 0] = peak_mask
            masks_arr[i, 1] = interval_mask

        data_dict[f"log_noisy_{split_name}"] = log_noisy_arr
        data_dict[f"log_clean_{split_name}"] = log_clean_arr
        data_dict[f"masks_{split_name}"] = masks_arr

    np.savez_compressed(filepath, **data_dict)
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Dataset successfully saved to '{filepath}' ({file_size_mb:.2f} MB)")


class PSDMaskDataset(Dataset):
    """Loads precomputed (log_noisy, log_clean, masks) directly from disk."""
    def __init__(self, filepath="psd_dataset_masks.npz", split="train"):
        assert split in ["train", "val", "test"]
        data = np.load(filepath, allow_pickle=True)
        self.f = data["f"]
        self.log_noisy = torch.tensor(data[f"log_noisy_{split}"], dtype=torch.float32).unsqueeze(1)
        self.log_clean = torch.tensor(data[f"log_clean_{split}"], dtype=torch.float32).unsqueeze(1)
        self.target_masks = torch.tensor(data[f"masks_{split}"], dtype=torch.float32)

    def __len__(self):
        return len(self.log_noisy)

    def __getitem__(self, idx):
        return self.log_noisy[idx], self.log_clean[idx], self.target_masks[idx]


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
