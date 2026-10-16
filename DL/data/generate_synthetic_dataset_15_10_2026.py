"""
=============================================================================
SYNTHETIC PSD DATASET GENERATOR WITH DIVERSE PEAK SHAPES & REALISTIC NOISE
=============================================================================
Key Design Choices & Harmonized Parameters:
1. Amplitude:
   - A ~ Uniform(0.3, 8.0) uniformly applied across all peaks (no special-cased
     delta band amplitudes). Power contribution is scaled as A^2.
2. Peak Width / Gamma:
   - gamma ~ Uniform(0.5, 2.5) Hz across all peak shapes.
   - Yields an effective Full-Width at Half-Maximum (FWHM) in the ~1.0 to 5.0 Hz range:
       * Pseudo-Voigt: FWHM = 2 * gamma -> [1.0, 5.0] Hz
       * Ornstein-Uhlenbeck: Evaluated directly in ordinary frequency Hz scale
         (2 * pi factored out of the resonant term) -> FWHM = 2 * gamma -> [1.0, 5.0] Hz
       * Pearson IV: FWHM ~ 2 * gamma * sqrt(2^(1/m) - 1) -> ~[0.8, 4.5] Hz
       * Student-t:  FWHM ~ 2 * gamma * sqrt(2^(2/alpha) - 1) -> ~[0.8, 5.0] Hz
3. Peak Line Shapes:
   - Randomly drawn from: Pseudo-Voigt, Ornstein-Uhlenbeck, Pearson IV, Student-t.
   - Includes optional right-side power-law decay tail (30% chance for f0 > 2 Hz).
4. Noise Model:
   - Welch Averaged Periodogram Noise: Multiplicative Gamma noise in linear power:
     noise_mult ~ Gamma(shape=K, scale=1/K), where K ~ Uniform(4, 32) Welch segments.
5. Grid & Resolution Interpolation:
   - Variable observation length N_obs ~ Uniform(80, 350) points.
   - Interpolates clean PSD to random observation grid, applies Welch noise,
     and re-interpolates back to canonical network input size (250 bins).
=============================================================================
"""

import os
import numpy as np
from scipy.interpolate import interp1d
import torch
from torch.utils.data import Dataset
from Functions.extract_psd_bumps import extract_intervals_method1_detrended_seed


# =====================================================================
# 1. CORE SAMPLE GENERATOR
# =====================================================================
def generate_single_psd_sample_unified(f: np.ndarray, f_max: float):
    """
    Synthesizes a clean continuous PSD with mixed peak shapes, universal 
    amplitude/gamma ranges, and returns metadata for interval extraction.
    """
    # A. Background Component: 1/f^chi with optional knee
    chi = np.random.uniform(0.8, 3.0)
    b_amp = np.random.uniform(0.5, 3.0)
    has_knee = np.random.choice([True, False], p=[0.4, 0.6])

    f_safe = np.maximum(f, 0.05)
    if has_knee:
        f_knee = np.random.uniform(2.0, 25.0)
        bg = b_amp / (f_knee**chi + f_safe**chi)
    else:
        bg = b_amp / (f_safe**chi)

    psd_clean = bg.copy()

    # B. Peak Injections (0 to 4 peaks, diverse shape models)
    n_peaks = np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.3, 0.3, 0.2, 0.1])
    peak_params = []
    peak_shape_names = ["pseudo_voigt", "ornstein_uhlenbeck", "pearson_iv", "student_t"]

    for _ in range(n_peaks):
        shape_type = np.random.choice(peak_shape_names)
        f0 = np.random.uniform(0.5, f_max - 5.0)
        
        # Universal Amplitude: A in [0.3, 8.0] across all frequencies
        A = np.random.uniform(0.3, 8.0)
        
        # Universal Gamma scale: gamma in [0.5, 2.5] Hz -> FWHM in [1.0, 5.0] Hz
        gamma = np.random.uniform(0.5, 2.5)

        if shape_type == "pseudo_voigt":
            eta = np.random.uniform(0.0, 1.0)
            sigma = gamma / np.sqrt(2.0 * np.log(2.0))
            f_pos, f_neg = f - f0, f + f0
            L_pos = (1.0 / np.pi) * (gamma / (f_pos**2 + gamma**2))
            L_neg = (1.0 / np.pi) * (gamma / (f_neg**2 + gamma**2))
            G_pos = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * (f_pos / sigma)**2)
            G_neg = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * (f_neg / sigma)**2)
            peak_signal = A**2 * (eta * L_pos + (1.0 - eta) * G_pos + eta * L_neg + (1.0 - eta) * G_neg)
            p_dict = {"shape": shape_type, "f0": f0, "A": A, "gamma": gamma, "eta": eta}

        elif shape_type == "ornstein_uhlenbeck":
            # Formulated in ordinary frequency (Hz) so gamma matches Hz scale directly
            f_pos, f_neg = f - f0, f + f0
            peak_signal = (A**2 / np.pi) * (
                (gamma / (gamma**2 + f_pos**2)) + (gamma / (gamma**2 + f_neg**2))
            )
            p_dict = {"shape": shape_type, "f0": f0, "A": A, "gamma": gamma}

        elif shape_type == "pearson_iv":
            m = np.random.uniform(1.1, 4.0)
            nu_p = np.random.uniform(-2.0, 2.0)
            y_pos, y_neg = (f - f0) / gamma, (f + f0) / gamma
            term_pos = (1.0 + y_pos**2)**(-m) * np.exp(-nu_p * np.arctan(y_pos))
            term_neg = (1.0 + y_neg**2)**(-m) * np.exp(-nu_p * np.arctan(y_neg))
            peak_signal = A**2 * (term_pos + term_neg)
            p_dict = {"shape": shape_type, "f0": f0, "A": A, "gamma": gamma, "m": m, "nu_p": nu_p}

        elif shape_type == "student_t":
            alpha = np.random.uniform(1.2, 4.0)
            term_pos = 1.0 + ((f - f0) / gamma)**2
            term_neg = 1.0 + ((f + f0) / gamma)**2
            peak_signal = (A**2 / np.pi) * (
                (1.0 / term_pos**(alpha / 2.0)) + (1.0 / term_neg**(alpha / 2.0))
            )
            p_dict = {"shape": shape_type, "f0": f0, "A": A, "gamma": gamma, "alpha": alpha}

        # Optional Asymmetric Post-Peak Decay Tail
        has_post_decay = np.random.choice([True, False], p=[0.3, 0.7])
        if has_post_decay and f0 > 2.0:
            beta_decay = np.random.uniform(1.0, 3.5)
            f_tail = np.maximum(f, 1e-6)
            post_decay = (
                (A**2 * 0.3)
                * ((f_tail / f0) ** (-beta_decay))
                * np.maximum(0.0, 1.0 - (f0 / f_tail))
            )
            peak_signal += np.where(f > f0, post_decay, 0.0)
            p_dict["beta_decay"] = beta_decay

        psd_clean += peak_signal
        peak_params.append(p_dict)

    log_clean = np.log(np.maximum(psd_clean, 1e-12)).astype(np.float32)
    meta = {
        "n_peaks": n_peaks,
        "b_amp": float(b_amp),
        "chi": float(chi),
        "has_knee": has_knee,
        "peaks": peak_params,
    }
    return log_clean, meta


# =====================================================================
# 2. DATASET GENERATION & NPZ EXPORT (MASK TARGETS)
# =====================================================================
def generate_and_save_mask_dataset(
    filepath="psd_dataset_masks.npz",
    n_train=20000,
    n_val=2500,
    n_test=2500,
    n_freqs=250,
    f_min=0.1,
    f_max=50.0,
    n_obs_range=(80, 350),
    k_welch_range=(4, 32),
    lam=1e2,
    max_high_ratio=0.01,
    min_boundary_ratio=0.5,
):
    """
    Generates train, val, and test splits with:
    - Multi-shape clean continuous PSDs
    - Extracted Method 1 interval & peak center masks
    - Realistic Welch periodogram noise simulation across variable observation grids
    """
    f_canonical = np.linspace(f_min, f_max, n_freqs, dtype=np.float32)
    splits = {"train": n_train, "val": n_val, "test": n_test}
    data_dict = {"f": f_canonical}

    print(f"Generating dataset. Canonical resolution: {n_freqs} bins [{f_min}, {f_max}] Hz.")

    for split_name, count in splits.items():
        print(f" -> Generating {split_name} set ({count} samples)...")
        log_clean_arr = np.zeros((count, n_freqs), dtype=np.float32)
        log_noisy_arr = np.zeros((count, n_freqs), dtype=np.float32)
        masks_arr = np.zeros((count, 2, n_freqs), dtype=np.float32)

        for i in range(count):
            # 1. Synthesize clean continuous PSD curve
            log_clean, meta = generate_single_psd_sample_unified(f_canonical, f_max)
            
            # Sort peaks ascendingly by f0
            sorted_peaks = sorted(meta.get("peaks", []), key=lambda p: p["f0"])
            f0_list = [p["f0"] for p in sorted_peaks]

            # 2. Extract interval boundaries using ground truth log_clean
            intervals = extract_intervals_method1_detrended_seed(
                f_canonical, log_clean, f0_list, lam=lam,
                max_high_ratio=max_high_ratio, min_boundary_ratio=min_boundary_ratio
            )

            # Construct 2-channel binary targets: Channel 0 = Peak Centers, Channel 1 = Interval Spans
            peak_mask = np.zeros(n_freqs, dtype=np.float32)
            interval_mask = np.zeros(n_freqs, dtype=np.float32)

            for info in intervals:
                f0 = info["f0"]
                f_left = info["peak_edges"][0] if info["peak_edges"][0] is not None else info["segment_bounds"][0]
                f_right = info["peak_edges"][1] if info["peak_edges"][1] is not None else info["segment_bounds"][1]

                # Channel 0: Nearest bin index for center frequency
                peak_mask[np.argmin(np.abs(f_canonical - f0))] = 1.0
                
                # Channel 1: Span coverage
                interval_mask[(f_canonical >= f_left) & (f_canonical <= f_right)] = 1.0

            # 3. Simulate variable physical observation length N_obs
            n_obs = int(np.random.randint(n_obs_range[0], n_obs_range[1] + 1))
            f_obs = np.linspace(f_min, f_max, n_obs, dtype=np.float32)

            # Sample clean shape onto the observation grid
            clean_interp = interp1d(f_canonical, log_clean, kind="linear", fill_value="extrapolate")
            log_clean_obs = clean_interp(f_obs)

            # 4. Inject Welch averaging noise at N_obs resolution
            k_seg = int(np.random.randint(k_welch_range[0], k_welch_range[1] + 1))
            noise_mult = np.random.gamma(shape=k_seg, scale=1.0 / k_seg, size=n_obs).astype(np.float32)
            linear_noisy_obs = np.exp(log_clean_obs) * noise_mult
            log_noisy_obs = np.log(np.maximum(linear_noisy_obs, 1e-12))

            # 5. Interpolate back to canonical network input size (250 bins)
            recon_interp = interp1d(f_obs, log_noisy_obs, kind="linear", fill_value="extrapolate")
            log_noisy = recon_interp(f_canonical).astype(np.float32)

            # Store processed sample
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


# =====================================================================
# 3. PYTORCH DATASET LOADER
# =====================================================================
class PSDMaskDataset(Dataset):
    """
    Loads precomputed (log_noisy, log_clean, masks) directly from NPZ.
    Outputs:
      - log_noisy: (1, n_freqs)
      - log_clean: (1, n_freqs)
      - target_masks: (2, n_freqs) -> [Peak Centers, Interval Spans]
    """
    def __init__(self, filepath="psd_dataset_masks.npz", split="train"):
        assert split in ["train", "val", "test"], "Split must be 'train', 'val', or 'test'"
        data = np.load(filepath, allow_pickle=True)
        self.f = data["f"]
        self.log_noisy = torch.tensor(data[f"log_noisy_{split}"], dtype=torch.float32).unsqueeze(1)
        self.log_clean = torch.tensor(data[f"log_clean_{split}"], dtype=torch.float32).unsqueeze(1)
        self.target_masks = torch.tensor(data[f"masks_{split}"], dtype=torch.float32)

    def __len__(self):
        return len(self.log_noisy)

    def __getitem__(self, idx):
        return self.log_noisy[idx], self.log_clean[idx], self.target_masks[idx]


# =====================================================================
# EXECUTION DEMO
# =====================================================================
if __name__ == "__main__":
    DATASET_PATH = "psd_dataset_masks.npz"

    # 1. Generate full dataset with Welch noise, grid interpolation & multiple peak lineshapes
    generate_and_save_mask_dataset(
        filepath=DATASET_PATH,
        n_train=20000,
        n_val=2500,
        n_test=2500,
        n_freqs=250,
        f_min=0.1,
        f_max=50.0,
    )

    # 2. Fast PyTorch Dataset initialization
    train_dataset = PSDMaskDataset(filepath=DATASET_PATH, split="train")
    val_dataset = PSDMaskDataset(filepath=DATASET_PATH, split="val")
    test_dataset = PSDMaskDataset(filepath=DATASET_PATH, split="test")

    print(f"\nPyTorch Multi-Task Datasets Ready:")
    print(f" - Train samples: {len(train_dataset)}")
    print(f" - Val samples:   {len(val_dataset)}")
    print(f" - Test samples:  {len(test_dataset)}")

    # Sanity verification on sample #0
    noisy_x, clean_y, mask_targets = train_dataset[0]
    print(f"\nSample #0 Tensor Shapes:")
    print(f" - Noisy Input (log_noisy)  : {noisy_x.shape}")
    print(f" - Ground Truth (log_clean) : {clean_y.shape}")
    print(f" - Mask Targets             : {mask_targets.shape}")