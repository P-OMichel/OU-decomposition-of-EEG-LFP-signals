'''
Implementation of the Spectral loss requires a cutoff frequency at which FFT of the PSD, both noisy output of model and clean reference,
is cropped. Values lower than threshold are considered to be smooth and are not compared between two psds in the loss. Only points for frequencies above 
are compared and contirbute to the loss with the aim of reducing those high frequency jittering.
'''


import numpy as np
import matplotlib.pyplot as plt

def find_fft_cutoff(clean_batch, noisy_batch, plot=True):
    """
    Args:
        clean_batch: np.ndarray of shape (20000, 250) - Ground truth PSDs
        noisy_batch: np.ndarray of shape (20000, 250) - Input noisy PSDs
    Returns:
        cutoff_ratio (float): Recommended cutoff for LossMSEFourier (0.0 to 1.0)
    """
    # 1. Real FFT along axis=-1 (the 250-length PSD dimension)
    # For length 250, rfft produces 126 complex frequency bins (N//2 + 1)
    clean_fft = np.fft.rfft(clean_batch, axis=-1)
    noisy_fft = np.fft.rfft(noisy_batch, axis=-1)

    # 2. Average magnitude across all 20,000 samples
    clean_mag_avg = np.mean(np.abs(clean_fft), axis=0)
    noisy_mag_avg = np.mean(np.abs(noisy_fft), axis=0)

    num_bins = clean_mag_avg.shape[0]  # 126 bins

    # 3. Convert to dB for stable comparison
    clean_db = 20 * np.log10(clean_mag_avg + 1e-8)
    noisy_db = 20 * np.log10(noisy_mag_avg + 1e-8)

    # 4. Find the first frequency bin where noise exceeds clean signal by > 3 dB
    diff_db = noisy_db - clean_db
    crossover_indices = np.where(diff_db > 1.0)[0]

    if len(crossover_indices) > 0:
        cutoff_bin = crossover_indices[0]
    else:
        # Fallback: pick bin where 95% of cumulative energy is contained
        cumulative_energy = np.cumsum(clean_mag_avg ** 2)
        total_energy = cumulative_energy[-1]
        cutoff_bin = np.where(cumulative_energy >= 0.95 * total_energy)[0][0]

    # Ratio relative to total rfft frequency bins (0.0 to 1.0 scale)
    cutoff_ratio = cutoff_bin / num_bins

    if plot:
        freq_bins = np.arange(num_bins)
        plt.figure(figsize=(8, 4))
        plt.plot(freq_bins, clean_db, label="Clean Avg Spectrum", color="blue")
        plt.plot(freq_bins, noisy_db, label="Noisy Avg Spectrum", color="red", linestyle="--")
        plt.axvline(x=cutoff_bin, color="black", linestyle=":", label=f"Cutoff Bin ({cutoff_bin})")
        plt.xlabel("Frequency Bin (k)")
        plt.ylabel("Magnitude (dB)")
        plt.title(f"FFT Spectrum (PSD Length 250 -> {num_bins} Frequency Bins)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    print(f"Recommended cutoff_ratio for PyTorch loss: {cutoff_ratio:.3f} (Bin {cutoff_bin} of {num_bins})")
    return cutoff_ratio




data = np.load('psd_dataset_splits.npz', allow_pickle=True)
split="train"

f = data["f"]

noisy_key = f"log_noisy_{split}"
clean_key = f"log_clean_{split}"

log_noisy_arr = data[noisy_key]
log_clean_arr = data[clean_key]

find_fft_cutoff(log_clean_arr, log_noisy_arr)