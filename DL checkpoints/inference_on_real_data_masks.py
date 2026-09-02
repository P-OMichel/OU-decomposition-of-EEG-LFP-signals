import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import torch

# Import mask-based multi-task UNet model
from DL.models.unet_mask import MultiTaskSpectralUNet1D, MultiTaskSpectralUNet1DFusion

# Import your visualization function (adjust import path if needed)
from DL.data.plot_mask_dataset import plot_psd_mask_sample

# Import your OU generation functions
from Functions.generate_OU import (
    get_analytical_psd,
    get_mixed_OU_signals_exact,
)


# =====================================================================
# 1. LOAD TRAINED MASK MODEL
# =====================================================================
def load_mask_model(checkpoint_path="checkpoints_masks/best_mask_unet.pth", hidden_dim=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found!")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Instantiate mask UNet model
    model = MultiTaskSpectralUNet1DFusion(in_channels=1, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Successfully loaded Multi-Task Mask UNet from '{checkpoint_path}'")
    return model, device


# =====================================================================
# 2. INFERENCE SCRIPT FOR MASK PREDICTIONS
# =====================================================================
if __name__ == "__main__":
    CHECKPOINT_PATH = "checkpoints_masks/best_mask_unet.pth"
    
    # --- Toggle Settings ---
    USE_SIMULATED = False   # Set to False for real recorded EEG file
    TARGET_N_FREQS = 250   # Matches training resolution (n_freqs = 250)
    TARGET_F_MAX = 50.0    # Matches network maximum frequency boundary (50 Hz)

    if USE_SIMULATED:
        print("\n--- Running Mask Model Inference on Simulated Mixed OU Data ---")
        T = 100  # Signal duration (s)
        dt = 0.001
        fs = 1 / dt

        lbda_list = [1, 2, 1]
        omega_list = [2 * np.pi * 0.3, 2 * np.pi * 10, 2 * np.pi * 30]
        sigma_list = [3, 2, 50]
        factor_list = [1, 1, 0.005]

        # Generate simulated signal
        t, y = get_mixed_OU_signals_exact(T, dt, lbda_list, omega_list, sigma_list, factor_list)

    else:
        print("\n--- Running Mask Model Inference on Real EEG Recording File ---")
        file = r"c:\Users\holcman\Documents\GitHub\EEG-labellisation-app---Spectrogram\anesthesia_database\rec_20240321_085300.npy"
        fs = 128
        y = np.load(file)
        y = y[2100 * fs : 2250 * fs]  # Extract 150 second slice
        t = np.arange(len(y)) / fs

    # --- Calculate Empirical PSD via Welch's Method ---
    nperseg = int(16 * fs)  # 16-second Welch windows
    f_emp, psd_emp = signal.welch(y, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

    # Restrict empirical range to match target network ceiling (0.1 to 50 Hz)
    freq_mask = (f_emp >= 0.1) & (f_emp <= TARGET_F_MAX)
    f_emp = f_emp[freq_mask]
    psd_emp = psd_emp[freq_mask]

    # --- RESAMPLE INPUT ONTO NETWORK FREQUENCY GRID ---
    f_grid = np.linspace(0.1, TARGET_F_MAX, TARGET_N_FREQS)
    log_psd_emp = np.log(psd_emp + 1e-12)
    resampled_log_psd = np.interp(f_grid, f_emp, log_psd_emp)

    # Format input tensor: Shape (1, 1, 250)
    input_tensor = (
        torch.tensor(resampled_log_psd, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    # --- RUN MASK MODEL INFERENCE ---
    model, device = load_mask_model(CHECKPOINT_PATH, hidden_dim=64)
    input_tensor = input_tensor.to(device)

    # 1. Forward Pass
    with torch.no_grad():
        pred_log_clean, pred_masks = model(input_tensor)

    # 2. Convert to NumPy
    pred_clean_arr = pred_log_clean.squeeze().cpu().numpy()
    center_mask_pred = pred_masks[0, 0].cpu().numpy()   # Channel 0: Peak Centers
    interval_mask_pred = pred_masks[0, 1].cpu().numpy() # Channel 1: Interval Range

    # Log summary peak extraction info
    detected_peak_bins = np.where(center_mask_pred >= 0.4)[0]
    detected_freqs = f_grid[detected_peak_bins]
    print(f"\nModel Prediction Output:")
    print(f" - Detected Peak Count (Ch 0 >= 0.4): {len(detected_freqs)}")
    if len(detected_freqs) > 0:
        print(f" - Peak Center Frequencies: {np.round(detected_freqs, 2)} Hz")

    # 3. Call Visualizer Directly
    plot_psd_mask_sample(
        f=f_grid,
        log_noisy=resampled_log_psd,  # Input empirical Welch PSD
        log_clean=pred_clean_arr,     # Denoised model reconstruction
        center_mask=center_mask_pred,
        interval_mask=interval_mask_pred,
        sample_idx=0
    )