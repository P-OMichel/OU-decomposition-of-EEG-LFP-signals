import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import torch

# Import new multi-task model architecture
from DL.models.resnet import MultiTaskSpectralResNet1D

# Import your OU generation functions
from Functions.generate_OU import (
    get_analytical_psd,
    get_mixed_OU_signals_exact,
)


# =====================================================================
# 1. LOAD TRAINED MULTI-TASK MODEL
# =====================================================================
def load_multitask_model(checkpoint_path="checkpoints_multitask/best_multitask_resnet.pth", max_peaks=4, f_max=50.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found!")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Instantiate multi-task model
    model = MultiTaskSpectralResNet1D(max_peaks=max_peaks, f_max=f_max).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Successfully loaded Multi-Task ResNet from '{checkpoint_path}'")
    return model, device


# =====================================================================
# 2. INFERENCE SCRIPT FOR MULTI-TASK PREDICTIONS
# =====================================================================
if __name__ == "__main__":
    CHECKPOINT_PATH = "checkpoints_multitask/best_multitask_resnet.pth"
    
    # --- Toggle Settings ---
    USE_SIMULATED = True  # Set to False for real recorded EEG file
    TARGET_N_FREQS = 250   # Matches training resolution (n_freqs = 250)
    TARGET_F_MAX = 50.0    # Matches network maximum frequency boundary (50 Hz)

    if USE_SIMULATED:
        print("\n--- Running Inference on Simulated Mixed OU Data ---")
        T = 1000  # Signal duration (s)
        dt = 0.001
        fs = 1 / dt

        lbda_list = [1, 2, 5]
        omega_list = [2 * np.pi * 0.3, 2 * np.pi * 10, 2 * np.pi * 30]
        sigma_list = [3, 2, 50]
        factor_list = [1, 1, 0.005]

        # Generate simulated signal
        t, y = get_mixed_OU_signals_exact(T, dt, lbda_list, omega_list, sigma_list, factor_list)

    else:
        print("\n--- Running Inference on Real EEG Recording File ---")
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

    # --- RUN MULTI-TASK MODEL INFERENCE ---
    model, device = load_multitask_model(CHECKPOINT_PATH, max_peaks=4, f_max=TARGET_F_MAX)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        pred_log_clean, pred_intervals, pred_count = model(input_tensor)
        pred_intervals = pred_intervals * 50

        # Move outputs back to CPU/NumPy
        pred_log_psd_grid = pred_log_clean.squeeze().cpu().numpy()
        pred_intervals_mat = pred_intervals.squeeze().cpu().numpy()  # Shape: (4, 3) -> [f0, f_left, f_right]
        raw_count_pred = pred_count.squeeze().item()
        estimated_n_peaks = int(np.clip(np.round(raw_count_pred), 0, 4))

    # Map reconstructed log PSD back to empirical frequencies
    pred_log_psd_emp = np.interp(f_emp, f_grid, pred_log_psd_grid)
    pred_psd_emp = np.exp(pred_log_psd_emp)

    print(f"\nModel Prediction Output:")
    print(f" - Estimated Peak Count : {estimated_n_peaks} (Raw network output: {raw_count_pred:.2f})")
    for k in range(estimated_n_peaks):
        f0, f_l, f_r = pred_intervals_mat[k]
        print(f" - Peak #{k+1}: f0 = {f0:.2f} Hz | Interval = [{f_l:.2f} Hz, {f_r:.2f} Hz]")

    # =====================================================================
    # 3. VISUALIZATION
    # =====================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    title_str = "Simulated Mixed OU Signals" if USE_SIMULATED else "Real Anesthesia EEG Data"

    # --- TOP AXIS: LOG PSD FIT & INTERVALS ---
    ax1.plot(f_emp, log_psd_emp, color="gray", alpha=0.5, lw=1.2, label="Welch Empirical Log PSD")
    ax1.plot(f_emp, pred_log_psd_emp, color="blue", lw=1.8, label="Model Reconstructed Fit")

    if USE_SIMULATED:
        f_analytical, psd_analytical = get_analytical_psd(100, 50, lbda_list, omega_list, sigma_list, factor_list)
        ax1.plot(f_analytical, np.log(psd_analytical + 1e-12), color="black", linestyle="--", lw=1.5, label="True Analytical PSD")

    # Plot predicted peak locations and frequency intervals
    for k in range(estimated_n_peaks):
        f0, f_l, f_r = pred_intervals_mat[k]
        
        # Mark predicted peak location
        p_label = "Predicted Peak (f0)" if k == 0 else ""
        ax1.axvline(f0, color="red", linestyle="--", alpha=0.8, label=p_label)
        
        # Shade predicted frequency interval
        int_label = "Predicted Interval" if k == 0 else ""
        ax1.axvspan(f_l, f_r, color="orange", alpha=0.25, label=int_label)

    ax1.set_ylabel("Log Power Spectral Density")
    ax1.set_title(f"Multi-Task Inference on {title_str} (Estimated Peaks: {estimated_n_peaks})")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Deduplicate legend entries
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc="upper right")

    # --- BOTTOM AXIS: LINEAR PSD FIT & INTERVALS ---
    ax2.plot(f_emp, psd_emp, color="gray", alpha=0.5, lw=1.2, label="Welch Empirical PSD")
    ax2.plot(f_emp, pred_psd_emp, color="blue", lw=1.8, label="Model Reconstructed Fit")

    if USE_SIMULATED:
        ax2.plot(f_analytical, psd_analytical, color="black", linestyle="--", lw=1.5, label="True Analytical PSD")

    for k in range(estimated_n_peaks):
        f0, f_l, f_r = pred_intervals_mat[k]
        ax2.axvline(f0, color="red", linestyle="--", alpha=0.8)
        ax2.axvspan(f_l, f_r, color="orange", alpha=0.25)

    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Linear Power")
    ax2.set_title(f"Linear PSD Fit with Frequency Interval Overlays")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label2 = dict(zip(labels2, handles2))
    ax2.legend(by_label2.values(), by_label2.keys(), loc="upper right")

    plt.tight_layout()
    plt.show()