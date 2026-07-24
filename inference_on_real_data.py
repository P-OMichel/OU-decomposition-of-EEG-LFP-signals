import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import torch

# Import your model architectures
from DL.models.resnet import SpectralResNet1D
from DL.models.unet import SpectralUNet1D

# Import your OU generation functions
from Functions.generate_OU import (
    get_analytical_psd,
    get_mixed_OU_signals_exact,
)


# =====================================================================
# 1. LOAD TRAINED MODEL
# =====================================================================
def load_trained_model(checkpoint_path="checkpoints/best_spectral_resnet.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file '{checkpoint_path}' not found!"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint.get("model_type", "resnet").lower()

    if model_type == "unet":
        model = SpectralUNet1D().to(device)
    elif model_type == "resnet":
        model = SpectralResNet1D().to(device)
    else:
        raise ValueError(f"Unknown model type '{model_type}' in checkpoint.")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded '{model_type}' model from '{checkpoint_path}'")
    return model, device


# =====================================================================
# 2. INFERENCE FUNCTION ON RAW DATA / PSD
# =====================================================================
def recover_psd_with_model(
    model, device, f_emp, psd_emp, target_n_freqs=225, target_f_max=45
):
    """Interpolates empirical PSD onto network frequency grid (0.1 - 100 Hz),

    runs model inference, and maps prediction back to linear scale.
    """
    f_grid = np.linspace(0.1, target_f_max, target_n_freqs)

    # Convert to log-domain & resample to network's grid
    log_psd_emp = np.log(psd_emp + 1e-12)
    resampled_log_psd = np.interp(f_grid, f_emp, log_psd_emp)

    # Format tensor: Shape (1, 1, 500)
    input_tensor = (
        torch.tensor(resampled_log_psd, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    # Network forward pass
    with torch.no_grad():
        pred_log_psd_grid = model(input_tensor).squeeze().cpu().numpy()

    # Map predictions back to empirical frequency grid and linear scale
    pred_log_psd_emp = np.interp(f_emp, f_grid, pred_log_psd_grid)
    pred_psd_emp = np.exp(pred_log_psd_emp)

    return pred_psd_emp, pred_log_psd_emp, f_grid, pred_log_psd_grid


# =====================================================================
# 3. MAIN SCRIPT
# =====================================================================
# if __name__ == "__main__":

#     # --- Load Network ---
#     CHECKPOINT_PATH = "checkpoints_ha/best_spectral_resnet.pth"
#     model, device = load_trained_model(CHECKPOINT_PATH)

#     # --- Toggle Data Generation Mode ---
#     USE_simulated = True  # Set to False for real recorded EEG file

#     if USE_simulated:
#         print("\n--- Running on Simulated Mixed OU Data ---")
#         T = 1000  # Signal duration (s)
#         dt = 0.001
#         fs = 1 / dt

#         lbda_list = [1, 2, 5]
#         omega_list = [2 * np.pi * 1, 2 * np.pi * 10, 2 * np.pi * 30]
#         sigma_list = [3, 2, 50]
#         factor_list = [1, 1, 0.005]

#         # Generate simulated signal
#         t, y = get_mixed_OU_signals_exact(
#             T, dt, lbda_list, omega_list, sigma_list, factor_list
#         )

#     else:
#         print("\n--- Running on Real EEG Recording File ---")
#         file = r"c:\Users\holcman\Documents\GitHub\EEG-labellisation-app---Spectrogram\anesthesia_database\rec_20240321_085300.npy"
#         fs = 128
#         y = np.load(file)
#         y = y[2100 * fs: 2250 * fs] #[0 * fs : 200 * fs]  # Extract 200 second slice
#         t = np.arange(len(y)) / fs

#     # --- Calculate Empirical PSD via Welch's Method ---
#     nperseg = int(16 * fs)  # 4-second Welch windows
#     f_emp, psd_emp = signal.welch(
#         y, fs=fs, nperseg=nperseg, noverlap=nperseg // 2
#     )

#     # Restrict evaluation to network range (0.1 to 100 Hz)
#     freq_mask = (f_emp >= 0.1) & (f_emp <= 45)
#     f_emp = f_emp[freq_mask]
#     psd_emp = psd_emp[freq_mask]

#     # --- Run Model Inference ---
#     pred_psd_emp, pred_log_psd_emp, f_grid, pred_log_psd_grid = (
#         recover_psd_with_model(model, device, f_emp, psd_emp, target_f_max=45)
#     )

#     # --- Visualization ---
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

#     # Plot 1: Log-Power Domain (What the network operates on)
#     ax1.plot(
#         f_emp,
#         np.log(psd_emp + 1e-12),
#         color="gray",
#         alpha=0.6,
#         linewidth=1.2,
#         label="Welch PSD (Empirical Input)",
#     )
#     ax1.plot(
#         f_emp,
#         pred_log_psd_emp,
#         color="crimson",
#         linewidth=2.0,
#         label="Deep Learning Model Recovery",
#     )

#     if USE_simulated:
#         # Overlay theoretical/analytical spectrum if using simulated data
#         f_analytical, psd_analytical = get_analytical_psd(100,
#             50, lbda_list, omega_list, sigma_list, factor_list
#         )
#         ax1.plot(
#             f_analytical,
#             np.log(psd_analytical + 1e-12),
#             color="black",
#             linestyle="--",
#             linewidth=1.5,
#             label="True Analytical PSD",
#         )

#     ax1.set_ylabel("Log Power (a.u.)")
#     title_str = (
#         "Simulated Mixed OU Signals"
#         if USE_simulated
#         else "Real Anesthesia EEG Data"
#     )
#     ax1.set_title(f"Model Recovery on {title_str} (Log Scale)")
#     ax1.grid(True, linestyle=":", alpha=0.6)
#     ax1.legend(loc="upper right")

#     # Plot 2: Linear-Power Domain
#     ax2.plot(
#         f_emp, psd_emp, color="gray", alpha=0.6, linewidth=1.2, label="Welch PSD"
#     )
#     ax2.plot(
#         f_emp,
#         pred_psd_emp,
#         color="crimson",
#         linewidth=2.0,
#         label="DL Model Recovery",
#     )

#     if USE_simulated:
#         ax2.plot(
#             f_analytical,
#             psd_analytical,
#             color="black",
#             linestyle="--",
#             linewidth=1.5,
#             label="True Analytical PSD",
#         )

#     ax2.set_xlabel("Frequency (Hz)")
#     ax2.set_ylabel("Linear Power")
#     ax2.set_title(f"Model Recovery on {title_str} (Linear Scale)")
#     ax2.grid(True, linestyle=":", alpha=0.6)
#     ax2.legend(loc="upper right")

#     plt.tight_layout()
#     plt.show()



if __name__ == "__main__":

    # --- Load Network ---
    CHECKPOINT_PATH = "checkpoints_ha/best_spectral_resnet.pth"
    model, device = load_trained_model(CHECKPOINT_PATH)

    # --- Toggle Settings ---
    USE_simulated = True  # Set to False for real recorded EEG file
    INTERP = False  # True: resample onto standard network grid with buffering | False: feed raw array

    if USE_simulated:
        print("\n--- Running on Simulated Mixed OU Data ---")
        T = 1000  # Signal duration (s)
        dt = 0.001
        fs = 1 / dt

        lbda_list = [1, 2, 5]
        omega_list = [2 * np.pi * 0.3, 2 * np.pi * 10, 2 * np.pi * 30]
        sigma_list = [3, 2, 50]
        factor_list = [1, 1, 0.005]

        # Generate simulated signal
        t, y = get_mixed_OU_signals_exact(
            T, dt, lbda_list, omega_list, sigma_list, factor_list
        )

    else:
        print("\n--- Running on Real EEG Recording File ---")
        file = r"c:\Users\holcman\Documents\GitHub\EEG-labellisation-app---Spectrogram\anesthesia_database\rec_20240321_085300.npy"
        fs = 128
        y = np.load(file)
        y = y[2100 * fs : 2250 * fs]  # Extract 200 second slice
        t = np.arange(len(y)) / fs

    # --- Calculate Empirical PSD via Welch's Method ---
    nperseg = int(16 * fs)  # 16-second Welch windows
    f_emp, psd_emp = signal.welch(
        y, fs=fs, nperseg=nperseg, noverlap=nperseg // 2
    )

    # Restrict evaluation to target frequency range (0.1 to 45 Hz)
    freq_mask = (f_emp >= 0) & (f_emp <= 45.0)
    f_emp = f_emp[freq_mask]
    psd_emp = psd_emp[freq_mask]

    # --- PREPARE INPUT ARRAY (Option B: Frequency Buffering + INTERP Toggle) ---
    if INTERP:
        buffer_hz = 2.0  # Extend 2 Hz below and above target range to kill edge artifacts
        f_min_buffered = max(0.0, f_emp[0] - buffer_hz)
        f_max_buffered = 45.0 + buffer_hz

        # Resample onto 500-point grid with padded frequency bounds
        f_in = np.linspace(f_min_buffered, f_max_buffered, 500)
        log_psd_in = np.interp(f_in, f_emp, np.log(psd_emp + 1e-12))
    else:
        # Pass raw empirical frequency array directly
        f_in = f_emp.copy()
        log_psd_in = np.log(psd_emp + 1e-12)

    # --- Run Model Inference ---
    # Convert input array to PyTorch Tensor (Batch=1, Channel=1, Length=N)
    input_tensor = (
        torch.tensor(log_psd_in, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        pred_log_psd_out = model(input_tensor).squeeze().cpu().numpy()

    # --- POST-PROCESS OUTPUT (Crop Buffer Back to Empirical Grid) ---
    if INTERP:
        # Map predictions from buffered grid back onto original f_emp (discards boundary edge artifacts)
        pred_log_psd_emp = np.interp(f_emp, f_in, pred_log_psd_out)
    else:
        pred_log_psd_emp = pred_log_psd_out.copy()

    pred_psd_emp = np.exp(pred_log_psd_emp)

    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Log-Power Domain (What the network operates on)
    ax1.plot(
        f_emp,
        np.log(psd_emp + 1e-12),
        color="gray",
        alpha=0.6,
        linewidth=1.2,
        label="Welch PSD (Empirical Input)",
    )
    ax1.plot(
        f_emp,
        pred_log_psd_emp,
        color="crimson",
        linewidth=2.0,
        label="Deep Learning Model Recovery",
    )

    if USE_simulated:
        # Overlay theoretical/analytical spectrum if using simulated data
        f_analytical, psd_analytical = get_analytical_psd(
            100, 50, lbda_list, omega_list, sigma_list, factor_list
        )
        ax1.plot(
            f_analytical,
            np.log(psd_analytical + 1e-12),
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="True Analytical PSD",
        )

    ax1.set_ylabel("Log Power (a.u.)")
    title_str = (
        "Simulated Mixed OU Signals"
        if USE_simulated
        else "Real Anesthesia EEG Data"
    )
    interp_str = "Resampled/Buffered Grid" if INTERP else "Raw Direct Array"
    ax1.set_title(
        f"Model Recovery on {title_str} ({interp_str} | Log Scale)"
    )
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # Plot 2: Linear-Power Domain
    ax2.plot(
        f_emp, psd_emp, color="gray", alpha=0.6, linewidth=1.2, label="Welch PSD"
    )
    ax2.plot(
        f_emp,
        pred_psd_emp,
        color="crimson",
        linewidth=2.0,
        label="DL Model Recovery",
    )

    if USE_simulated:
        ax2.plot(
            f_analytical,
            psd_analytical,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="True Analytical PSD",
        )

    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Linear Power")
    ax2.set_title(
        f"Model Recovery on {title_str} ({interp_str} | Linear Scale)"
    )
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()