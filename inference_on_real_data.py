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



if __name__ == "__main__":

    list_checkpoints = [
        'checkpoints/best_spectral_resnet.pth',
        'checkpoints_ha_Laplacian/best_spectral_resnet.pth',
        'checkpoints_ha/best_spectral_resnet.pth',
        'checkpoints_ha_spectral/best_spectral_resnet.pth'
    ]

    # --- Toggle Settings ---
    USE_simulated = False  # Set to False for real recorded EEG file
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

    # --- PREPARE INPUT ARRAY ---
    if INTERP:
        buffer_hz = 2.0  # Extend 2 Hz below and above target range to kill edge artifacts
        f_min_buffered = max(0.0, f_emp[0] - buffer_hz)
        f_max_buffered = 45.0 + buffer_hz

        # Resample onto 500-point grid with padded frequency bounds
        f_in = np.linspace(f_min_buffered, f_max_buffered, 225)
        log_psd_in = np.interp(f_in, f_emp, np.log(psd_emp + 1e-12))
    else:
        # Pass raw empirical frequency array directly
        f_in = f_emp.copy()
        log_psd_in = np.log(psd_emp + 1e-12)

    # --- INFERENCE ACROSS ALL CHECKPOINTS ---
    # Store results for each checkpoint: {label: (pred_log_psd, pred_psd)}
    predictions = {}

    for path in list_checkpoints:
        print(f"Running inference for checkpoint: {path}")
        model, device = load_trained_model(path)

        input_tensor = (
            torch.tensor(log_psd_in, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            pred_log_psd_out = model(input_tensor).squeeze().cpu().numpy()

        # Post-process per checkpoint
        if INTERP:
            pred_log_psd_emp = np.interp(f_emp, f_in, pred_log_psd_out)
        else:
            pred_log_psd_emp = pred_log_psd_out.copy()

        pred_psd_emp = np.exp(pred_log_psd_emp)

        # Create a clean label from folder path or filename
        model_name = path.split('/')[0] if '/' in path else path
        predictions[model_name] = (pred_log_psd_emp, pred_psd_emp)

    # --- VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    # Ground truth / empirical data base lines
    ax1.plot(
        f_emp,
        np.log(psd_emp + 1e-12),
        color="gray",
        alpha=0.5,
        linewidth=1.2,
        label="Welch PSD (Empirical Input)",
    )
    ax2.plot(
        f_emp,
        psd_emp,
        color="gray",
        alpha=0.5,
        linewidth=1.2,
        label="Welch PSD",
    )

    if USE_simulated:
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
        ax2.plot(
            f_analytical,
            psd_analytical,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="True Analytical PSD",
        )

    # Plot each model prediction
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))

    for idx, (label, (pred_log_psd, pred_psd)) in enumerate(predictions.items()):
        color = colors[idx]
        ax1.semilogx(f_emp, pred_log_psd, color=color, linewidth=1.8, label=f"Model: {label}")
        ax2.plot(f_emp, pred_psd, color=color, linewidth=1.8, label=f"Model: {label}")

    title_str = "Simulated Mixed OU Signals" if USE_simulated else "Real Anesthesia EEG Data"
    interp_str = "Resampled/Buffered Grid" if INTERP else "Raw Direct Array"

    # Setup Axes 1 (Log Power)
    ax1.set_ylabel("Log Power (a.u.)")
    ax1.set_title(f"Checkpoint Comparison on {title_str} ({interp_str} | Log Scale)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize="small")

    # Setup Axes 2 (Linear Power)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Linear Power")
    ax2.set_title(f"Checkpoint Comparison on {title_str} ({interp_str} | Linear Scale)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize="small")

    plt.tight_layout()
    plt.show()