import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import torch
import torch.nn.functional as F

from DL.models.model import MultiTaskUNet1D

# Import your OU generation functions
from Functions.generate_OU import (
    get_analytical_psd,
    get_mixed_OU_signals_exact,
)


# =====================================================================
# 1. PEAK & BANDWIDTH DECODER (1D NMS)
# =====================================================================

def decode_keypoint_predictions(pred_logits, pred_widths, f_grid, center_thresh=0.35, kernel_size=5):
    """
    Decodes keypoint heatmap logits and predicted width vectors into physical frequencies.
    
    Returns:
        peaks_info: List of dicts containing f0, f_left, f_right, and heatmap_score
    """
    # 1. Apply Sigmoid to heatmap logits
    heatmap_prob = torch.sigmoid(pred_logits)  # Shape: [1, 1, n_freqs]
    
    # 2. Local Maxima Pooling (1D Non-Maximum Suppression)
    pad = (kernel_size - 1) // 2
    hmax = F.max_pool1d(heatmap_prob, kernel_size=kernel_size, stride=1, padding=pad)
    keep = (heatmap_prob == hmax) & (heatmap_prob >= center_thresh)
    
    peak_indices = torch.nonzero(keep.squeeze()).squeeze(-1)
    if peak_indices.ndim == 0 and peak_indices.numel() > 0:
        peak_indices = peak_indices.unsqueeze(0)
        
    peak_indices = peak_indices.cpu().numpy()
    
    # Frequency bin step size (df in Hz)
    df = f_grid[1] - f_grid[0] if len(f_grid) > 1 else 1.0
    
    peaks_info = []
    pred_widths_np = pred_widths.squeeze().cpu().numpy()  # [2, n_freqs]
    probs_np = heatmap_prob.squeeze().cpu().numpy()        # [n_freqs]
    
    for idx in peak_indices:
        f0 = f_grid[idx]
        score = probs_np[idx]
        
        # Read predicted left/right bin widths from Head 3
        left_bins = pred_widths_np[0, idx]
        right_bins = pred_widths_np[1, idx]
        
        # Convert bin offsets into Hz bounds
        f_left = max(f_grid[0], f0 - (left_bins * df))
        f_right = min(f_grid[-1], f0 + (right_bins * df))
        
        peaks_info.append({
            "idx": idx,
            "f0": f0,
            "f_left": f_left,
            "f_right": f_right,
            "score": score
        })
        
    return peaks_info, probs_np


# =====================================================================
# 2. KEYPOINT RESULT VISUALIZER
# =====================================================================

def plot_keypoint_psd_inference(f_grid, log_noisy, pred_clean, peaks_info, heatmap_prob):
    """
    Plots the empirical noisy log PSD, the network's clean curve reconstruction, 
    the continuous 1D peak heatmap, and the detected peak center/interval regions.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})

    # --- Top Subplot: PSD Curves and Peak Bounds ---
    ax1.plot(f_grid, log_noisy, label="Noisy Empirical Welch PSD", color="gray", alpha=0.5, linestyle="--")
    ax1.plot(f_grid, pred_clean, label="Keypoint U-Net Denoised PSD", color="crimson", linewidth=2.0)

    # Draw detected peak centers and interval bounds
    for i, p in enumerate(peaks_info):
        # Vertical line for Peak Center (f0)
        ax1.axvline(x=p["f0"], color="blue", linestyle=":", linewidth=1.5, 
                    label="Detected Peak Center" if i == 0 else "")
        
        # Shaded frequency interval [f_left, f_right]
        ax1.axvspan(p["f_left"], p["f_right"], color="royalblue", alpha=0.2, 
                    label="Predicted Frequency Interval" if i == 0 else "")
        
        # Annotation text
        peak_y = pred_clean[p["idx"]]
        ax1.annotate(
            f"Peak {i+1}: {p['f0']:.2f} Hz\n[{p['f_left']:.1f} - {p['f_right']:.1f} Hz]",
            xy=(p["f0"], peak_y),
            xytext=(p["f0"], peak_y + 0.6),
            ha='center',
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5)
        )

    ax1.set_ylabel("Log Power Spectral Density")
    ax1.set_title("Keypoint Multi-Task U-Net: PSD Denoising & Peak Interval Detection")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # --- Bottom Subplot: Predicted Center Heatmap ---
    ax2.plot(f_grid, heatmap_prob, color="darkorange", linewidth=1.8, label="Predicted Center Heatmap Head")
    ax2.axhline(y=0.35, color="black", linestyle="--", alpha=0.6, label="Threshold (0.35)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Probability")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


# =====================================================================
# 3. MODEL LOAD & INFERENCE EXECUTION SCRIPT
# =====================================================================

def load_keypoint_model(checkpoint_path="checkpoints_keypoints/best_keypoint_unet.pth", base_filters=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found!")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = MultiTaskUNet1D(in_channels=1, base_filters=base_filters).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded Keypoint UNet from '{checkpoint_path}'")
    return model, device


if __name__ == "__main__":
    CHECKPOINT_PATH = "checkpoints_keypoints/best_keypoint_unet.pth"
    
    # --- Settings ---
    USE_SIMULATED = False   # Toggle False for real EEG recording
    TARGET_N_FREQS = 250   # Matches training resolution
    TARGET_F_MAX = 50.0    # Frequency ceiling (50 Hz)

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
        print("\n--- Running Keypoint Inference on Real EEG Recording File ---")
        file = r"c:\Users\holcman\Documents\GitHub\EEG-labellisation-app---Spectrogram\anesthesia_database\rec_20240321_085300.npy"
        fs = 128
        y = np.load(file)
        y = y[2100 * fs : 2250 * fs]  # Extract 150 second slice
        t = np.arange(len(y)) / fs
    
    # --- Compute Empirical PSD via Welch's Method ---
    nperseg = int(8 * fs)
    f_emp, psd_emp = signal.welch(y, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

    # Restrict to [0.1, 50 Hz]
    freq_mask = (f_emp >= 0.1) & (f_emp <= TARGET_F_MAX)
    f_emp = f_emp[freq_mask]
    psd_emp = psd_emp[freq_mask]

    # Resample onto model grid (n_freqs = 250)
    f_grid = np.linspace(0.1, TARGET_F_MAX, TARGET_N_FREQS)
    log_psd_emp = np.log(psd_emp + 1e-12)
    resampled_log_psd = np.interp(f_grid, f_emp, log_psd_emp)

    # Format input tensor [1, 1, n_freqs]
    input_tensor = torch.tensor(resampled_log_psd, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # --- RUN INFERENCE ---
    model, device = load_keypoint_model(CHECKPOINT_PATH, base_filters=32)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        pred_clean_tensor, pred_logits, pred_widths = model(input_tensor)

    # Extract NumPy predictions
    pred_clean_arr = pred_clean_tensor.squeeze().cpu().numpy()
    
    # Decode Heatmaps and Widths into peak information
    peaks_info, heatmap_prob = decode_keypoint_predictions(
        pred_logits, pred_widths, f_grid, center_thresh=0.35
    )

    # Print Summary Logs
    print(f"\n--- Model Output Summary ---")
    print(f"Detected Peak Count: {len(peaks_info)}")
    for i, p in enumerate(peaks_info):
        print(f" Peak {i+1}: Center = {p['f0']:.2f} Hz | Interval = [{p['f_left']:.2f} Hz, {p['f_right']:.2f} Hz] | Heatmap Score = {p['score']:.3f}")

    # --- VISUALIZE RESULTS ---
    plot_keypoint_psd_inference(
        f_grid=f_grid,
        log_noisy=resampled_log_psd,
        pred_clean=pred_clean_arr,
        peaks_info=peaks_info,
        heatmap_prob=heatmap_prob
    )