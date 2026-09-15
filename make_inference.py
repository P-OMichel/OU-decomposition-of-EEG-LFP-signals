'''
File to run inference on saved models
'''
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.signal as signal

# ===================================================
# generate OU signal
# ===================================================
def get_OU_signal_exact(T, dt, lbda, omega, sigma):
    """
    Generates a 2D Ornstein-Uhlenbeck signal using the exact discrete-time step
    (matrix exponential solution) with zero numerical damping.
    """
    N = int(T / dt)
    t = np.linspace(0, T, N)

    x = np.zeros(N)
    y = np.zeros(N)

    # 1. Exact transition matrix M = exp(A * dt)
    decay = np.exp(-lbda * dt)
    cos_w = np.cos(omega * dt)
    sin_w = np.sin(omega * dt)

    M = decay * np.array([
        [ cos_w,  sin_w],
        [-sin_w,  cos_w]
    ])

    # 2. Exact standard deviation for discrete noise step
    sigma_eta = sigma * np.sqrt((1.0 - np.exp(-2.0 * lbda * dt)) / lbda)

    # 3. Simulate exact step
    for i in range(1, N):
        # Generate independent Gaussian noise
        eta_x = np.random.normal(0, sigma_eta)
        eta_y = np.random.normal(0, sigma_eta)

        # Matrix multiplication for deterministic state transition
        prev_state = np.array([x[i-1], y[i-1]])
        next_state = M @ prev_state + np.array([eta_x, eta_y])

        x[i] = next_state[0]
        y[i] = next_state[1]

    return t, x

def get_mixed_OU_signals_exact(T, dt, lbda_list, omega_list, sigma_list, factor_list):
    """
    Generates mixed OU signals using the exact solver.
    """
    N = int(T / dt)
    t = np.linspace(0, T, N)
    mixed_OU = np.zeros(N)

    for i in range(len(lbda_list)):
        _, x = get_OU_signal_exact(T, dt, lbda_list[i], omega_list[i], sigma_list[i])
        mixed_OU += x * factor_list[i]

    return t, mixed_OU


# ====================================================
# NN 
# ====================================================

class ResConvBlock1D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, drop_prob=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_c),
            nn.GELU(),
            nn.Dropout1d(p=drop_prob),  # Drop full feature channels
            nn.Conv1d(out_c, out_c, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_c),
        )
        self.shortcut = (
            nn.Conv1d(in_c, out_c, kernel_size=1)
            if in_c != out_c
            else nn.Identity()
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.block(x) + self.shortcut(x))


class ResMultiTaskUNet1D(nn.Module):
    """
    Multi-Task 1D UNet Architecture:
    - Backbone: Residual 1D UNet with 3 Encoder stages and 2 Decoder stages.
    - Head 1 (Denoise): Reconstructs clean PSD curve (1 channel output).
    - Head 2 (Heatmap): Predicts peak center probability logits (1 channel output).
    - Head 3 (Width): Predicts left & right peak widths (2 channel output, Softplus activation).
    """
    def __init__(self, in_channels=1, base_filters=32):
        super().__init__()

        # -----------------------------------------------------------------
        # ENCODER (Downsampling Path)
        # -----------------------------------------------------------------
        self.enc1 = ResConvBlock1D(in_channels, base_filters)
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = ResConvBlock1D(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool1d(2)

        # Bottleneck
        self.enc3 = ResConvBlock1D(base_filters * 2, base_filters * 4)

        # -----------------------------------------------------------------
        # DECODER (Upsampling Path)
        # -----------------------------------------------------------------
        self.up2 = nn.ConvTranspose1d(
            base_filters * 4, base_filters * 2, kernel_size=2, stride=2
        )
        self.dec2 = ResConvBlock1D(base_filters * 4, base_filters * 2)

        self.up1 = nn.ConvTranspose1d(
            base_filters * 2, base_filters, kernel_size=2, stride=2
        )
        self.dec1 = ResConvBlock1D(base_filters * 2, base_filters)

        # -----------------------------------------------------------------
        # TASK HEADS (Divergent Predictions)
        # -----------------------------------------------------------------
        # Head 1: PSD Denoising Head
        self.head_denoise = nn.Sequential(
            nn.Conv1d(base_filters, base_filters // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout1d(p=0.1),
            nn.Conv1d(base_filters // 2, 1, kernel_size=1)
        )

        # Head 2: Peak Center Heatmap Head
        self.head_heatmap = nn.Sequential(
            nn.Conv1d(base_filters, base_filters // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout1d(p=0.1),
            nn.Conv1d(base_filters // 2, 1, kernel_size=1)
        )

        # Head 3: Peak Width Estimation Head
        self.head_width = nn.Sequential(
            nn.Conv1d(base_filters, base_filters // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout1d(p=0.1),
            nn.Conv1d(base_filters // 2, 2, kernel_size=1),
            nn.Softplus()
        )

    def forward(self, x):
        # --- Encoder Pass ---
        e1 = self.enc1(x)               # Shape: [B, base_filters, N]
        e2 = self.enc2(self.pool1(e1))  # Shape: [B, base_filters*2, N/2]
        e3 = self.enc3(self.pool2(e2))  # Shape: [B, base_filters*4, N/4]

        # --- Decoder Pass Stage 2 ---
        d2 = self.up2(e3)
        if d2.shape[-1] != e2.shape[-1]:  # Spatial size safeguard for odd sequence lengths
            d2 = F.interpolate(d2, size=e2.shape[-1], mode='linear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        # --- Decoder Pass Stage 1 ---
        d1 = self.up1(d2)
        if d1.shape[-1] != e1.shape[-1]:  # Spatial size safeguard for odd sequence lengths
            d1 = F.interpolate(d1, size=e1.shape[-1], mode='linear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # --- Multi-Task Predictions ---
        clean_psd      = self.head_denoise(d1)   # [B, 1, N]
        heatmap_logits = self.head_heatmap(d1)   # [B, 1, N]
        widths         = self.head_width(d1)     # [B, 2, N]

        return clean_psd, heatmap_logits, widths



# =====================================================================
# 1. PEAK & BANDWIDTH DECODER (1D NMS)
# =====================================================================
def decode_keypoint_predictions(
    pred_logits,
    pred_widths,
    f_grid,
    center_thresh=0.35,
    kernel_size=5,
    min_width_bins=0.5,
):
    """Decodes keypoint heatmap logits and predicted width vectors into physical frequencies.

    Args:
        pred_logits: Raw heatmap logits [1, 1, n_freqs] or [1, n_freqs]
        pred_widths: Width predictions [1, 2, n_freqs] or [2, n_freqs]
        f_grid: 1D array of frequency values in Hz
        center_thresh: Sigmoid score threshold for peak detection
        kernel_size: Window size for 1D Non-Maximum Suppression
        min_width_bins: Minimum allowable width in bins (prevents collapse)

    Returns:
        peaks_info: List of dicts containing f0, f_left, f_right, and
        heatmap_score
        probs_np: 1D numpy array of heatmap probabilities
    """
    # 1. Ensure 3D Shape [1, C, N]
    if pred_logits.ndim == 2:
        pred_logits = pred_logits.unsqueeze(0)
    if pred_widths.ndim == 2:
        pred_widths = pred_widths.unsqueeze(0)

    # 2. Apply Sigmoid to heatmap logits
    heatmap_prob = torch.sigmoid(pred_logits)  # Shape: [1, 1, n_freqs]

    # 3. Local Maxima Pooling (1D Non-Maximum Suppression)
    pad = (kernel_size - 1) // 2
    hmax = F.max_pool1d(
        heatmap_prob, kernel_size=kernel_size, stride=1, padding=pad
    )
    keep = (heatmap_prob == hmax) & (heatmap_prob >= center_thresh)

    peak_indices = torch.nonzero(keep.squeeze()).squeeze(-1)

    probs_np = heatmap_prob.squeeze().detach().cpu().numpy()

    # Early exit if no peaks found
    if peak_indices.numel() == 0:
        return [], probs_np

    if peak_indices.ndim == 0:
        peak_indices = peak_indices.unsqueeze(0)

    peak_indices = peak_indices.detach().cpu().numpy()

    # Frequency bin step size (df in Hz)
    df = f_grid[1] - f_grid[0] if len(f_grid) > 1 else 1.0

    # Ensure widths are positive (apply ReLU if needed)
    widths_clamped = torch.relu(pred_widths).squeeze().detach().cpu().numpy()

    peaks_info = []

    for idx in peak_indices:
        f0 = f_grid[idx]
        score = probs_np[idx]

        # Read predicted left/right bin widths and clamp to min threshold
        left_bins = max(widths_clamped[0, idx], min_width_bins)
        right_bins = max(widths_clamped[1, idx], min_width_bins)

        # Convert bin offsets into Hz bounds
        f_left = max(f_grid[0], f0 - (left_bins * df))
        f_right = min(f_grid[-1], f0 + (right_bins * df))

        peaks_info.append({
            "idx": int(idx),
            "f0": float(f0),
            "f_left": float(f_left),
            "f_right": float(f_right),
            "bandwidth_hz": float(f_right - f_left),
            "left_bins": float(left_bins),
            "right_bins": float(right_bins),
            "score": float(score),
        })

    return peaks_info, probs_np
# =====================================================================
# 2. KEYPOINT RESULT VISUALIZER
# =====================================================================

def plot_keypoint_psd_inference(f_grid, log_noisy, pred_clean, peaks_info, heatmap_prob, left_width, right_width):
    """
    Plots the empirical noisy log PSD, the network's clean curve reconstruction,
    the continuous 1D peak heatmap, and the detected peak center/interval regions.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 7), sharex=True, gridspec_kw={'height_ratios': [2.5, 1, 1]})

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

    ax3.plot(f_grid, left_width, color="teal", linewidth=1.8, label="Raw Predicted Left Width Δf_L (Hz)")
    ax3.plot(f_grid, right_width, color="purple", linewidth=1.8, linestyle="-.", label="Raw Predicted Right Width Δf_R (Hz)")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Width (Hz)")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right")

    plt.tight_layout()
    plt.show()



# ================================================================
# Load model
# ================================================================

def load_keypoint_model(checkpoint_path="best_keypoint_res_unet_cascade.pth", base_filters=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found!")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # model = MultiTaskUNet1D(in_channels=1, base_filters=base_filters).to(device)
    model = ResMultiTaskUNet1D(in_channels=1, base_filters=base_filters).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded Keypoint UNet from '{checkpoint_path}'")
    return model, device



if __name__ == "__main__":
    CHECKPOINT_PATH = "best_keypoint_res_unet.pth"

    # --- Settings ---
    USE_SIMULATED = True   # Toggle False for real EEG recording
    TARGET_N_FREQS = 250   # Matches training resolution
    TARGET_F_MAX = 50.0    # Frequency ceiling (50 Hz)

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

    # Extract Raw Width Channels from Head 3 (Shape: [2, n_freqs])
    raw_widths = pred_widths.squeeze().cpu().numpy()
    df = f_grid[1] - f_grid[0]  # Frequency bin resolution in Hz

    # Convert raw width predictions from "number of bins" to "Hz"
    raw_left_width_hz  = raw_widths[0, :] * df   # Channel 0: Left Width (Hz)
    raw_right_width_hz = raw_widths[1, :] * df   # Channel 1: Right Width (Hz)

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
        heatmap_prob=heatmap_prob,
        left_width = raw_left_width_hz,
        right_width = raw_right_width_hz
    )