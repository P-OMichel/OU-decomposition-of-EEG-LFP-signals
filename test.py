import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# =====================================================================
# 1. SETUP DUMMY DATA (Simulated Log-PSD with high low-frequency power)
# =====================================================================
f = np.linspace(0.1, 45.0, 500)
# Typical EEG PSD: steep 1/f decay starting around 5.8 log power at 0.1 Hz
log_psd = 5.8 - 2.1 * np.log10(f + 0.2) + 0.5 * np.exp(-((f - 10.0) ** 2) / 2)
# Add small noise
np.random.seed(42)
log_psd_noisy = log_psd + np.random.normal(0, 0.1, size=len(f))

# Convert to PyTorch Tensor: shape (Batch=1, Channels=1, Length=500)
x_tensor = (
    torch.tensor(log_psd_noisy, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
)

# =====================================================================
# 2. STRATEGY 1: VISUALIZING PADDED SEQUENCES
# =====================================================================
kernel_size = 15
pad_len = (kernel_size - 1) // 2  # 7 bins

# A. Zero Padding Manual Construction
zero_padded = np.pad(log_psd_noisy[:20], (pad_len, 0), mode="constant")

# B. Reflection Padding Manual Construction
reflect_padded = np.pad(log_psd_noisy[:20], (pad_len, 0), mode="reflect")

# =====================================================================
# 3. STRATEGY 2: PYTORCH LAYER OUTPUT COMPARISON
# =====================================================================
# Layer A: Zero Padded Conv
conv_zero = nn.Conv1d(
    in_channels=1, out_channels=1, kernel_size=kernel_size, padding=pad_len
)

# Layer B: Reflection Padded Conv
pad_reflect = nn.ReflectionPad1d(pad_len)
conv_reflect = nn.Conv1d(
    in_channels=1, out_channels=1, kernel_size=kernel_size, padding=0
)

# Use identity-like filter weights to observe raw boundary propagation
with torch.no_grad():
    conv_zero.weight.fill_(1.0 / kernel_size)
    conv_zero.bias.zero_()
    conv_reflect.weight.fill_(1.0 / kernel_size)
    conv_reflect.bias.zero_()

    out_zero = conv_zero(x_tensor).squeeze().numpy()
    out_reflect = conv_reflect(pad_reflect(x_tensor)).squeeze().numpy()

# =====================================================================
# 4. PLOTTING & COMPARISON
# =====================================================================
fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))

# --- Subplot 1: What the Convolution Layer "Sees" at 0.1 Hz ---
ax1 = axes[0]
bins_x = np.arange(-pad_len, 20)

ax1.plot(
    bins_x[pad_len:],
    log_psd_noisy[:20],
    "o-",
    color="black",
    linewidth=2,
    label="Original Signal (Starts @ 0.1 Hz)",
)
ax1.plot(
    bins_x[: pad_len + 1],
    zero_padded[: pad_len + 1],
    "s--",
    color="crimson",
    linewidth=1.8,
    label="Zero Padding (Creates artificial 'Cliff')",
)
ax1.plot(
    bins_x[: pad_len + 1],
    reflect_padded[: pad_len + 1],
    "^--",
    color="forestgreen",
    linewidth=1.8,
    label="Reflection Padding (Smooth Mirroring)",
)

ax1.axvline(
    x=0, color="gray", linestyle=":", linewidth=1.5, label="Boundary (0.1 Hz)"
)
ax1.set_title(
    "1. Boundary Input Structure Comparison (First 20 Frequency Bins)",
    fontsize=12,
    fontweight="bold",
)
ax1.set_xlabel("Array Bin Index (0 = 0.1 Hz)", fontsize=10)
ax1.set_ylabel("Log Power", fontsize=10)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper right", fontsize=9)

# --- Subplot 2: Feature Distortions in Convolution Output ---
ax2 = axes[1]

ax2.plot(
    f[:60],
    log_psd_noisy[:60],
    color="gray",
    alpha=0.5,
    linewidth=1.2,
    label="Raw Input Spectrum",
)
ax2.plot(
    f[:60],
    out_zero[:60],
    color="crimson",
    linewidth=2.0,
    label="Conv Output with Zero Padding (Attenuated Edge Artifact)",
)
ax2.plot(
    f[:60],
    out_reflect[:60],
    color="forestgreen",
    linewidth=2.0,
    label="Conv Output with Reflection Padding (Preserved Baseline)",
)

ax2.set_title(
    "2. Convolution Response near Low-Frequency Edge (0.1 Hz to 5.0 Hz)",
    fontsize=12,
    fontweight="bold",
)
ax2.set_xlabel("Frequency (Hz)", fontsize=10)
ax2.set_ylabel("Log Power Feature Output", fontsize=10)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="lower right", fontsize=9)

plt.tight_layout()
plt.show()