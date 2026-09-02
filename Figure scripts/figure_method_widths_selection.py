import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, peak_prominences
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve

# =====================================================================
# 1. CORE ALGORITHMIC FUNCTIONS
# =====================================================================
def whittaker_smooth_asymmetric(y, lam=1e2, p=0.01, niter=15):
    """Fits an asymmetric Whittaker baseline to log(PSD)."""
    L = len(y)
    if L < 4:
        return np.copy(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    D = csc_matrix(D)
    w = np.ones(L)
    for _ in range(niter):
        W = diags([w], [0], shape=(L, L))
        Z = W + lam * D.T * D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


# =====================================================================
# 2. DATA LOADING / SYNTHESIS FALLBACK
# =====================================================================
DATASET_PATH = "psd_dataset_masks.npz"
sample_idx = 12

try:
    data = np.load(DATASET_PATH, allow_pickle=True)
    f = data["f"]
    log_clean = data["log_clean_train"][sample_idx]
    center_mask = data["masks_train"][sample_idx, 0]
    peak_indices = np.where(center_mask > 0.5)[0]
    f0_list = sorted(f[peak_indices].tolist())
except Exception:
    # Synthetic fallback if NPZ file is unavailable
    print("Dataset not found. Generating synthetic multi-peak PSD fallback.")
    f = np.linspace(1.0, 40.0, 600)
    background = 5.0 - 1.8 * np.log(f)
    peak1 = 1.2 * np.exp(-0.5 * ((f - 8.0) / 1.1) ** 2)
    peak2 = 2.4 * np.exp(-0.5 * ((f - 18.0) / 1.8) ** 2)
    peak3 = 0.9 * np.exp(-0.5 * ((f - 30.0) / 1.4) ** 2)
    log_clean = background + peak1 + peak2 + peak3
    f0_list = [8.0, 18.0, 30.0]

# Choose target peak index to illustrate (middle peak by default)
target_peak_idx = 1 if len(f0_list) > 1 else 0
target_f0 = f0_list[target_peak_idx]

# =====================================================================
# 3. STEP-BY-STEP METHOD EXECUTION (TARGET PEAK)
# =====================================================================
n_peaks = len(f0_list)
midpoints = [(f0_list[k] + f0_list[k + 1]) / 2.0 for k in range(n_peaks - 1)]

# Step 1: Subdivision Bounds
f_min, f_max = f[0], f[-1]
f_a_i = f_min if target_peak_idx == 0 else midpoints[target_peak_idx - 1]
f_b_i = f_max if target_peak_idx == n_peaks - 1 else midpoints[target_peak_idx]

idx = np.where((f >= f_a_i) & (f <= f_b_i))[0]
sub_f = f[idx]
sub_y_log = log_clean[idx]
psd_lin = np.exp(sub_y_log)

# Step 2: Asymmetric Whittaker Baseline & Detrending
lam_val, p_val = 1e2, 0.01
b_sub = whittaker_smooth_asymmetric(sub_y_log, lam=lam_val, p=p_val)
e_sub = np.maximum(0, sub_y_log - b_sub)

# Step 3: Peak Seeding & Initial Base Identification
target_sub_idx = np.argmin(np.abs(sub_f - target_f0))
local_peaks, _ = find_peaks(e_sub, distance=2, prominence=0.01)

best_p_idx = np.argmin(np.abs(local_peaks - target_sub_idx))
best_p = local_peaks[best_p_idx]
f0_star = sub_f[best_p]
p_max_lin = psd_lin[best_p]

_, left_bases, right_bases = peak_prominences(e_sub, local_peaks)
l_init = left_bases[best_p_idx]
r_init = right_bases[best_p_idx]
f_l_init, f_r_init = sub_f[l_init], sub_f[r_init]

# Step 4: Asymmetric Power-Ratio Boundary Refinement
l_idx, r_idx = l_init, r_init
max_high_ratio = 0.15
min_boundary_ratio = 0.50

high_side = "left" if psd_lin[l_idx] >= psd_lin[r_idx] else "right"
high_target_lin = max_high_ratio * p_max_lin

if high_side == "left":
    while l_idx < best_p and psd_lin[l_idx] < high_target_lin:
        l_idx += 1
    val_high_lin = psd_lin[l_idx]
else:
    while r_idx > best_p and psd_lin[r_idx] < high_target_lin:
        r_idx -= 1
    val_high_lin = psd_lin[r_idx]

low_target_lin = min_boundary_ratio * val_high_lin

if high_side == "right":
    while l_idx < best_p and psd_lin[l_idx] < low_target_lin:
        if l_idx + 1 < best_p and psd_lin[l_idx + 1] > val_high_lin:
            break
        l_idx += 1
else:
    while r_idx > best_p and psd_lin[r_idx] < low_target_lin:
        if r_idx - 1 > best_p and psd_lin[r_idx - 1] > val_high_lin:
            break
        r_idx -= 1

f_left_final = sub_f[l_idx]
f_right_final = sub_f[r_idx]

# =====================================================================
# 4. PUBLICATION-READY 4-PANEL FIGURE GENERATION
# =====================================================================
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 12,
})

fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True, constrained_layout=True)

# ---------------------------------------------------------------------
# Panel 1: Step 1 - Spectral Partitioning
# ---------------------------------------------------------------------
axes[0].plot(f, log_clean, color="#1f77b4", lw=1.8, label=r"Global Log-PSD $\log(\mathbf{S})$")
axes[0].axvspan(f_a_i, f_b_i, color="#ff7f0e", alpha=0.15, label=r"Target Interval $[f_{a_i}, f_{b_i}]$")
for f0_val in f0_list:
    axes[0].axvline(f0_val, color="red", linestyle=":", lw=1.2)
axes[0].axvline(f_a_i, color="black", linestyle="--", lw=1.2)
axes[0].axvline(f_b_i, color="black", linestyle="--", lw=1.2)

axes[0].text(f_a_i, axes[0].get_ylim()[0] + 0.15, r" $f_{a_i}$", color="black", va="bottom", fontweight="bold")
axes[0].text(f_b_i, axes[0].get_ylim()[0] + 0.15, r" $f_{b_i}$", color="black", va="bottom", fontweight="bold")
axes[0].text(target_f0, log_clean[np.argmin(np.abs(f - target_f0))] + 0.15, r"$f_0^{(i)}$", color="red", ha="center", fontweight="bold")

axes[0].set_title(r"Step 1: Partitioning into Local Subdivisions $[f_{a_i}, f_{b_i}]$", loc="left", fontweight="bold")
axes[0].set_ylabel(r"$\log(\mathbf{S})$")
axes[0].legend(loc="upper right", framealpha=0.9)
axes[0].grid(True, linestyle=":", alpha=0.5)

# ---------------------------------------------------------------------
# Panel 2: Step 2 - Baseline Estimation
# ---------------------------------------------------------------------
axes[1].plot(sub_f, sub_y_log, color="#1f77b4", lw=1.8, label=r"Local Log-PSD $\log(\mathbf{S}_{\mathrm{sub}}^{(i)})$")
axes[1].plot(sub_f, b_sub, color="#d62728", lw=2.0, linestyle="--", label=r"Asymmetric Baseline $\mathbf{b}_{\mathrm{sub}}^{(i)}$ ($\lambda=10^2, p=0.01$)")
axes[1].axvline(f_a_i, color="black", linestyle="--", lw=1.0)
axes[1].axvline(f_b_i, color="black", linestyle="--", lw=1.0)

axes[1].set_title(r"Step 2: Asymmetric Whittaker Baseline Fitting", loc="left", fontweight="bold")
axes[1].set_ylabel(r"$\log(\mathbf{S}_{\mathrm{sub}}^{(i)})$")
axes[1].legend(loc="upper right", framealpha=0.9)
axes[1].grid(True, linestyle=":", alpha=0.5)

# ---------------------------------------------------------------------
# Panel 3: Step 3 - Rectified Excess & Prominence Base Seeding
# ---------------------------------------------------------------------
axes[2].axhline(0, color="gray", lw=1.0)
axes[2].plot(sub_f, e_sub, color="#008080", lw=1.8, label=r"Rectified Excess $\mathbf{e}_{\mathrm{sub}}^{(i)} = \max(0, \log(\mathbf{S}_{\mathrm{sub}}^{(i)}) - \mathbf{b}_{\mathrm{sub}}^{(i)})$")
axes[2].fill_between(sub_f, 0, e_sub, color="#008080", alpha=0.15)

# Peak apex and initial base markers
axes[2].plot(f0_star, e_sub[best_p], "ro", markersize=6, label=r"Detected Apex $f_0^{(i),*}$")
axes[2].plot([f_l_init, f_r_init], [e_sub[l_init], e_sub[r_init]], "ks", markersize=5, label=r"Initial Bases $(f_l^{(i)}, f_r^{(i)})$")

axes[2].annotate(r"$f_0^{(i),*}$", xy=(f0_star, e_sub[best_p]), xytext=(f0_star, e_sub[best_p] + 0.18),
                ha="center", arrowprops=dict(arrowstyle="->", color="red", lw=1.0))
axes[2].annotate(r"$f_l^{(i)}$", xy=(f_l_init, e_sub[l_init]), xytext=(f_l_init - 1.2, e_sub[l_init] + 0.15),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.9))
axes[2].annotate(r"$f_r^{(i)}$", xy=(f_r_init, e_sub[r_init]), xytext=(f_r_init + 0.6, e_sub[r_init] + 0.15),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.9))

axes[2].set_title(r"Step 3: Topological Peak Finding \& Initial Prominence Bases", loc="left", fontweight="bold")
axes[2].set_ylabel(r"Excess $\mathbf{e}_{\mathrm{sub}}^{(i)}$")
axes[2].legend(loc="upper right", framealpha=0.9)
axes[2].grid(True, linestyle=":", alpha=0.5)

# ---------------------------------------------------------------------
# Panel 4: Step 4 - Boundary Refinement
# ---------------------------------------------------------------------
axes[3].plot(sub_f, sub_y_log, color="#1f77b4", lw=1.8, label=r"$\log(\mathbf{S}_{\mathrm{sub}}^{(i)})$")
axes[3].plot(sub_f, b_sub, color="#d62728", lw=1.2, linestyle=":")

# Mark Initial and Final Intervals
axes[3].axvspan(f_l_init, f_r_init, color="gray", alpha=0.2, label=r"Initial Prominence Interval $[f_l^{(i)}, f_r^{(i)}]$")
axes[3].axvspan(f_left_final, f_right_final, color="#2ca02c", alpha=0.35, label=r"Final Refined Interval $[f_{\mathrm{left}}, f_{\mathrm{right}}]$")

# Inward contraction indicator arrows
if f_left_final > f_l_init:
    axes[3].annotate("", xy=(f_left_final, sub_y_log[l_idx]), xytext=(f_l_init, sub_y_log[l_init]),
                    arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.8))
if f_right_final < f_r_init:
    axes[3].annotate("", xy=(f_right_final, sub_y_log[r_idx]), xytext=(f_r_init, sub_y_log[r_init]),
                    arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.8))

axes[3].text(f_left_final, sub_y_log[l_idx] - 0.25, r"$f_{\mathrm{left}}$", color="darkgreen", ha="right", fontweight="bold")
axes[3].text(f_right_final, sub_y_log[r_idx] - 0.25, r"$f_{\mathrm{right}}$", color="darkgreen", ha="left", fontweight="bold")

axes[3].set_title(r"Step 4: Asymmetric Power-Ratio Inward Boundary Refinement", loc="left", fontweight="bold")
axes[3].set_xlabel(r"Frequency (Hz)")
axes[3].set_ylabel(r"$\log(\mathbf{S}_{\mathrm{sub}}^{(i)})$")
axes[3].legend(loc="upper right", framealpha=0.9)
axes[3].grid(True, linestyle=":", alpha=0.5)

plt.show()