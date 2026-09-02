import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks, peak_prominences


# =====================================================================
# 1. YOUR EXACT METHOD 1 IMPLEMENTATION
# =====================================================================
def whittaker_smooth_asymmetric(y, lam=1e4, p=0.01, niter=10):
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


def extract_intervals_method1_detrended_seed(
    f, y_log, f0_list, lam=1e2, p=0.01, max_high_ratio=0.01, min_boundary_ratio=0.5
):
    """Extracts peak frequency intervals (f0, f_left, f_right) using Method 1."""
    if len(f0_list) == 0:
        return []

    sorted_f0 = sorted(f0_list)
    n_peaks = len(sorted_f0)
    intervals_info = []

    midpoints = [(sorted_f0[i] + sorted_f0[i + 1]) / 2.0 for i in range(n_peaks - 1)]
    f_min, f_max = f[0], f[-1]
    bounds = []

    for i in range(n_peaks):
        left = f_min if i == 0 else midpoints[i - 1]
        right = f_max if i == n_peaks - 1 else midpoints[i]
        bounds.append((left, right))

    for i, (left_f, right_f) in enumerate(bounds):
        target_f0 = sorted_f0[i]
        idx = np.where((f >= left_f) & (f <= right_f))[0]
        if len(idx) < 4:
            continue

        sub_f = f[idx]
        sub_y_log = y_log[idx]
        psd_lin = np.exp(sub_y_log)  # Linear PSD for exact power ratios

        baseline_log = whittaker_smooth_asymmetric(sub_y_log, lam=lam, p=p)
        excess_seg = np.maximum(0, sub_y_log - baseline_log)

        target_sub_idx = np.argmin(np.abs(sub_f - target_f0))
        local_peaks, _ = find_peaks(excess_seg, distance=2, prominence=0.01)

        left_edge_f, right_edge_f = None, None

        if len(local_peaks) > 0:
            best_p_idx = np.argmin(np.abs(local_peaks - target_sub_idx))
            best_p = local_peaks[best_p_idx]
            p_max_lin = psd_lin[best_p]

            _, left_bases, right_bases = peak_prominences(excess_seg, local_peaks)
            l_idx = left_bases[best_p_idx]
            r_idx = right_bases[best_p_idx]

            high_side = "left" if psd_lin[l_idx] >= psd_lin[r_idx] else "right"

            # Condition 1: High edge ceiling
            high_target_lin = max_high_ratio * p_max_lin
            if high_side == "left":
                while l_idx < best_p and psd_lin[l_idx] < high_target_lin:
                    l_idx += 1
                val_high_lin = psd_lin[l_idx]
            else:
                while r_idx > best_p and psd_lin[r_idx] < high_target_lin:
                    r_idx -= 1
                val_high_lin = psd_lin[r_idx]

            # Condition 2: Low edge floor
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

            left_edge_f = sub_f[l_idx]
            right_edge_f = sub_f[r_idx]

        intervals_info.append({
            "f0": target_f0,
            "peak_edges": (left_edge_f, right_edge_f),
            "segment_bounds": (left_f, right_f),
            "sub_f": sub_f,
            "sub_y_log": sub_y_log,
            "baseline": baseline_log,
            "excess": excess_seg,
        })

    return intervals_info


# =====================================================================
# 2. DUAL-PANEL METHODOLOGY FIGURE GENERATOR
# =====================================================================
def plot_method1_explanation_figure(
    f, log_clean, center_mask, interval_mask, lam=1e2, p=0.01, sample_idx=0
):
    # Extract peak centers from mask Channel 0
    peak_indices = np.where(center_mask > 0.5)[0]
    f0_list = f[peak_indices].tolist()

    # Run your Method 1 extraction logic
    intervals_info = extract_intervals_method1_detrended_seed(
        f=f, y_log=log_clean, f0_list=f0_list, lam=lam, p=p
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 8.5), sharex=True, gridspec_kw={'height_ratios': [2, 1.3]}
    )

    # =================================================================
    # TOP SUBPLOT: LOG PSD & PIECEWISE BASELINES
    # =================================================================
    ax1.plot(f, log_clean, color="blue", lw=1.8, label="Clean Log PSD Target ($y$)")

    # Target Interval Mask Ground Truth (Channel 1)
    ax1.fill_between(
        f, log_clean.min() - 0.5, log_clean.max() + 0.5, where=(interval_mask > 0.5),
        color="orange", alpha=0.15, label="Target Interval (Mask Ch 1)"
    )

    for i, info in enumerate(intervals_info):
        sub_f = info["sub_f"]
        sub_y = info["sub_y_log"]
        base = info["baseline"]
        left_b, right_b = info["segment_bounds"]
        left_e, right_e = info["peak_edges"]

        # Midpoint Segment Boundaries
        ax1.axvline(
            left_b, color="black", linestyle=":", alpha=0.5,
            label="Segment Boundary (Midpoint)" if i == 0 else ""
        )
        if i == len(intervals_info) - 1:
            ax1.axvline(right_b, color="black", linestyle=":", alpha=0.5)

        # Local Whittaker Baseline Fit Line
        ax1.plot(
            sub_f, base, color="magenta", linestyle="--", lw=2.0,
            label="Local Whittaker Baseline ($z_i$)" if i == 0 else ""
        )

        # Highlight Method 1 Extracted Intervals
        if left_e is not None and right_e is not None:
            ax1.axvspan(
                left_e, right_e, color="cyan", alpha=0.35,
                label="Method 1 Extracted Interval" if i == 0 else ""
            )

    # Draw Peak Centers (Channel 0)
    for i, f0 in enumerate(f0_list):
        ax1.axvline(
            f0, color="red", linestyle="--", lw=1.2, alpha=0.8,
            label="Peak Center (Mask Ch 0)" if i == 0 else ""
        )

    ax1.set_title(
        f"Method 1: Piecewise Whittaker Fits & Extracted Intervals — Sample #{sample_idx}",
        fontsize=12, pad=10
    )
    ax1.set_ylabel("Log PSD", fontsize=11)
    ax1.set_ylim(log_clean.min() - 0.2, log_clean.max() + 0.5)
    ax1.grid(True, linestyle=":", alpha=0.6)

    handles1, labels1 = ax1.get_legend_handles_labels()
    by_label1 = dict(zip(labels1, handles1))
    ax1.legend(by_label1.values(), by_label1.keys(), loc="upper right", framealpha=0.95, fontsize=9.5)

    # =================================================================
    # BOTTOM SUBPLOT: ISOLATED DETRENDED EXCESS CURVES PER REGION
    # =================================================================
    ax2.axhline(0, color="gray", linestyle="-", lw=1.0, alpha=0.7)

    for i, info in enumerate(intervals_info):
        sub_f = info["sub_f"]
        excess = info["excess"]
        left_b, right_b = info["segment_bounds"]

        # Midpoint Segment Boundaries
        ax2.axvline(left_b, color="black", linestyle=":", alpha=0.5)
        if i == len(intervals_info) - 1:
            ax2.axvline(right_b, color="black", linestyle=":", alpha=0.5)

        # Local Detrended Excess Curve
        ax2.plot(
            sub_f, excess, color="teal", lw=1.8,
            label="Detrended Excess ($y - z_i > 0$)" if i == 0 else ""
        )

        # Fill under detrended excess curve
        ax2.fill_between(sub_f, 0, excess, color="cyan", alpha=0.35)

    # Peak Centers on bottom plot
    for i, f0 in enumerate(f0_list):
        ax2.axvline(f0, color="red", linestyle="--", lw=1.2, alpha=0.8)

    ax2.set_title("Isolated Local Detrended Excess Curves ($y - z_i$ per Region)", fontsize=11, pad=8)
    ax2.set_xlabel("Frequency (Hz)", fontsize=11)
    ax2.set_ylabel("Detrended Excess", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)

    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label2 = dict(zip(labels2, handles2))
    ax2.legend(by_label2.values(), by_label2.keys(), loc="upper right", framealpha=0.95, fontsize=9.5)

    plt.tight_layout()
    plt.show()


# =====================================================================
# 3. DEMO EXECUTION
# =====================================================================
if __name__ == "__main__":
    DATASET_PATH = "psd_dataset_masks.npz"

    try:
        data = np.load(DATASET_PATH, allow_pickle=True)
        f = data["f"]
        log_clean_train = data["log_clean_train"]
        masks_train = data["masks_train"]

        sample_idx = 12

        plot_method1_explanation_figure(
            f=f,
            log_clean=log_clean_train[sample_idx],
            center_mask=masks_train[sample_idx, 0],   # Channel 0: Centers
            interval_mask=masks_train[sample_idx, 1], # Channel 1: Intervals
            lam=1e2,
            p=0.01,
            sample_idx=sample_idx
        )

    except FileNotFoundError:
        print(f"File '{DATASET_PATH}' not found. Check the file path.")