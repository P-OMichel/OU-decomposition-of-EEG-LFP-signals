import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks, peak_prominences


# =====================================================================
# 1. DATA LOADER
# =====================================================================
def load_psd_sample(filepath="psd_dataset_splits_high_amplitude.npz", split="train", sample_idx=0):
    """
    Loads a single sample from the saved NPZ dataset.
    """
    data = np.load(filepath, allow_pickle=True)

    f = data["f"]
    log_noisy = data[f"log_noisy_{split}"][sample_idx]
    log_clean = data[f"log_clean_{split}"][sample_idx]
    try:
        meta = data[f"metadata_{split}"][sample_idx]
    except:
        meta = None

    f0_list = [peak["f0"] for peak in meta.get("peaks", [])]

    return f, log_noisy, log_clean, f0_list, meta


# =====================================================================
# 2. ASYMMETRIC WHITTAKER SMOOTHER
# =====================================================================
def whittaker_smooth_asymmetric(y, lam=1e4, p=0.01, niter=10):
    """
    Fits an asymmetric Whittaker baseline to log(PSD).
    
    Parameters:
        lam: smoothing weight (higher = stiffer continuum)
        p: asymmetry weight (p << 0.5 forces baseline underneath peaks)
        niter: re-weighting iterations
    """
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
# METHOD 1: DETRENDED SEED + LINEAR-SPACE PSD REFINEMENT
# =====================================================================
def extract_intervals_method1_detrended_seed(
    f, y_log, f0_list, lam=1e4, p=0.01, max_high_ratio=0.05, min_boundary_ratio=0.5
):
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
        
        # --- CRITICAL: Convert to Linear Power Scale ---
        psd_lin = np.exp(sub_y_log)

        # Baseline & Detrended Excess
        baseline_log = whittaker_smooth_asymmetric(sub_y_log, lam=lam, p=p)
        excess_seg = np.maximum(0, sub_y_log - baseline_log)

        target_sub_idx = np.argmin(np.abs(sub_f - target_f0))
        local_peaks, _ = find_peaks(excess_seg, distance=2, prominence=0.01)

        l_idx, r_idx = None, None

        if len(local_peaks) > 0:
            best_p_idx = np.argmin(np.abs(local_peaks - target_sub_idx))
            best_p = local_peaks[best_p_idx]
            
            # Linear peak maximum
            p_max_lin = psd_lin[best_p]

            # Initial prominence boundaries on detrended signal
            _, left_bases, right_bases = peak_prominences(excess_seg, local_peaks)
            l_idx = left_bases[best_p_idx]
            r_idx = right_bases[best_p_idx]

            # Determine initial higher boundary on Linear PSD
            high_side = "left" if psd_lin[l_idx] >= psd_lin[r_idx] else "right"

            # --- CONDITION 1: Push higher edge to >= (max_high_ratio * P_peak) ---
            high_target_lin = max_high_ratio * p_max_lin

            if high_side == "left":
                while l_idx < best_p and psd_lin[l_idx] < high_target_lin:
                    l_idx += 1
                val_high_lin = psd_lin[l_idx]
            else:
                while r_idx > best_p and psd_lin[r_idx] < high_target_lin:
                    r_idx -= 1
                val_high_lin = psd_lin[r_idx]

            # --- CONDITION 2: Ensure lower edge >= (min_boundary_ratio * P_high) ---
            low_target_lin = min_boundary_ratio * val_high_lin

            if high_side == "right":
                # Left is lower -> push left inward toward peak
                while l_idx < best_p and psd_lin[l_idx] < low_target_lin:
                    if l_idx + 1 < best_p and psd_lin[l_idx + 1] > val_high_lin:
                        break
                    l_idx += 1
            else:
                # Right is lower -> push right inward toward peak
                while r_idx > best_p and psd_lin[r_idx] < low_target_lin:
                    if r_idx - 1 > best_p and psd_lin[r_idx - 1] > val_high_lin:
                        break
                    r_idx -= 1

            left_edge_f = sub_f[l_idx]
            right_edge_f = sub_f[r_idx]
        else:
            left_edge_f, right_edge_f = None, None

        intervals_info.append({
            "peak_idx": i,
            "f0": target_f0,
            "sub_f": sub_f,
            "sub_y_log": sub_y_log,
            "baseline": baseline_log,
            "excess_seg": excess_seg,
            "segment_bounds": (left_f, right_f),
            "peak_edges": (left_edge_f, right_edge_f),
            "edge_indices": (l_idx, r_idx)
        })

    return intervals_info


# =====================================================================
# METHOD 2: DIRECT LINEAR-SPACE PSD BOUNDARY EXTRACTION
# =====================================================================
def extract_intervals_method2_direct_psd(
    f, y_log, f0_list, max_high_ratio=0.05, min_boundary_ratio=0.5
):
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

        # --- CRITICAL: Convert to Linear Power Scale ---
        psd_lin = np.exp(sub_y_log)

        # Peak index closest to target f0
        best_p = np.argmin(np.abs(sub_f - target_f0))
        p_max_lin = psd_lin[best_p]

        # Initial search starts at segment boundaries
        l_idx = 0
        r_idx = len(sub_f) - 1

        high_side = "left" if psd_lin[l_idx] >= psd_lin[r_idx] else "right"

        # --- CONDITION 1: Push higher edge to >= (max_high_ratio * P_peak) ---
        high_target_lin = max_high_ratio * p_max_lin

        if high_side == "left":
            while l_idx < best_p and psd_lin[l_idx] < high_target_lin:
                l_idx += 1
            val_high_lin = psd_lin[l_idx]
        else:
            while r_idx > best_p and psd_lin[r_idx] < high_target_lin:
                r_idx -= 1
            val_high_lin = psd_lin[r_idx]

        # --- CONDITION 2: Ensure lower edge >= (min_boundary_ratio * P_high) ---
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
            "peak_idx": i,
            "f0": target_f0,
            "sub_f": sub_f,
            "sub_y_log": sub_y_log,
            "segment_bounds": (left_f, right_f),
            "peak_edges": (left_edge_f, right_edge_f),
            "edge_indices": (l_idx, r_idx)
        })

    return intervals_info

# =====================================================================
# 4. RUN DEMO & PLOT
# =====================================================================
if __name__ == "__main__":
    filepath = "psd_dataset_splits_high_amplitude.npz"
    split = "train"
    sample_idx = 12 # 4 to visualize effect on asymetric peak

    # 1. Load data
    f, log_noisy, log_clean, f0_list, meta = load_psd_sample(
        filepath=filepath, split=split, sample_idx=sample_idx
    )

    print(f"Sample #{sample_idx} loaded with {len(f0_list)} peaks at: {f0_list}")

    # 2. Extract intervals using both methods
    if len(f0_list) > 0:
        # Method 1: Detrended Seed + PSD Refinement
        intervals_m1 = extract_intervals_method1_detrended_seed(
            f, log_clean, f0_list, lam=1e2, max_high_ratio=0.01, min_boundary_ratio=0.5
        )
        # Method 2: Direct PSD Refinement (No baseline fit)
        intervals_m2 = extract_intervals_method2_direct_psd(
            f, log_clean, f0_list, max_high_ratio=0.01, min_boundary_ratio=0.5
        )

        # 3. Visualization Setup (2 Subplots for direct comparison)
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

        # =================================================================
        # TOP PLOT: Method 1 (Whittaker Baseline + Prominence Seed)
        # =================================================================
        axes[0].plot(f, log_clean, label="Clean Ground Truth Log PSD", color="blue", lw=1.5)
        axes[0].set_title(f"Method 1: Detrended Seed + PSD Refinement (Sample #{sample_idx})")
        axes[0].set_ylabel("Log PSD")
        axes[0].grid(True, linestyle=":", alpha=0.6)

        for i, info in enumerate(intervals_m1):
            f0 = info["f0"]
            left_b, right_b = info["segment_bounds"]
            left_e, right_e = info["peak_edges"]

            # Baseline fit
            w_label = "Asymmetric Whittaker Baseline" if i == 0 else ""
            axes[0].plot(info["sub_f"], info["baseline"], color="magenta", linestyle="--", lw=1.8, label=w_label)

            # Division boundaries
            div_label = "Segment Boundary" if i == 0 else ""
            axes[0].axvline(left_b, color="black", linestyle=":", alpha=0.5, label=div_label)
            axes[0].axvline(right_b, color="black", linestyle=":", alpha=0.5)

            # Peak marker
            p_label = "True Peak (f0)" if i == 0 else ""
            axes[0].axvline(f0, color="red", linestyle="--", alpha=0.7, label=p_label)

            # Highlight extracted interval
            int_label = "Extracted Interval (M1)" if i == 0 else ""
            if left_e is not None and right_e is not None:
                axes[0].axvspan(left_e, right_e, color="orange", alpha=0.25, label=int_label)

        # =================================================================
        # BOTTOM PLOT: Method 2 (Direct PSD - No Baseline Detrending)
        # =================================================================
        axes[1].plot(f, log_clean, label="Clean Ground Truth Log PSD", color="blue", lw=1.5)
        axes[1].set_title(f"Method 2: Direct PSD Refinement (No Detrending)")
        axes[1].set_ylabel("Log PSD")
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].grid(True, linestyle=":", alpha=0.6)

        for i, info in enumerate(intervals_m2):
            f0 = info["f0"]
            left_b, right_b = info["segment_bounds"]
            left_e, right_e = info["peak_edges"]

            # Division boundaries
            div_label = "Segment Boundary" if i == 0 else ""
            axes[1].axvline(left_b, color="black", linestyle=":", alpha=0.5, label=div_label)
            axes[1].axvline(right_b, color="black", linestyle=":", alpha=0.5)

            # Peak marker
            p_label = "True Peak (f0)" if i == 0 else ""
            axes[1].axvline(f0, color="red", linestyle="--", alpha=0.7, label=p_label)

            # Highlight extracted interval
            int_label = "Extracted Interval (M2)" if i == 0 else ""
            if left_e is not None and right_e is not None:
                axes[1].axvspan(left_e, right_e, color="green", alpha=0.25, label=int_label)

        # Deduplicate and format legends
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper right")

        plt.tight_layout()
        plt.show()