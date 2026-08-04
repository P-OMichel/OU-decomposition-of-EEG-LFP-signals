import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks, peak_prominences


# =====================================================================
# 1. ASYMMETRIC WHITTAKER & METHOD 1 INTERVAL EXTRACTOR
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
    f, y_log, f0_list, lam=1e2, p=0.01, max_high_ratio=0.15, min_boundary_ratio=0.6
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
        })

    return intervals_info