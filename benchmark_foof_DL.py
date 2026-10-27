"""
Unified head-to-head multi-noise benchmark comparing ResMultiTaskUNet1D against FOOOF
across progressive noise tiers (decreasing Welch segments K), with Sub-Bin Peak Refinement.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from fooof import FOOOF
from DL.models.models import ResMultiTaskUNet1D, DeformResMultiTaskUNet1D, TokenResMultiTaskUNet1D


# =====================================================================
# 1. HELPER FUNCTIONS WITH SUB-BIN PARABOLIC REFINEMENT
# =====================================================================
def decode_peak_frequencies_subbin(pred_logits, f_grid, center_thresh=0.35, kernel_size=5):
    """
    Decodes peak centers using 1D local NMS, then applies 3-point parabolic
    sub-bin interpolation on the heatmap response to achieve sub-grid precision.
    """
    if pred_logits.ndim == 1:
        pred_logits = pred_logits.unsqueeze(0).unsqueeze(0)
    elif pred_logits.ndim == 2:
        pred_logits = pred_logits.unsqueeze(0)

    heatmap_prob = torch.sigmoid(pred_logits)
    pad = (kernel_size - 1) // 2
    hmax = F.max_pool1d(heatmap_prob, kernel_size=kernel_size, stride=1, padding=pad)
    keep = (heatmap_prob == hmax) & (heatmap_prob >= center_thresh)

    peak_indices = torch.nonzero(keep.squeeze()).squeeze(-1)
    if peak_indices.numel() == 0:
        return np.array([], dtype=np.float64)
    if peak_indices.ndim == 0:
        peak_indices = peak_indices.unsqueeze(0)

    indices = peak_indices.detach().cpu().numpy()
    prob_1d = heatmap_prob.squeeze().detach().cpu().numpy()
    n_bins = len(f_grid)

    refined_peaks = []
    for idx in indices:
        # Check boundary condition for 3-point neighborhood
        if 0 < idx < n_bins - 1:
            y_left = prob_1d[idx - 1]
            y_center = prob_1d[idx]
            y_right = prob_1d[idx + 1]

            # Parabolic vertex displacement delta in bins
            denom = 2.0 * (y_left - 2.0 * y_center + y_right)
            if abs(denom) > 1e-6:
                delta = (y_left - y_right) / denom
                # Clamp delta to [-0.5, 0.5] bin space
                delta = np.clip(delta, -0.5, 0.5)
            else:
                delta = 0.0

            # Grid step size around the local maximum
            df = 0.5 * (f_grid[idx + 1] - f_grid[idx - 1])
            refined_freq = f_grid[idx] + delta * df
        else:
            refined_freq = f_grid[idx]

        refined_peaks.append(refined_freq)

    return np.array(refined_peaks, dtype=np.float64)


def match_peaks_and_compute_errors(pred_peaks, gt_peaks, max_dist_hz=2.0):
    n_p = len(pred_peaks)
    n_g = len(gt_peaks)

    if n_p == 0 and n_g == 0:
        return [], 0, 0, 0
    if n_p == 0:
        return [], 0, 0, n_g
    if n_g == 0:
        return [], 0, n_p, 0

    dist_matrix = np.abs(pred_peaks[:, None] - gt_peaks[None, :])
    pred_idx, gt_idx = linear_sum_assignment(dist_matrix)

    abs_errors = []
    matched_p = set()
    matched_g = set()

    for p, g in zip(pred_idx, gt_idx):
        err = abs(pred_peaks[p] - gt_peaks[g])
        if err <= max_dist_hz:
            abs_errors.append(err)
            matched_p.add(p)
            matched_g.add(g)

    tp = len(matched_p)
    fp = n_p - tp
    fn = n_g - len(matched_g)

    return abs_errors, tp, fp, fn


# =====================================================================
# 3. MULTI-NOISE JOINT BENCHMARK LOOP
# =====================================================================
def run_multinoise_joint_benchmark(
    model,
    dataset_path="DL/data/eeg_psd_multinoise_benchmark_foof_advantage.npz",
    checkpoint_path="DL_checkpoints/Baseline_UNet.pth",
    batch_size=64,
    center_thresh=0.35,
    max_match_dist=2.0,
    freq_range=None,
    fooof_aperiodic="knee",
    fooof_peak_width=[0.8, 8.0],
    fooof_max_peaks=4,
    fooof_min_height=0.25,
    fooof_threshold=2.5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'checkpoint: {checkpoint_path}')
    # Load DL Model
    model.to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"Loaded weights from '{checkpoint_path}'")
    else:
        print(f"Warning: Checkpoint '{checkpoint_path}' not found! Using initialized weights.")
    model.eval()

    # Load Multi-Noise Dataset
    data = np.load(dataset_path, allow_pickle=True)
    f = data["f"].astype(np.float64)
    k_levels = data["noise_levels_k"]
    n_levels = len(k_levels)

    log_noisy_all = data["log_noisy"]  # [N, n_levels, n_freqs]
    log_clean_all = data["log_clean"]  # [N, n_freqs]
    masks_all = data["masks"]          # [N, 2, n_freqs]
    n_samples = len(log_clean_all)

    f_fooof = np.linspace(f[0], f[-1], len(f), dtype=np.float64)
    if freq_range is None:
        freq_range = [float(f_fooof[0]), float(f_fooof[-1])]
    freq_mask = (f_fooof >= freq_range[0]) & (f_fooof <= freq_range[1])

    print(f"Benchmarking {n_samples} samples across {n_levels} noise levels (Welch K = {list(k_levels)})...")
    print("Using Sub-Bin Parabolic Refinement for UNet peak centers.")

    results = {
        "dl": {"f1": [], "mae_pos": [], "rmse_recon": []},
        "fooof": {"f1": [], "mae_pos": [], "rmse_recon": []},
    }

    sample_vis_idx = 0
    vis_data = {
        "f_eval": f_fooof[freq_mask],
        "gt_clean": log_clean_all[sample_vis_idx][freq_mask],
        "noisy_low": log_noisy_all[sample_vis_idx, 0][freq_mask],
        "noisy_high": log_noisy_all[sample_vis_idx, -1][freq_mask],
        "dl_low": None,
        "dl_high": None,
        "fooof_low": None,
        "fooof_high": None,
    }

    for lvl_idx, k_val in enumerate(k_levels):
        print(f"\nEvaluating Noise Level {lvl_idx + 1}/{n_levels} (Welch K = {k_val})...")

        fm = FOOOF(
            aperiodic_mode=fooof_aperiodic,
            peak_width_limits=fooof_peak_width,
            max_n_peaks=fooof_max_peaks,
            min_peak_height=fooof_min_height,
            peak_threshold=fooof_threshold,
            verbose=False,
        )

        dl_tp, dl_fp, dl_fn = 0, 0, 0
        fooof_tp, fooof_fp, fooof_fn = 0, 0, 0
        dl_abs_errors, fooof_abs_errors = [], []
        dl_spec_rmse, fooof_spec_rmse = [], []

        # 1. Batched UNet Inference
        dl_clean_preds = []
        dl_pred_peaks_list = []

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_inputs = torch.tensor(
                log_noisy_all[start_idx:end_idx, lvl_idx], dtype=torch.float32
            ).unsqueeze(1).to(device)

            with torch.no_grad():
                c_preds, batch_logits, _ = model(batch_inputs)
                dl_clean_preds.extend(c_preds.squeeze(1).cpu().numpy())

            for b in range(end_idx - start_idx):
                # Apply Sub-Bin Refinement
                p_f = decode_peak_frequencies_subbin(batch_logits[b, 0], f_fooof, center_thresh=center_thresh)
                p_f = p_f[(p_f >= freq_range[0]) & (p_f <= freq_range[1])]
                dl_pred_peaks_list.append(p_f)

        # 2. Iterate Samples for UNet and FOOOF
        for i in range(n_samples):
            # Target Ground Truth
            gt_indices = np.where(masks_all[i, 0] > 0.5)[0]
            gt_f = f_fooof[gt_indices]
            gt_f = gt_f[(gt_f >= freq_range[0]) & (gt_f <= freq_range[1])]
            target_clean_curve = log_clean_all[i][freq_mask]

            # UNet Evaluation
            pred_f_unet = dl_pred_peaks_list[i]
            u_errs, u_tp, u_fp, u_fn = match_peaks_and_compute_errors(pred_f_unet, gt_f, max_dist_hz=max_match_dist)
            dl_abs_errors.extend(u_errs)
            dl_tp += u_tp
            dl_fp += u_fp
            dl_fn += u_fn

            unet_curve = dl_clean_preds[i][freq_mask]
            dl_spec_rmse.append(np.sqrt(np.mean((unet_curve - target_clean_curve) ** 2)))

            # FOOOF Evaluation
            psd_linear = np.exp(log_noisy_all[i, lvl_idx]).astype(np.float64)
            fm.fit(f_fooof, psd_linear, freq_range)

            if fm.has_model and len(fm.peak_params_) > 0:
                pred_f_fooof = fm.peak_params_[:, 0]
            else:
                pred_f_fooof = np.array([])

            f_errs, f_tp, f_fp, f_fn = match_peaks_and_compute_errors(pred_f_fooof, gt_f, max_dist_hz=max_match_dist)
            fooof_abs_errors.extend(f_errs)
            fooof_tp += f_tp
            fooof_fp += f_fp
            fooof_fn += f_fn

            if fm.has_model:
                fooof_curve_ln = fm.fooofed_spectrum_ * np.log(10.0)
                fooof_spec_rmse.append(np.sqrt(np.mean((fooof_curve_ln - target_clean_curve) ** 2)))
            else:
                fooof_curve_ln = log_noisy_all[i, lvl_idx][freq_mask]
                fooof_spec_rmse.append(np.nan)

            if i == sample_vis_idx:
                if lvl_idx == 0:
                    vis_data["dl_low"] = unet_curve
                    vis_data["fooof_low"] = fooof_curve_ln
                elif lvl_idx == n_levels - 1:
                    vis_data["dl_high"] = unet_curve
                    vis_data["fooof_high"] = fooof_curve_ln

        # Compute level statistics
        prec_u = dl_tp / (dl_tp + dl_fp) if (dl_tp + dl_fp) > 0 else 0.0
        rec_u = dl_tp / (dl_tp + dl_fn) if (dl_tp + dl_fn) > 0 else 0.0
        f1_u = 2 * prec_u * rec_u / (prec_u + rec_u) if (prec_u + rec_u) > 0 else 0.0

        prec_f = fooof_tp / (fooof_tp + fooof_fp) if (fooof_tp + fooof_fp) > 0 else 0.0
        rec_f = fooof_tp / (fooof_tp + fooof_fn) if (fooof_tp + fooof_fn) > 0 else 0.0
        f1_f = 2 * prec_f * rec_f / (prec_f + rec_f) if (prec_f + rec_f) > 0 else 0.0

        results["dl"]["f1"].append(f1_u)
        results["dl"]["mae_pos"].append(np.mean(dl_abs_errors) if len(dl_abs_errors) > 0 else 0.0)
        results["dl"]["rmse_recon"].append(np.mean(dl_spec_rmse))

        results["fooof"]["f1"].append(f1_f)
        results["fooof"]["mae_pos"].append(np.mean(fooof_abs_errors) if len(fooof_abs_errors) > 0 else 0.0)
        results["fooof"]["rmse_recon"].append(np.nanmean(fooof_spec_rmse))

        print(f"Noise K={k_val:2d} | UNet  -> F1: {f1_u:.3f} | MAE: {results['dl']['mae_pos'][-1]:.3f} Hz | RMSE: {results['dl']['rmse_recon'][-1]:.4f}")
        print(f"Noise K={k_val:2d} | FOOOF -> F1: {f1_f:.3f} | MAE: {results['fooof']['mae_pos'][-1]:.3f} Hz | RMSE: {results['fooof']['rmse_recon'][-1]:.4f}")

    # =====================================================================
    # 4. COMPREHENSIVE FIGURE
    # =====================================================================
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0], hspace=0.32, wspace=0.25)

    f_eval = vis_data["f_eval"]

    # Top Row - Left: Example Reconstruction at Lowest Noise
    ax_top1 = fig.add_subplot(gs[0, :2])
    ax_top1.plot(f_eval, vis_data["noisy_low"], color="silver", lw=1.2, label=f"Noisy Input (Lowest Noise: K={k_levels[0]})")
    ax_top1.plot(f_eval, vis_data["gt_clean"], "k--", lw=2.2, label="Analytical Ground Truth")
    ax_top1.plot(f_eval, vis_data["dl_low"], color="royalblue", lw=1.8, label="UNet Denoised")
    ax_top1.plot(f_eval, vis_data["fooof_low"], color="forestgreen", linestyle="-.", lw=1.6, label="FOOOF Fit")
    ax_top1.set_title(f"Example Reconstruction at Lowest Noise ($K = {k_levels[0]}$ Welch Segments)", fontweight="bold")
    ax_top1.set_xlabel("Frequency (Hz)")
    ax_top1.set_ylabel("Log Power (natural log)")
    ax_top1.grid(True, linestyle=":", alpha=0.5)
    ax_top1.legend(loc="upper right", framealpha=0.9)

    # Top Row - Right: Example Reconstruction at Highest Noise
    ax_top2 = fig.add_subplot(gs[0, 2])
    ax_top2.plot(f_eval, vis_data["noisy_high"], color="silver", lw=1.0, label=f"Noisy (K={k_levels[-1]})")
    ax_top2.plot(f_eval, vis_data["gt_clean"], "k--", lw=1.8, label="Ground Truth")
    ax_top2.plot(f_eval, vis_data["dl_high"], color="royalblue", lw=1.6, label="UNet")
    ax_top2.plot(f_eval, vis_data["fooof_high"], color="forestgreen", linestyle="-.", lw=1.4, label="FOOOF")
    ax_top2.set_title(f"Highest Noise ($K = {k_levels[-1]}$)", fontweight="bold")
    ax_top2.set_xlabel("Frequency (Hz)")
    ax_top2.grid(True, linestyle=":", alpha=0.5)
    ax_top2.legend(loc="upper right", fontsize=8)

    # Bottom Row: Metric Degradation Curves
    x_indices = np.arange(n_levels)
    noise_labels = [f"K={k}\n(L{i+1})" for i, k in enumerate(k_levels)]

    # 1. Peak Detection F1 Score vs Noise
    ax_b1 = fig.add_subplot(gs[1, 0])
    ax_b1.plot(x_indices, results["dl"]["f1"], "o-", color="royalblue", lw=2, ms=6, label="UNet Model")
    ax_b1.plot(x_indices, results["fooof"]["f1"], "s--", color="forestgreen", lw=2, ms=6, label="FOOOF")
    ax_b1.set_xticks(x_indices)
    ax_b1.set_xticklabels(noise_labels)
    ax_b1.set_ylabel("Peak F1 Score")
    ax_b1.set_xlabel("Noise Severity (Decreasing Welch K)")
    ax_b1.set_title("Peak Detection F1 vs. Noise", fontweight="bold")
    ax_b1.grid(True, linestyle=":", alpha=0.6)
    ax_b1.legend(loc="lower left")

    # 2. Peak Center Error MAE vs Noise
    ax_b2 = fig.add_subplot(gs[1, 1])
    ax_b2.plot(x_indices, results["dl"]["mae_pos"], "o-", color="royalblue", lw=2, ms=6, label="UNet (Sub-Bin Refined)")
    ax_b2.plot(x_indices, results["fooof"]["mae_pos"], "s--", color="forestgreen", lw=2, ms=6, label="FOOOF")
    ax_b2.set_xticks(x_indices)
    ax_b2.set_xticklabels(noise_labels)
    ax_b2.set_ylabel("Localization MAE (Hz)")
    ax_b2.set_xlabel("Noise Severity (Decreasing Welch K)")
    ax_b2.set_title("Peak Localization MAE vs. Noise", fontweight="bold")
    ax_b2.grid(True, linestyle=":", alpha=0.6)
    ax_b2.legend(loc="upper left")

    # 3. Continuous PSD Reconstruction RMSE vs Noise
    ax_b3 = fig.add_subplot(gs[1, 2])
    ax_b3.plot(x_indices, results["dl"]["rmse_recon"], "o-", color="royalblue", lw=2, ms=6, label="UNet Model")
    ax_b3.plot(x_indices, results["fooof"]["rmse_recon"], "s--", color="forestgreen", lw=2, ms=6, label="FOOOF")
    ax_b3.set_xticks(x_indices)
    ax_b3.set_xticklabels(noise_labels)
    ax_b3.set_ylabel("Reconstruction RMSE (ln units)")
    ax_b3.set_xlabel("Noise Severity (Decreasing Welch K)")
    ax_b3.set_title("Continuous PSD Reconstruction RMSE", fontweight="bold")
    ax_b3.grid(True, linestyle=":", alpha=0.6)
    ax_b3.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_multinoise_joint_benchmark(
        #model = ResMultiTaskUNet1D(in_channels=1, base_filters=32),
        model = DeformResMultiTaskUNet1D(in_channels=1, base_filters=32),
        #model = TokenResMultiTaskUNet1D(in_channels=1, base_filters=32),
        dataset_path="DL/data/eeg_psd_multinoise_benchmark_foof_advantage.npz",
        #checkpoint_path="DL_checkpoints/Baseline_UNet.pth",
        checkpoint_path="DL_checkpoints/Deformable_UNet.pth",
        #checkpoint_path="DL_checkpoints/TaskToken_UNet.pth",
        batch_size=64,
        center_thresh=0.35,
        max_match_dist=2.0,
        freq_range=[0.5, 45.0],
        fooof_aperiodic="knee",
        fooof_peak_width=[0.8, 8.0],
        fooof_max_peaks=4,
        fooof_min_height=0.25,
        fooof_threshold=2.5,
    )