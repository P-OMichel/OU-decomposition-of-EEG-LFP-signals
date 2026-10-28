"""
Unified head-to-head multi-noise benchmark comparing ResMultiTaskUNet1D against FOOOF
using identical frequency grids.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from fooof import FOOOF

from DL.models.models import (
    ResMultiTaskUNet1D,
    DeformResMultiTaskUNet1D,
    TokenResMultiTaskUNet1D,
)
from Inference.helpers import decode_peak_frequencies_subbin


# =====================================================================
# 1. HELPER FUNCTIONS
# =====================================================================
def match_peaks_and_compute_errors(pred_peaks, gt_peaks, max_dist_hz=2.0):
    pred_peaks = np.asarray(pred_peaks, dtype=np.float64)
    gt_peaks = np.asarray(gt_peaks, dtype=np.float64)

    n_p, n_g = len(pred_peaks), len(gt_peaks)
    if n_p == 0 and n_g == 0:
        return [], [], 0, 0, 0
    if n_p == 0:
        return [], [], 0, 0, n_g
    if n_g == 0:
        return [], [], 0, n_p, 0

    dist_matrix = np.abs(pred_peaks[:, None] - gt_peaks[None, :])
    pred_idx, gt_idx = linear_sum_assignment(dist_matrix)

    abs_errors, signed_errors = [], []
    matched_p, matched_g = set(), set()

    for p, g in zip(pred_idx, gt_idx):
        signed_err = pred_peaks[p] - gt_peaks[g]
        abs_err = abs(signed_err)
        if abs_err <= max_dist_hz:
            abs_errors.append(abs_err)
            signed_errors.append(signed_err)
            matched_p.add(p)
            matched_g.add(g)

    tp = len(matched_p)
    fp = n_p - tp
    fn = n_g - len(matched_g)
    return abs_errors, signed_errors, tp, fp, fn


def plot_head_to_head_error_distributions(
    dl_abs, dl_signed, fooof_abs, fooof_signed, max_dist_hz=2.0, bins=40
):
    dl_abs, dl_signed = np.array(dl_abs), np.array(dl_signed)
    fooof_abs, fooof_signed = np.array(fooof_abs), np.array(fooof_signed)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # UNet Signed
    if len(dl_signed) > 0:
        axes[0, 0].hist(dl_signed, bins=bins, range=(-max_dist_hz, max_dist_hz),
                        color="royalblue", edgecolor="black", alpha=0.7, density=True)
        mean_u, std_u = np.mean(dl_signed), np.std(dl_signed)
        axes[0, 0].axvline(0, color="red", linestyle="--", lw=1.5, label="Zero Bias")
        axes[0, 0].axvline(mean_u, color="darkorange", lw=1.8, label=f"Mean: {mean_u:+.3f} Hz")
        axes[0, 0].set_title(f"UNet Signed Error ($f_{{pred}} - f_{{gt}}$)\nStd: {std_u:.3f} Hz", fontweight="bold")
    axes[0, 0].set_ylabel("Probability Density")
    axes[0, 0].set_xlabel("Frequency Offset (Hz)")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend(loc="upper right")

    # FOOOF Signed
    if len(fooof_signed) > 0:
        axes[0, 1].hist(fooof_signed, bins=bins, range=(-max_dist_hz, max_dist_hz),
                        color="forestgreen", edgecolor="black", alpha=0.7, density=True)
        mean_f, std_f = np.mean(fooof_signed), np.std(fooof_signed)
        axes[0, 1].axvline(0, color="red", linestyle="--", lw=1.5, label="Zero Bias")
        axes[0, 1].axvline(mean_f, color="darkorange", lw=1.8, label=f"Mean: {mean_f:+.3f} Hz")
        axes[0, 1].set_title(f"FOOOF Signed Error ($f_{{pred}} - f_{{gt}}$)\nStd: {std_f:.3f} Hz", fontweight="bold")
    axes[0, 1].set_ylabel("Probability Density")
    axes[0, 1].set_xlabel("Frequency Offset (Hz)")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend(loc="upper right")

    # UNet Absolute
    if len(dl_abs) > 0:
        axes[1, 0].hist(dl_abs, bins=bins, range=(0, max_dist_hz),
                        color="royalblue", edgecolor="black", alpha=0.7, density=True)
        mae_u, med_u = np.mean(dl_abs), np.median(dl_abs)
        axes[1, 0].axvline(mae_u, color="crimson", lw=1.8, label=f"Mean MAE: {mae_u:.3f} Hz")
        axes[1, 0].axvline(med_u, color="gold", linestyle="--", lw=1.8, label=f"Median MAE: {med_u:.3f} Hz")
        axes[1, 0].set_title("UNet Absolute Error Distribution", fontweight="bold")
    axes[1, 0].set_ylabel("Probability Density")
    axes[1, 0].set_xlabel("Absolute Error (Hz)")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend(loc="upper right")

    # FOOOF Absolute
    if len(fooof_abs) > 0:
        axes[1, 1].hist(fooof_abs, bins=bins, range=(0, max_dist_hz),
                        color="forestgreen", edgecolor="black", alpha=0.7, density=True)
        mae_f, med_f = np.mean(fooof_abs), np.median(fooof_abs)
        axes[1, 1].axvline(mae_f, color="crimson", lw=1.8, label=f"Mean MAE: {mae_f:.3f} Hz")
        axes[1, 1].axvline(med_f, color="gold", linestyle="--", lw=1.8, label=f"Median MAE: {med_f:.3f} Hz")
        axes[1, 1].set_title("FOOOF Absolute Error Distribution", fontweight="bold")
    axes[1, 1].set_ylabel("Probability Density")
    axes[1, 1].set_xlabel("Absolute Error (Hz)")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend(loc="upper right")

    plt.suptitle("Peak Localization Error Distributions (UNet vs. FOOOF)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# =====================================================================
# 2. BENCHMARK EXECUTION
# =====================================================================
def run_multinoise_joint_benchmark(
    model,
    dataset_path="eeg_psd_multinoise_benchmark_aligned.npz",
    checkpoint_path="DL_checkpoints/Deformable_UNet(3).pth",
    batch_size=64,
    center_thresh=0.35,
    max_match_dist=2.0,
    eval_freq_range=[0.5, 45.0],
    fooof_aperiodic="knee",
    fooof_peak_width=[0.8, 8.0],
    fooof_max_peaks=4,
    fooof_min_height=0.25,
    fooof_threshold=2.5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Checkpoint: {checkpoint_path}")

    # Load Model
    model.to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"Loaded weights from '{checkpoint_path}'")
    else:
        print(f"Warning: Checkpoint '{checkpoint_path}' not found! Using initialized weights.")
    model.eval()

    # Load Dataset
    data = np.load(dataset_path, allow_pickle=True)
    raw_f = data["f"]
    # Reconstruct uniform float64 linear grid to guarantee exact step size for FOOOF
    f = np.linspace(float(raw_f[0]), float(raw_f[-1]), len(raw_f), dtype=np.float64)

    k_levels = data["noise_levels_k"]
    n_levels = len(k_levels)
    log_noisy_all = data["log_noisy"]  # [N, n_levels, n_freqs]
    log_clean_all = data["log_clean"]  # [N, n_freqs]
    masks_all = data["masks"]          # [N, 2, n_freqs]
    n_samples = len(log_clean_all)

    # Evaluation slice
    eval_mask = (f >= eval_freq_range[0]) & (f <= eval_freq_range[1])
    f_eval = f[eval_mask]

    print(f"Grid: [{f[0]:.2f}, {f[-1]:.2f}] Hz ({len(f)} bins)")
    print(f"Active Evaluation Window: [{eval_freq_range[0]:.2f}, {eval_freq_range[1]:.2f}] Hz")

    results = {
        "dl": {"f1": [], "mae_pos": [], "rmse_recon": []},
        "fooof": {"f1": [], "mae_pos": [], "rmse_recon": []},
    }

    all_dl_abs_errors, all_dl_signed_errors = [], []
    all_fooof_abs_errors, all_fooof_signed_errors = [], []

    sample_vis_idx = 0
    vis_data = {
        "f_eval": f_eval,
        "gt_clean": log_clean_all[sample_vis_idx][eval_mask],
        "noisy_low": log_noisy_all[sample_vis_idx, 0][eval_mask],
        "noisy_high": log_noisy_all[sample_vis_idx, -1][eval_mask],
        "dl_low": None, "dl_high": None, "fooof_low": None, "fooof_high": None,
    }

    for lvl_idx, k_val in enumerate(k_levels):
        print(f"\n--- Evaluating Noise Level {lvl_idx + 1}/{n_levels} (Welch K = {k_val}) ---")

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
        dl_abs_errors, dl_signed_errors = [], []
        fooof_abs_errors, fooof_signed_errors = [], []
        dl_spec_rmse, fooof_spec_rmse = [], []

        # 1. Batched UNet Inference directly on identical grid
        dl_clean_preds = []
        dl_pred_peaks_list = []

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_tensors = (
                torch.from_numpy(log_noisy_all[start_idx:end_idx, lvl_idx])
                .unsqueeze(1)
                .to(device)
            )

            with torch.no_grad():
                c_preds, batch_logits, _ = model(batch_tensors)
                dl_clean_preds.extend(c_preds.squeeze(1).cpu().numpy())

            for b in range(end_idx - start_idx):
                p_f = decode_peak_frequencies_subbin(batch_logits[b, 0], f, center_thresh=center_thresh)
                p_f = p_f[(p_f >= eval_freq_range[0]) & (p_f <= eval_freq_range[1])]
                dl_pred_peaks_list.append(p_f)

        # 2. Iterate Samples
        for i in range(n_samples):
            # Ground truth peaks
            gt_indices = np.where(masks_all[i, 0] > 0.5)[0]
            gt_f = f[gt_indices]
            gt_f = gt_f[(gt_f >= eval_freq_range[0]) & (gt_f <= eval_freq_range[1])]
            target_clean_curve = log_clean_all[i][eval_mask]

            # UNet Evaluation
            pred_f_unet = dl_pred_peaks_list[i]
            u_abs, u_sgn, u_tp, u_fp, u_fn = match_peaks_and_compute_errors(
                pred_f_unet, gt_f, max_dist_hz=max_match_dist
            )
            dl_abs_errors.extend(u_abs)
            dl_signed_errors.extend(u_sgn)
            dl_tp += u_tp
            dl_fp += u_fp
            dl_fn += u_fn

            unet_curve_eval = dl_clean_preds[i][eval_mask]
            dl_spec_rmse.append(np.sqrt(np.mean((unet_curve_eval - target_clean_curve) ** 2)))

            # FOOOF Evaluation
            psd_linear = np.exp(log_noisy_all[i, lvl_idx]).astype(np.float64)
            fm.fit(f, psd_linear, eval_freq_range)

            if fm.has_model and len(fm.peak_params_) > 0:
                pred_f_fooof = fm.peak_params_[:, 0]
                pred_f_fooof = pred_f_fooof[(pred_f_fooof >= eval_freq_range[0]) & (pred_f_fooof <= eval_freq_range[1])]
            else:
                pred_f_fooof = np.array([], dtype=np.float64)

            f_abs, f_sgn, f_tp, f_fp, f_fn = match_peaks_and_compute_errors(
                pred_f_fooof, gt_f, max_dist_hz=max_match_dist
            )
            fooof_abs_errors.extend(f_abs)
            fooof_signed_errors.extend(f_sgn)
            fooof_tp += f_tp
            fooof_fp += f_fp
            fooof_fn += f_fn

            if fm.has_model:
                fooof_curve_ln = fm.fooofed_spectrum_ * np.log(10.0)
                fooof_spec_rmse.append(np.sqrt(np.mean((fooof_curve_ln - target_clean_curve) ** 2)))
            else:
                fooof_curve_ln = log_noisy_all[i, lvl_idx][eval_mask]
                fooof_spec_rmse.append(np.nan)

            if i == sample_vis_idx:
                if lvl_idx == 0:
                    vis_data["dl_low"] = unet_curve_eval
                    vis_data["fooof_low"] = fooof_curve_ln
                elif lvl_idx == n_levels - 1:
                    vis_data["dl_high"] = unet_curve_eval
                    vis_data["fooof_high"] = fooof_curve_ln

        all_dl_abs_errors.extend(dl_abs_errors)
        all_dl_signed_errors.extend(dl_signed_errors)
        all_fooof_abs_errors.extend(fooof_abs_errors)
        all_fooof_signed_errors.extend(fooof_signed_errors)

        # F1 Metrics
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

    # Plot summary figures
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0], hspace=0.32, wspace=0.25)

    ax_top1 = fig.add_subplot(gs[0, :2])
    ax_top1.plot(vis_data["f_eval"], vis_data["noisy_low"], color="silver", lw=1.2, label=f"Noisy Input (K={k_levels[0]})")
    ax_top1.plot(vis_data["f_eval"], vis_data["gt_clean"], "k--", lw=2.2, label="Analytical Ground Truth")
    ax_top1.plot(vis_data["f_eval"], vis_data["dl_low"], color="royalblue", lw=1.8, label="UNet Denoised")
    ax_top1.plot(vis_data["f_eval"], vis_data["fooof_low"], color="forestgreen", linestyle="-.", lw=1.6, label="FOOOF Fit")
    ax_top1.set_title(f"Reconstruction at Lowest Noise ($K = {k_levels[0]}$)", fontweight="bold")
    ax_top1.set_xlabel("Frequency (Hz)")
    ax_top1.set_ylabel("Log Power")
    ax_top1.grid(True, linestyle=":", alpha=0.5)
    ax_top1.legend(loc="upper right")

    ax_top2 = fig.add_subplot(gs[0, 2])
    ax_top2.plot(vis_data["f_eval"], vis_data["noisy_high"], color="silver", lw=1.0, label=f"Noisy (K={k_levels[-1]})")
    ax_top2.plot(vis_data["f_eval"], vis_data["gt_clean"], "k--", lw=1.8, label="Ground Truth")
    ax_top2.plot(vis_data["f_eval"], vis_data["dl_high"], color="royalblue", lw=1.6, label="UNet")
    ax_top2.plot(vis_data["f_eval"], vis_data["fooof_high"], color="forestgreen", linestyle="-.", lw=1.4, label="FOOOF")
    ax_top2.set_title(f"Highest Noise ($K = {k_levels[-1]}$)", fontweight="bold")
    ax_top2.set_xlabel("Frequency (Hz)")
    ax_top2.grid(True, linestyle=":", alpha=0.5)
    ax_top2.legend(loc="upper right", fontsize=8)

    x_indices = np.arange(n_levels)
    noise_labels = [f"K={k}\n(L{i+1})" for i, k in enumerate(k_levels)]

    ax_b1 = fig.add_subplot(gs[1, 0])
    ax_b1.plot(x_indices, results["dl"]["f1"], "o-", color="royalblue", lw=2, ms=6, label="UNet")
    ax_b1.plot(x_indices, results["fooof"]["f1"], "s--", color="forestgreen", lw=2, ms=6, label="FOOOF")
    ax_b1.set_xticks(x_indices)
    ax_b1.set_xticklabels(noise_labels)
    ax_b1.set_ylabel("Peak F1 Score")
    ax_b1.set_title("Peak Detection F1 vs. Noise", fontweight="bold")
    ax_b1.grid(True, linestyle=":", alpha=0.6)
    ax_b1.legend()

    ax_b2 = fig.add_subplot(gs[1, 1])
    ax_b2.plot(x_indices, results["dl"]["mae_pos"], "o-", color="royalblue", lw=2, ms=6, label="UNet")
    ax_b2.plot(x_indices, results["fooof"]["mae_pos"], "s--", color="forestgreen", lw=2, ms=6, label="FOOOF")
    ax_b2.set_xticks(x_indices)
    ax_b2.set_xticklabels(noise_labels)
    ax_b2.set_ylabel("Localization MAE (Hz)")
    ax_b2.set_title("Peak Localization MAE vs. Noise", fontweight="bold")
    ax_b2.grid(True, linestyle=":", alpha=0.6)
    ax_b2.legend()

    ax_b3 = fig.add_subplot(gs[1, 2])
    ax_b3.plot(x_indices, results["dl"]["rmse_recon"], "o-", color="royalblue", lw=2, ms=6, label="UNet")
    ax_b3.plot(x_indices, results["fooof"]["rmse_recon"], "s--", color="forestgreen", lw=2, ms=6, label="FOOOF")
    ax_b3.set_xticks(x_indices)
    ax_b3.set_xticklabels(noise_labels)
    ax_b3.set_ylabel("Reconstruction RMSE (ln units)")
    ax_b3.set_title("Continuous PSD Reconstruction RMSE", fontweight="bold")
    ax_b3.grid(True, linestyle=":", alpha=0.6)
    ax_b3.legend()

    plt.tight_layout()
    plt.show()

    plot_head_to_head_error_distributions(
        all_dl_abs_errors, all_dl_signed_errors,
        all_fooof_abs_errors, all_fooof_signed_errors,
        max_dist_hz=max_match_dist, bins=40,
    )


if __name__ == "__main__":
    unet_model = DeformResMultiTaskUNet1D(in_channels=1, base_filters=32)

    run_multinoise_joint_benchmark(
        model=unet_model,
        dataset_path="eeg_psd_multinoise_benchmark_aligned.npz",
        checkpoint_path="DL_checkpoints/Deformable_UNet(3).pth",
        batch_size=64,
        center_thresh=0.35,
        max_match_dist=2.0,
        eval_freq_range=[0.5, 45.0],
        fooof_aperiodic="knee",
        fooof_peak_width=[0.8, 8.0],
        fooof_max_peaks=4,
        fooof_min_height=0.25,
        fooof_threshold=2.5,
    )