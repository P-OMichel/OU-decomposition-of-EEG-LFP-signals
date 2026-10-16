"""
Unified head-to-head benchmark comparing ResMultiTaskUNet1D against FOOOF
on:
  1. Peak localization (Absolute Error) and count estimation
  2. Full spectral reconstruction error against ground-truth clean PSD (log_clean)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from fooof import FOOOF


# =====================================================================
# 1. MODEL ARCHITECTURE
# =====================================================================
class SqueezeExcitation1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid_c = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, mid_c, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(mid_c, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class ResConvBlock1D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, drop_prob=0.1, use_se=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_c),
            nn.GELU(),
            nn.Dropout1d(p=drop_prob),
            nn.Conv1d(out_c, out_c, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_c),
        )
        self.se = SqueezeExcitation1D(out_c) if use_se else nn.Identity()
        self.shortcut = nn.Conv1d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        out = self.se(self.block(x))
        return self.act(out + self.shortcut(x))


class ASPP1D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        branch_c = out_c // 4
        self.b0 = nn.Sequential(nn.Conv1d(in_c, branch_c, 1), nn.BatchNorm1d(branch_c), nn.GELU())
        self.b1 = nn.Sequential(nn.Conv1d(in_c, branch_c, 3, padding=2, dilation=2), nn.BatchNorm1d(branch_c), nn.GELU())
        self.b2 = nn.Sequential(nn.Conv1d(in_c, branch_c, 3, padding=4, dilation=4), nn.BatchNorm1d(branch_c), nn.GELU())
        self.b3 = nn.Sequential(nn.Conv1d(in_c, branch_c, 3, padding=8, dilation=8), nn.BatchNorm1d(branch_c), nn.GELU())
        self.project = nn.Sequential(nn.Conv1d(branch_c * 4, out_c, 1), nn.BatchNorm1d(out_c), nn.GELU())

    def forward(self, x):
        feat = torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return self.project(feat)


class ResMultiTaskUNet1D(nn.Module):
    def __init__(self, in_channels=1, base_filters=32):
        super().__init__()
        self.enc1 = ResConvBlock1D(in_channels, base_filters, use_se=True)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = ResConvBlock1D(base_filters, base_filters * 2, use_se=True)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = ResConvBlock1D(base_filters * 2, base_filters * 4, use_se=True)
        self.aspp = ASPP1D(base_filters * 4, base_filters * 4)

        self.up2 = nn.ConvTranspose1d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ResConvBlock1D(base_filters * 4, base_filters * 2, use_se=True)
        self.up1 = nn.ConvTranspose1d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ResConvBlock1D(base_filters * 2, base_filters, use_se=True)

        head_mid_c = base_filters // 2
        self.denoise_feat = nn.Sequential(
            nn.Conv1d(base_filters, head_mid_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(head_mid_c),
            nn.GELU(),
            nn.Dropout1d(p=0.1),
        )
        self.denoise_out = nn.Conv1d(head_mid_c, 1, kernel_size=1)

        self.heatmap_feat = nn.Sequential(
            nn.Conv1d(base_filters + head_mid_c, head_mid_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(head_mid_c),
            nn.GELU(),
            nn.Dropout1d(p=0.1),
        )
        self.heatmap_out = nn.Conv1d(head_mid_c, 1, kernel_size=1)

        self.width_feat = nn.Sequential(
            nn.Conv1d(base_filters + head_mid_c + 1, head_mid_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(head_mid_c),
            nn.GELU(),
            nn.Dropout1d(p=0.1),
        )
        self.width_out = nn.Sequential(nn.Conv1d(head_mid_c, 2, kernel_size=1), nn.Softplus())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e3 = self.aspp(e3)

        d2 = self.up2(e3)
        if d2.shape[-1] != e2.shape[-1]:
            d2 = F.interpolate(d2, size=e2.shape[-1], mode="linear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[-1] != e1.shape[-1]:
            d1 = F.interpolate(d1, size=e1.shape[-1], mode="linear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        f_denoise = self.denoise_feat(d1)
        noise_residual = self.denoise_out(f_denoise)
        clean_psd = x - noise_residual

        f_heat = self.heatmap_feat(torch.cat([d1, f_denoise], dim=1))
        heatmap_logits = self.heatmap_out(f_heat)

        peak_prob = torch.sigmoid(heatmap_logits)
        f_width = self.width_feat(torch.cat([d1, f_denoise, peak_prob], dim=1))
        widths = self.width_out(f_width)

        return clean_psd, heatmap_logits, widths


# =====================================================================
# 2. HELPER FUNCTIONS
# =====================================================================
def decode_peak_frequencies(pred_logits, f_grid, center_thresh=0.35, kernel_size=5):
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
        return np.array([])
    if peak_indices.ndim == 0:
        peak_indices = peak_indices.unsqueeze(0)

    return f_grid[peak_indices.detach().cpu().numpy()]


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
# 3. JOINT BENCHMARK LOOP
# =====================================================================
def run_joint_benchmark(
    dataset_path,
    checkpoint_path,
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

    # Load neural network
    model = ResMultiTaskUNet1D(in_channels=1, base_filters=32).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # Load dataset
    data = np.load(dataset_path, allow_pickle=True)
    f = data["f"].astype(np.float64)
    log_noisy = data["log_noisy"]
    log_clean = data["log_clean"]   # Ground-truth noiseless PSD
    masks = data["masks"]

    f_fooof = np.linspace(f[0], f[-1], len(f), dtype=np.float64)
    n_samples = len(log_noisy)

    if freq_range is None:
        freq_range = [float(f_fooof[0]), float(f_fooof[-1])]

    # Mask for frequency region of interest
    freq_mask = (f_fooof >= freq_range[0]) & (f_fooof <= freq_range[1])

    print(f"Comparing UNet Model and FOOOF on {n_samples} samples against ground-truth clean PSD...")

    fm = FOOOF(
        aperiodic_mode=fooof_aperiodic,
        peak_width_limits=fooof_peak_width,
        max_n_peaks=fooof_max_peaks,
        min_peak_height=fooof_min_height,
        peak_threshold=fooof_threshold,
        verbose=False,
    )

    # Peak localization accumulators
    unet_abs_errors, unet_count_errs = [], []
    unet_tp, unet_fp, unet_fn = 0, 0, 0

    fooof_abs_errors, fooof_count_errs = [], []
    fooof_tp, fooof_fp, fooof_fn = 0, 0, 0

    # Reconstruction error accumulators (Curve vs log_clean)
    unet_spec_mae, unet_spec_rmse = [], []
    fooof_spec_mae, fooof_spec_rmse = [], []

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_inputs = torch.tensor(log_noisy[start_idx:end_idx], dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            clean_preds, batch_logits, _ = model(batch_inputs)
            clean_preds = clean_preds.squeeze(1).cpu().numpy()

        for i in range(end_idx - start_idx):
            sample_idx = start_idx + i

            # ---------------- Target Ground Truth ----------------
            gt_indices = np.where(masks[sample_idx, 0] > 0.5)[0]
            gt_f = f_fooof[gt_indices]
            gt_f = gt_f[(gt_f >= freq_range[0]) & (gt_f <= freq_range[1])]

            # True clean target spectrum (sliced to freq_range)
            target_clean_curve = log_clean[sample_idx][freq_mask]

            # ---------------- 1. UNet Prediction ----------------
            pred_f_unet = decode_peak_frequencies(batch_logits[i, 0], f_fooof, center_thresh=center_thresh)
            pred_f_unet = pred_f_unet[(pred_f_unet >= freq_range[0]) & (pred_f_unet <= freq_range[1])]

            u_errs, u_tp, u_fp, u_fn = match_peaks_and_compute_errors(pred_f_unet, gt_f, max_dist_hz=max_match_dist)
            unet_abs_errors.extend(u_errs)
            unet_count_errs.append(len(pred_f_unet) - len(gt_f))
            unet_tp += u_tp
            unet_fp += u_fp
            unet_fn += u_fn

            # UNet spectral curve reconstruction error
            unet_curve = clean_preds[i][freq_mask]
            unet_spec_mae.append(np.mean(np.abs(unet_curve - target_clean_curve)))
            unet_spec_rmse.append(np.sqrt(np.mean((unet_curve - target_clean_curve) ** 2)))

            # ---------------- 2. FOOOF Prediction ----------------
            psd_linear = np.exp(log_noisy[sample_idx]).astype(np.float64)
            fm.fit(f_fooof, psd_linear, freq_range)

            if fm.has_model and len(fm.peak_params_) > 0:
                pred_f_fooof = fm.peak_params_[:, 0]
            else:
                pred_f_fooof = np.array([])

            f_errs, f_tp, f_fp, f_fn = match_peaks_and_compute_errors(pred_f_fooof, gt_f, max_dist_hz=max_match_dist)
            fooof_abs_errors.extend(f_errs)
            fooof_count_errs.append(len(pred_f_fooof) - len(gt_f))
            fooof_tp += f_tp
            fooof_fp += f_fp
            fooof_fn += f_fn

            # FOOOF spectral curve reconstruction error
            if fm.has_model:
                # fm.fooofed_spectrum_ is in log10 space -> convert to natural log
                fooof_curve_ln = fm.fooofed_spectrum_ * np.log(10.0)
                fooof_spec_mae.append(np.mean(np.abs(fooof_curve_ln - target_clean_curve)))
                fooof_spec_rmse.append(np.sqrt(np.mean((fooof_curve_ln - target_clean_curve) ** 2)))
            else:
                fooof_spec_mae.append(np.nan)
                fooof_spec_rmse.append(np.nan)

        print(f"Processed samples: {end_idx}/{n_samples}", end="\r")

    print("\nProcessing complete.")

    # Peak metrics
    unet_abs_errors = np.array(unet_abs_errors)
    fooof_abs_errors = np.array(fooof_abs_errors)
    unet_count_errs = np.array(unet_count_errs)
    fooof_count_errs = np.array(fooof_count_errs)

    prec_u = unet_tp / (unet_tp + unet_fp) if (unet_tp + unet_fp) > 0 else 0.0
    rec_u = unet_tp / (unet_tp + unet_fn) if (unet_tp + unet_fn) > 0 else 0.0
    f1_u = 2 * prec_u * rec_u / (prec_u + rec_u) if (prec_u + rec_u) > 0 else 0.0

    prec_f = fooof_tp / (fooof_tp + fooof_fp) if (fooof_tp + fooof_fp) > 0 else 0.0
    rec_f = fooof_tp / (fooof_tp + fooof_fn) if (fooof_tp + fooof_fn) > 0 else 0.0
    f1_f = 2 * prec_f * rec_f / (prec_f + rec_f) if (prec_f + rec_f) > 0 else 0.0

    # Reconstruction metrics across all samples
    u_mean_mae = np.mean(unet_spec_mae)
    u_mean_rmse = np.mean(unet_spec_rmse)
    f_mean_mae = np.nanmean(fooof_spec_mae)
    f_mean_rmse = np.nanmean(fooof_spec_rmse)

    # =====================================================================
    # 4. PRINT COMPARISON REPORT
    # =====================================================================
    print("\n" + "=" * 70)
    print(f"{'EVALUATION METRIC':<32} | {'UNet MODEL':<16} | {'FOOOF':<16}")
    print("=" * 70)
    print("--- 1. PEAK DETECTION & COUNT ---")
    print(f"{'Precision':<32} | {prec_u:<16.4f} | {prec_f:<16.4f}")
    print(f"{'Recall':<32} | {rec_u:<16.4f} | {rec_f:<16.4f}")
    print(f"{'F1 Score':<32} | {f1_u:<16.4f} | {f1_f:<16.4f}")
    print(f"{'Exact Count Match (%)':<32} | {np.mean(unet_count_errs == 0) * 100:<16.2f} | {np.mean(fooof_count_errs == 0) * 100:<16.2f}")
    print(f"{'Peak Center MAE (Hz)':<32} | {np.mean(unet_abs_errors):<16.4f} | {np.mean(fooof_abs_errors):<16.4f}")
    print("--- 2. SPECTRAL CURVE RECONSTRUCTION (vs log_clean) ---")
    print(f"{'Reconstruction MAE (ln units)':<32} | {u_mean_mae:<16.4f} | {f_mean_mae:<16.4f}")
    print(f"{'Reconstruction RMSE (ln units)':<32} | {u_mean_rmse:<16.4f} | {f_mean_rmse:<16.4f}")
    print("=" * 70)

# =====================================================================
    # 5. COMPREHENSIVE VISUALIZATION (PEAKS + RECONSTRUCTION)
    # =====================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # -------------------------------------------------------------
    # ROW 1: PEAK METRICS
    # -------------------------------------------------------------
    # 1. Absolute Localization Error Histogram
    ax = axes[0, 0]
    bins = np.linspace(0, max_match_dist, 30)
    if len(unet_abs_errors) > 0:
        ax.hist(unet_abs_errors, bins=bins, density=True, alpha=0.5, color="royalblue", 
                label=f"UNet (Mean={np.mean(unet_abs_errors):.2f} Hz)")
    if len(fooof_abs_errors) > 0:
        ax.hist(fooof_abs_errors, bins=bins, density=True, alpha=0.5, color="forestgreen", 
                label=f"FOOOF (Mean={np.mean(fooof_abs_errors):.2f} Hz)")
    ax.set_title("Peak Absolute Error Distribution")
    ax.set_xlabel("Absolute Error $|f_{pred} - f_{gt}|$ (Hz)")
    ax.set_ylabel("Probability Density")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()

    # 2. Cumulative Distribution Function (CDF)
    ax = axes[0, 1]
    if len(unet_abs_errors) > 0:
        s_u = np.sort(unet_abs_errors)
        ax.plot(s_u, np.linspace(0, 1, len(s_u)), color="royalblue", lw=2, 
                label=f"UNet (Median={np.median(unet_abs_errors):.2f} Hz)")
    if len(fooof_abs_errors) > 0:
        s_f = np.sort(fooof_abs_errors)
        ax.plot(s_f, np.linspace(0, 1, len(s_f)), color="forestgreen", lw=2, 
                label=f"FOOOF (Median={np.median(fooof_abs_errors):.2f} Hz)")
    ax.set_title("CDF of Absolute Localization Error")
    ax.set_xlabel("Absolute Error $|f_{pred} - f_{gt}|$ (Hz)")
    ax.set_ylabel("Cumulative Fraction")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")

    # 3. Discrete Peak Count Error (Delta N)
    ax = axes[0, 2]
    all_diffs = np.union1d(np.unique(unet_count_errs), np.unique(fooof_count_errs))
    width = 0.35
    u_counts = [np.mean(unet_count_errs == d) * 100 for d in all_diffs]
    f_counts = [np.mean(fooof_count_errs == d) * 100 for d in all_diffs]

    ax.bar(all_diffs - width / 2, u_counts, width=width, color="royalblue", alpha=0.8, 
           label=f"UNet ({np.mean(unet_count_errs == 0) * 100:.1f}% exact)")
    ax.bar(all_diffs + width / 2, f_counts, width=width, color="forestgreen", alpha=0.8, 
           label=f"FOOOF ({np.mean(fooof_count_errs == 0) * 100:.1f}% exact)")
    ax.set_xticks(all_diffs)
    ax.set_title("Peak Count Difference ($N_{pred} - N_{gt}$)")
    ax.set_xlabel("Count Error $\\Delta N$")
    ax.set_ylabel("Samples (%)")
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax.legend()

    # -------------------------------------------------------------
    # ROW 2: SPECTRAL RECONSTRUCTION (vs log_clean)
    # -------------------------------------------------------------
    # 4. Reconstruction MAE Distribution
    ax = axes[1, 0]
    ax.hist(unet_spec_mae, bins=25, alpha=0.6, color="royalblue", 
            label=f"UNet (Mean={u_mean_mae:.3f})")
    ax.hist(fooof_spec_mae, bins=25, alpha=0.6, color="forestgreen", 
            label=f"FOOOF (Mean={f_mean_mae:.3f})")
    ax.set_title("Spectral Reconstruction MAE vs. Clean")
    ax.set_xlabel("MAE (Natural Log Power)")
    ax.set_ylabel("Sample Count")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()

    # 5. Reconstruction RMSE Distribution
    ax = axes[1, 1]
    ax.hist(unet_spec_rmse, bins=25, alpha=0.6, color="royalblue", 
            label=f"UNet (Mean={u_mean_rmse:.3f})")
    ax.hist(fooof_spec_rmse, bins=25, alpha=0.6, color="forestgreen", 
            label=f"FOOOF (Mean={f_mean_rmse:.3f})")
    ax.set_title("Spectral Reconstruction RMSE vs. Clean")
    ax.set_xlabel("RMSE (Natural Log Power)")
    ax.set_ylabel("Sample Count")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()

    # 6. Single-Sample Spectrum Reconstruction Overlay
    ax = axes[1, 2]
    ax.plot(f_fooof[freq_mask], target_clean_curve, color="black", lw=2, label="Target Clean")
    ax.plot(f_fooof[freq_mask], unet_curve, color="royalblue", linestyle="--", label="UNet Denoised")
    if fm.has_model:
        ax.plot(f_fooof[freq_mask], fooof_curve_ln, color="forestgreen", linestyle=":", label="FOOOF Fit")
    ax.set_title("Single-Sample Fit Comparison")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Natural Log Power")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_joint_benchmark(
        dataset_path="eeg_psd_benchmark_fooof_advantage.npz",
        checkpoint_path="best_keypoint_se_cascade.pth",
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


if __name__ == "__main__":
    run_joint_benchmark(
        dataset_path="eeg_psd_benchmark.npz",
        checkpoint_path="best_keypoint_se_cascade.pth",
        batch_size=64,
        center_thresh=0.35,
        max_match_dist=2.0,
        freq_range=[0.5, 45.0],
        fooof_aperiodic="knee",
        fooof_peak_width=[0.5, 12.0],
        fooof_max_peaks=6,
        fooof_min_height=0.1,
    )
