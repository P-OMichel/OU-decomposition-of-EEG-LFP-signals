import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


# =====================================================================
# 1. MODEL DEFINITIONS (Must match trained architecture)
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
# 2. KEYPOINT NMS DECODER
# =====================================================================
def decode_peak_frequencies(pred_logits, f_grid, center_thresh=0.35, kernel_size=5):
    """Extracts detected peak frequencies using 1D local maxima suppression."""
    # Ensure shape is [1, 1, n_freqs]
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


# =====================================================================
# 3. BIPARTITE MATCHING FOR PEAK POSITION EVALUATION
# =====================================================================
def match_peaks_and_compute_errors(pred_peaks, gt_peaks, max_dist_hz=2.0):
    """
    Pairs predicted and true peaks using Hungarian assignment.
    
    Returns:
        signed_errors: list of (f_pred - f_gt) for true matched pairs
        n_matched: number of matched pairs (TP)
        n_unmatched_pred: predicted peaks outside max_dist_hz radius (FP)
        n_unmatched_gt: true peaks missed (FN)
    """
    n_p = len(pred_peaks)
    n_g = len(gt_peaks)

    if n_p == 0 and n_g == 0:
        return [], 0, 0, 0
    if n_p == 0:
        return [], 0, 0, n_g
    if n_g == 0:
        return [], 0, n_p, 0

    # Pairwise distance matrix [n_pred, n_gt]
    dist_matrix = np.abs(pred_peaks[:, None] - gt_peaks[None, :])
    pred_idx, gt_idx = linear_sum_assignment(dist_matrix)

    signed_errors = []
    matched_p = set()
    matched_g = set()

    for p, g in zip(pred_idx, gt_idx):
        err = pred_peaks[p] - gt_peaks[g]
        if abs(err) <= max_dist_hz:
            signed_errors.append(err)
            matched_p.add(p)
            matched_g.add(g)

    tp = len(matched_p)
    fp = n_p - tp
    fn = n_g - len(matched_g)

    return signed_errors, tp, fp, fn


# =====================================================================
# 4. DATASET EVALUATION LOOP & VISUALIZATION
# =====================================================================
def run_dataset_benchmark(dataset_path, checkpoint_path, batch_size=64, center_thresh=0.35, max_match_dist=2.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ResMultiTaskUNet1D(in_channels=1, base_filters=32).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # Load dataset
    data = np.load(dataset_path, allow_pickle=True)
    f = data["f"]
    log_noisy = data["log_noisy"]
    masks = data["masks"]  # shape: [N, 2, n_freqs], channel 0 = center_mask

    n_samples = len(log_noisy)
    print(f"Evaluating {n_samples} samples on device: {device}")

    all_signed_errors = []
    count_errors = []  # N_pred - N_gt
    n_gt_peaks_all = []
    n_pred_peaks_all = []
    total_tp, total_fp, total_fn = 0, 0, 0

    # Batch processing
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_inputs = torch.tensor(log_noisy[start_idx:end_idx], dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            _, batch_logits, _ = model(batch_inputs)

        for i in range(end_idx - start_idx):
            sample_idx = start_idx + i
            pred_f = decode_peak_frequencies(batch_logits[i, 0], f, center_thresh=center_thresh)

            # Extract ground-truth peaks from mask channel 0
            gt_indices = np.where(masks[sample_idx, 0] > 0.5)[0]
            gt_f = f[gt_indices]

            # Matching and error computation
            errors, tp, fp, fn = match_peaks_and_compute_errors(pred_f, gt_f, max_dist_hz=max_match_dist)

            all_signed_errors.extend(errors)
            count_errors.append(len(pred_f) - len(gt_f))
            n_gt_peaks_all.append(len(gt_f))
            n_pred_peaks_all.append(len(pred_f))

            total_tp += tp
            total_fp += fp
            total_fn += fn

    all_signed_errors = np.array(all_signed_errors)
    count_errors = np.array(count_errors)

    # Summary Statistics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_count_acc = np.mean(count_errors == 0) * 100.0
    mae = np.mean(np.abs(all_signed_errors)) if len(all_signed_errors) > 0 else 0.0
    rmse = np.sqrt(np.mean(all_signed_errors**2)) if len(all_signed_errors) > 0 else 0.0

    print("\n" + "=" * 50)
    print("PEAK POSITION & COUNT ESTIMATION REPORT")
    print("=" * 50)
    print(f"Total Ground-Truth Peaks: {sum(n_gt_peaks_all)}")
    print(f"Total Detected Peaks    : {sum(n_pred_peaks_all)}")
    print(f"Matched True Positives  : {total_tp}")
    print(f"False Positives         : {total_fp}")
    print(f"False Negatives (Missed): {total_fn}")
    print(f"Precision               : {precision:.4f}")
    print(f"Recall                  : {recall:.4f}")
    print(f"F1 Score                : {f1:.4f}")
    print(f"Exact Peak Count Match  : {exact_count_acc:.2f}%")
    print(f"Position Error MAE      : {mae:.4f} Hz")
    print(f"Position Error RMSE     : {rmse:.4f} Hz")
    print("=" * 50)

    # =====================================================================
    # 5. DISTRIBUTION PLOTS
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Plot 1: Signed Frequency Error Histogram
    ax = axes[0]
    ax.hist(all_signed_errors, bins=40, color="royalblue", edgecolor="black", alpha=0.75, density=True)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
    ax.axvline(np.mean(all_signed_errors), color="darkorange", linestyle=":", linewidth=2,
               label=f"Mean: {np.mean(all_signed_errors):.3f} Hz")
    ax.set_title(f"Peak Frequency Error ($\Delta f$)\nMAE = {mae:.3f} Hz, RMSE = {rmse:.3f} Hz")
    ax.set_xlabel("Frequency Error: $f_{pred} - f_{true}$ (Hz)")
    ax.set_ylabel("Probability Density")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

    # Plot 2: Absolute Frequency Error Boxplot / CDF
    ax = axes[1]
    sorted_abs_err = np.sort(np.abs(all_signed_errors))
    cdf = np.arange(1, len(sorted_abs_err) + 1) / len(sorted_abs_err)
    ax.plot(sorted_abs_err, cdf, color="teal", lw=2, label="Cumulative Dist.")
    p50 = np.percentile(sorted_abs_err, 50)
    p95 = np.percentile(sorted_abs_err, 95)
    ax.axvline(p50, color="darkorange", linestyle="--", label=f"Median (50%): {p50:.3f} Hz")
    ax.axvline(p95, color="crimson", linestyle=":", label=f"95th %-ile: {p95:.3f} Hz")
    ax.set_title("CDF of Absolute Localization Error")
    ax.set_xlabel("Absolute Error $|f_{pred} - f_{true}|$ (Hz)")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")

    # Plot 3: Discrete Peak Count Error Distribution
    ax = axes[2]
    unique_errs, counts = np.unique(count_errors, return_counts=True)
    bars = ax.bar(unique_errs, counts / len(count_errors) * 100, color="mediumpurple", edgecolor="black", alpha=0.8)
    ax.set_xticks(unique_errs)
    ax.set_title(f"Peak Count Difference ($N_{{pred}} - N_{{true}}$)\nExact Match: {exact_count_acc:.1f}%")
    ax.set_xlabel("Count Difference $\Delta N$")
    ax.set_ylabel("Samples (%)")
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")

    # Annotate percentage on bars
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_dataset_benchmark(
        dataset_path="eeg_psd_benchmark.npz",
        checkpoint_path="best_keypoint_se_cascade.pth",
        batch_size=64,
        center_thresh=0.35,
        max_match_dist=2.0,  # Max distance in Hz to consider a true positive
    )