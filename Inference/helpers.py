import numpy as np
import torch
import torch.nn.functional as F

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