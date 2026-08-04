import torch
import torch.nn as nn
import torch.nn.functional as F


class LossMSEGradMSE(nn.Module):

    def __init__(self, lambda_grad=0.5):
        super().__init__()
        self.lambda_grad = lambda_grad

    def forward(self, pred_log_psd, target_log_psd):
        # 1. Scale-invariant Pointwise Loss in Log Space
        # Smooth L1 (Huber) or MSE handles global shape & percentage errors
        loss_pointwise = F.mse_loss(pred_log_psd, target_log_psd)

        # 2. Spectral Derivative Loss (First difference along frequency axis)
        # Forces exact peak locations (f0) and sharpness without smoothing
        pred_grad = pred_log_psd[:, :, 1:] - pred_log_psd[:, :, :-1]
        target_grad = target_log_psd[:, :, 1:] - target_log_psd[:, :, :-1]
        loss_grad = F.mse_loss(pred_grad, target_grad)

        return loss_pointwise + self.lambda_grad * loss_grad

class LossMSEGradMSELaplacienMSE(nn.Module):
    def __init__(self, lambda_grad=0.5, lambda_curvature=0.1):
        super().__init__()
        self.lambda_grad = lambda_grad
        self.lambda_curvature = lambda_curvature

    def forward(self, pred_log_psd, target_log_psd):
        # 1. Pointwise MSE (reconstruction fidelity)
        loss_pointwise = F.mse_loss(pred_log_psd, target_log_psd)

        # 2. First-Derivative Loss (matches target slope & peak alignment)
        pred_grad = pred_log_psd[:, :, 1:] - pred_log_psd[:, :, :-1]
        target_grad = target_log_psd[:, :, 1:] - target_log_psd[:, :, :-1]
        loss_grad = F.mse_loss(pred_grad, target_grad)

        # 3. Second-Derivative Loss (Curvature Penalty)
        # Discrete 2nd derivative: f''(x) ≈ f(x+1) - 2f(x) + f(x-1)
        pred_curv = pred_log_psd[:, :, 2:] - 2 * pred_log_psd[:, :, 1:-1] + pred_log_psd[:, :, :-2]
        
        # Unsupervised smoothness constraint: penalize overall curvature magnitude on predictions
        loss_curvature = torch.mean(pred_curv ** 2)

        total_loss = (
            loss_pointwise 
            + self.lambda_grad * loss_grad 
            + self.lambda_curvature * loss_curvature
        )
        return total_loss


class LossMSEGradSpectral(nn.Module):
    def __init__(self, lambda_grad=0.5, lambda_spectral=0.1, cutoff_ratio=0.25):
        """
        Args:
            lambda_grad (float): Weight for 1st-derivative alignment.
            lambda_spectral (float): Weight for high-frequency noise penalty.
            cutoff_ratio (float): Fraction of upper frequencies considered noise (0.0 to 1.0).
        """
        super().__init__()
        self.lambda_grad = lambda_grad
        self.lambda_spectral = lambda_spectral
        self.cutoff_ratio = cutoff_ratio

    def forward(self, pred_log_psd, target_log_psd):
        # 1. Pointwise Loss (Global Shape Fidelity)
        loss_pointwise = F.mse_loss(pred_log_psd, target_log_psd)

        # 2. First-Derivative Loss (Peak Alignment & Slope Matching)
        pred_grad = pred_log_psd[:, :, 1:] - pred_log_psd[:, :, :-1]
        target_grad = target_log_psd[:, :, 1:] - target_log_psd[:, :, :-1]
        loss_grad = F.mse_loss(pred_grad, target_grad)

        # 3. High-Frequency Spectral Penalty (Explicit Denoising)
        # Compute 1D Real FFT along the frequency dimension (last axis)
        fft_pred = torch.fft.rfft(pred_log_psd, dim=-1)
        magnitude_pred = torch.abs(fft_pred)

        # Determine cutoff index for high frequencies
        num_freq_bins = magnitude_pred.shape[-1]
        cutoff_idx = int(num_freq_bins * (1.0 - self.cutoff_ratio))

        # Penalize energy in the upper frequency band of the prediction
        high_freq_energy = magnitude_pred[:, :, cutoff_idx:]
        loss_spectral = torch.mean(high_freq_energy ** 2)

        # Combined Total Loss
        total_loss = (
            loss_pointwise 
            + self.lambda_grad * loss_grad 
            + self.lambda_spectral * loss_spectral
        )
        return total_loss


class LossMSEGradMSELogRatio(nn.Module):
    def __init__(self, lambda_grad=0.5, lambda_ratio=0.1):
        super().__init__()
        self.lambda_grad = lambda_grad
        self.lambda_ratio = lambda_ratio

    def forward(self, pred, target):
        # 1. Base MSE Loss
        loss_mse = F.mse_loss(pred, target)
        
        # 2. Spectral Gradient Loss (First derivative along frequency axis)
        pred_grad = pred[:, :, 1:] - pred[:, :, :-1]
        target_grad = target[:, :, 1:] - target[:, :, :-1]
        loss_grad = F.mse_loss(pred_grad, target_grad)
        
        # 3. Log-Power Ratio / Scale Loss (Penalizes relative deviation)
        loss_ratio = torch.mean(torch.abs(torch.exp(pred) - torch.exp(target)) / (torch.exp(target) + 1e-6))
        
        return loss_mse + self.lambda_grad * loss_grad + self.lambda_ratio * loss_ratio



# Add loss on peaks number and frequency intervals of peaks
class MultiTaskPSDIntervalLoss(nn.Module):
    def __init__(
        self,
        lambda_grad=0.5,
        lambda_curvature=0.1,
        lambda_interval=1.0,
        lambda_count=0.5,
        lambda_order=0.1,  # Weight for boundary order constraint
    ):
        super().__init__()
        self.lambda_grad = lambda_grad
        self.lambda_curvature = lambda_curvature
        self.lambda_interval = lambda_interval
        self.lambda_count = lambda_count
        self.lambda_order = lambda_order

    def forward(
        self,
        pred_clean,
        target_clean,
        pred_intervals,
        target_intervals,
        target_masks,
        pred_count,
        target_count,
    ):
        # =================================================================
        # 1. CURVE RECONSTRUCTION LOSS
        # =================================================================
        loss_pointwise = F.mse_loss(pred_clean, target_clean)

        # First Derivative Loss (Slope matching)
        pred_grad = pred_clean[:, :, 1:] - pred_clean[:, :, :-1]
        target_grad = target_clean[:, :, 1:] - target_clean[:, :, :-1]
        loss_grad = F.mse_loss(pred_grad, target_grad)

        # Second Derivative Loss (Curvature penalty)
        pred_curv = pred_clean[:, :, 2:] - 2 * pred_clean[:, :, 1:-1] + pred_clean[:, :, :-2]
        loss_curv = torch.mean(pred_curv ** 2)

        loss_curve = loss_pointwise + self.lambda_grad * loss_grad + self.lambda_curvature * loss_curv

        # =================================================================
        # 2. PEAK FREQUENCY & INTERVAL EDGE LOSS (MASKED MSE)
        # =================================================================
        # target_masks shape: (Batch, max_peaks) -> expand to (Batch, max_peaks, 3)
        mask_expanded = target_masks.unsqueeze(-1).expand_as(target_intervals)

        # Distance error on normalized [f0, f_left, f_right]
        interval_diff = (pred_intervals - target_intervals) ** 2
        loss_interval = torch.sum(interval_diff * mask_expanded) / (torch.sum(mask_expanded) * 3.0 + 1e-6)

        # -----------------------------------------------------------------
        # ADDITION: INTERVAL ORDERING CONSTRAINT (f_left <= f0 <= f_right)
        # -----------------------------------------------------------------
        pred_f0 = pred_intervals[:, :, 0]
        pred_l = pred_intervals[:, :, 1]
        pred_r = pred_intervals[:, :, 2]

        # Penalize if f_left > f0 or f0 > f_right
        er_left = F.relu(pred_l - pred_f0) ** 2
        er_right = F.relu(pred_f0 - pred_r) ** 2
        loss_order = torch.sum((er_left + er_right) * target_masks) / (torch.sum(target_masks) + 1e-6)

        # =================================================================
        # 3. PEAK COUNT LOSS
        # =================================================================
        loss_count = F.smooth_l1_loss(pred_count, target_count)

        # =================================================================
        # TOTAL COMPOSITE LOSS
        # =================================================================
        total_loss = (
            loss_curve
            + self.lambda_interval * loss_interval
            + self.lambda_order * loss_order
            + self.lambda_count * loss_count
        )

        return total_loss, loss_curve, loss_interval, loss_count


# ===========================================================================
# Losses for mask model
# ===========================================================================
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # BCE Loss
        bce = F.binary_cross_entropy(pred, target)

        # Dice Loss
        # Use .reshape(-1) instead of .view(-1) to handle non-contiguous sliced tensors
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth))

        return bce + dice


class MaskMultiTaskLoss(nn.Module):
    def __init__(self, lambda_grad=0.5, lambda_curvature=0.1, lambda_mask=1.0):
        super().__init__()
        self.lambda_grad = lambda_grad
        self.lambda_curvature = lambda_curvature
        self.lambda_mask = lambda_mask
        self.dice_bce = DiceBCELoss()

    def forward(self, pred_clean, target_clean, pred_masks, target_masks):
        # 1. Denoising Curve Loss
        loss_pointwise = F.mse_loss(pred_clean, target_clean)
        
        pred_grad = pred_clean[:, :, 1:] - pred_clean[:, :, :-1]
        target_grad = target_clean[:, :, 1:] - target_clean[:, :, :-1]
        loss_grad = F.mse_loss(pred_grad, target_grad)

        pred_curv = pred_clean[:, :, 2:] - 2 * pred_clean[:, :, 1:-1] + pred_clean[:, :, :-2]
        loss_curv = torch.mean(pred_curv ** 2)

        #print(f'loss_pointwise: {loss_pointwise} | loss_grad: {loss_grad} | loss_curv: {loss_curv}')
        loss_curve = loss_pointwise + self.lambda_grad * loss_grad + self.lambda_curvature * loss_curv

        # 2. Mask Loss (Channel 0 = Centers, Channel 1 = Intervals)
        loss_mask_centers = self.dice_bce(pred_masks[:, 0, :], target_masks[:, 0, :])
        loss_mask_intervals = self.dice_bce(pred_masks[:, 1, :], target_masks[:, 1, :])
        loss_mask = loss_mask_centers + loss_mask_intervals
        #print(f'loss_mask_centers: {loss_mask_centers} | loss_mask_intervals: {loss_mask_intervals}')

        # Total Loss
        total_loss = loss_curve + self.lambda_mask * loss_mask
        return total_loss, loss_curve, loss_mask