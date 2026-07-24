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