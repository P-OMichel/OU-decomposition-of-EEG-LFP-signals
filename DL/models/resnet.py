import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(channels)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class SpectralResNet1D(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64, num_blocks=4):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=15, padding=7),
            nn.GELU()
        )
        self.res_blocks = nn.ModuleList([ResBlock1D(hidden_dim) for _ in range(num_blocks)])
        self.out_proj = nn.Conv1d(hidden_dim, in_channels, kernel_size=15, padding=7)

    def forward(self, x):
        # Global Skip Connection: Output = Input + Residual Correction
        h = self.in_proj(x)
        for block in self.res_blocks:
            h = block(h)
        residual_correction = self.out_proj(h)
        return x + residual_correction



# network to output fit, number of peaks and freqeuncy intervals
class MultiTaskSpectralResNet1D(nn.Module):
    """
    1D ResNet performing joint:
    1. PSD Denoising (Reconstruction)
    2. Peak Interval & Frequency Location Extraction (Normalized [0, 1])
    3. Peak Count Estimation
    """
    def __init__(self, in_channels=1, hidden_dim=64, num_blocks=4, max_peaks=4, f_max=50.0):
        super().__init__()
        
        # 1. Feature Extractor Backbone
        self.in_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=15, padding=7),
            nn.GELU()
        )
        self.res_blocks = nn.ModuleList([ResBlock1D(hidden_dim) for _ in range(num_blocks)])
        
        # 2. Denoising Head (Global Residual Reconstruction)
        self.recon_head = nn.Conv1d(hidden_dim, in_channels, kernel_size=15, padding=7)
        
        # 3. Peak & Interval Prediction Head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.interval_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, max_peaks * 3),
            nn.Sigmoid()  # Outputs normalized coordinates strictly in [0, 1]
        )
        
        # 4. Peak Count Head
        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        self.max_peaks = max_peaks
        self.f_max = f_max

    def forward(self, x):
        h = self.in_proj(x)
        for block in self.res_blocks:
            h = block(h)
            
        # Denoised Reconstruction Output
        residual_correction = self.recon_head(h)
        pred_clean = x + residual_correction
        
        # Global Feature Pooling
        feat = self.global_pool(h).squeeze(-1)
        
        # Predict Interval Parameters: (Batch, max_peaks, 3) in normalized range [0.0, 1.0]
        # REMOVED `* self.f_max` HERE so loss evaluates on normalized targets
        pred_intervals = self.interval_head(feat).view(-1, self.max_peaks, 3)
        
        # Predict Peak Count
        pred_count = self.count_head(feat).squeeze(-1)
        
        return pred_clean, pred_intervals, pred_count