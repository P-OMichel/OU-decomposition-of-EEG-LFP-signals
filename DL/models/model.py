import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 2. MODEL ARCHITECTURE (MULTI-TASK 1D U-NET)
# =====================================================================

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )
    def forward(self, x):
        return self.block(x)

class MultiTaskUNet1D(nn.Module):
    def __init__(self, in_channels=1, base_filters=32):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock1D(in_channels, base_filters)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = ConvBlock1D(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = ConvBlock1D(base_filters * 2, base_filters * 4)
        
        # Shared Decoder
        self.up2 = nn.ConvTranspose1d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock1D(base_filters * 4, base_filters * 2)
        
        self.up1 = nn.ConvTranspose1d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ConvBlock1D(base_filters * 2, base_filters)
        
        # --- Head 1: Denoising Head (Clean Curve Reconstruction) ---
        self.head_denoise = nn.Sequential(
            nn.Conv1d(base_filters, base_filters // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(base_filters // 2, 1, kernel_size=1)
        )
        
        # --- Head 2: Peak Center Heatmap Head ---
        self.head_heatmap = nn.Sequential(
            nn.Conv1d(base_filters, base_filters // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(base_filters // 2, 1, kernel_size=1)
        )
        
        # --- Head 3: Interval / Bandwidth Width Head ---
        self.head_width = nn.Sequential(
            nn.Conv1d(base_filters, base_filters // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(base_filters // 2, 2, kernel_size=1), # 2 channels: [left_width, right_width]
            nn.ReLU()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        d2 = self.up2(e3)
        # Handle odd length padding mismatches from transposed convolution
        if d2.shape[-1] != e2.shape[-1]:
            d2 = F.interpolate(d2, size=e2.shape[-1], mode='linear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape[-1] != e1.shape[-1]:
            d1 = F.interpolate(d1, size=e1.shape[-1], mode='linear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        clean_psd     = self.head_denoise(d1)
        heatmap_logits = self.head_heatmap(d1)
        widths         = self.head_width(d1)
        
        return clean_psd, heatmap_logits, widths