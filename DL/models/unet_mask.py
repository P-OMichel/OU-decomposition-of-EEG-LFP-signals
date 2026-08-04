import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=15, padding=7),
            nn.BatchNorm1d(out_c),
            nn.GELU(),
            nn.Conv1d(out_c, out_c, kernel_size=15, padding=7),
            nn.BatchNorm1d(out_c),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class MultiTaskSpectralUNet1D(nn.Module):
    """
    Fully Convolutional Network predicting:
    1. pred_clean: Reconstructed PSD curve (Shape: Batch, 1, 250)
    2. pred_masks: 2 Binary Masks (Shape: Batch, 2, 250)
       - Channel 0: Peak Center Heatmap
       - Channel 1: Interval Range Mask
    """
    def __init__(self, in_channels=1, hidden_dim=64):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock1D(in_channels, hidden_dim)
        self.enc2 = ConvBlock1D(hidden_dim, hidden_dim * 2)

        # Bottleneck
        self.bottleneck = ConvBlock1D(hidden_dim * 2, hidden_dim * 2)

        # Decoder for PSD Denoising Head
        self.dec_clean = nn.Sequential(
            ConvBlock1D(hidden_dim * 2, hidden_dim),
            nn.Conv1d(hidden_dim, in_channels, kernel_size=15, padding=7)
        )

        # Decoder for Mask Prediction Head
        self.dec_masks = nn.Sequential(
            ConvBlock1D(hidden_dim * 2, hidden_dim),
            nn.Conv1d(hidden_dim, 2, kernel_size=15, padding=7),
            nn.Sigmoid()  # Maps mask probabilities strictly to [0.0, 1.0]
        )

    def forward(self, x):
        # Feature Extraction
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        b = self.bottleneck(h2)

        # Branch 1: Global Skip-Connection Denoising
        pred_clean = x + self.dec_clean(b)

        # Branch 2: Mask Predictions (2 Channels)
        pred_masks = self.dec_masks(b)

        return pred_clean, pred_masks



class MultiTaskSpectralUNet1DFusion(nn.Module):
    """
    Fully Convolutional Network with Soft Feature Fusion predicting:
    1. pred_clean: Reconstructed PSD curve (Shape: Batch, 1, 250)
    2. pred_masks: 2 Mask Channels (Shape: Batch, 2, 250)
       - Channel 0: Peak Center Heatmap (Soft-fused)
       - Channel 1: Interval Range Mask (Soft-fused)
    """
    def __init__(self, in_channels=1, hidden_dim=64):
        super().__init__()

        # 1. Encoder & Bottleneck
        self.enc1 = ConvBlock1D(in_channels, hidden_dim)
        self.enc2 = ConvBlock1D(hidden_dim, hidden_dim * 2)
        self.bottleneck = ConvBlock1D(hidden_dim * 2, hidden_dim * 2)

        # 2. Decoder for PSD Denoising Head
        self.dec_clean = nn.Sequential(
            ConvBlock1D(hidden_dim * 2, hidden_dim),
            nn.Conv1d(hidden_dim, in_channels, kernel_size=15, padding=7)
        )

        # 3. Soft Feature Fusion Mask Head
        self.dec_mask_feats = ConvBlock1D(hidden_dim * 2, hidden_dim)

        # Task-specific feature extractors (32 channels each)
        self.ctr_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.GELU()
        )
        self.int_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.GELU()
        )

        # Non-linear joint fusion layer (processes stacked 32 + 32 = 64 feature channels)
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 2, kernel_size=7, padding=3),
            nn.Sigmoid()  # Outputs continuous mask probabilities strictly in [0.0, 1.0]
        )

    def forward(self, x):
        # --- Shared Feature Extraction ---
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        b = self.bottleneck(h2)

        # --- Branch 1: Denoised Curve Reconstruction ---
        pred_clean = x + self.dec_clean(b)

        # --- Branch 2: Soft Feature Fusion Mask Prediction ---
        m_feats = self.dec_mask_feats(b)  # Shape: (Batch, 64, 250)

        # Extract specialized task features
        f_ctr = self.ctr_conv(m_feats)    # Peak shape/curvature features: (Batch, 32, 250)
        f_int = self.int_conv(m_feats)    # Interval power elevation features: (Batch, 32, 250)

        # Concatenate along channel dimension to form joint vector space
        f_combined = torch.cat([f_ctr, f_int], dim=1)  # Shape: (Batch, 64, 250)

        # Bi-directional non-linear fusion -> Outputs 2-channel mask
        pred_masks = self.fusion_conv(f_combined)      # Shape: (Batch, 2, 250)

        return pred_clean, pred_masks