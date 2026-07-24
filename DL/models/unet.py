import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralUNet1D(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        
        # Encoder (Downsampling)
        self.enc1 = nn.Sequential(nn.Conv1d(in_channels, 32, 15, padding=7), nn.BatchNorm1d(32), nn.GELU())
        self.down1 = nn.Conv1d(32, 64, 4, stride=2, padding=1) # 500 -> 250
        
        self.enc2 = nn.Sequential(nn.Conv1d(64, 64, 15, padding=7), nn.BatchNorm1d(64), nn.GELU())
        self.down2 = nn.Conv1d(64, 128, 4, stride=2, padding=1) # 250 -> 125
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(128, 128, 15, padding=7),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        
        # Decoder (Upsampling with Skip Connections)
        self.up2 = nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1) # 125 -> 250
        self.dec2 = nn.Sequential(nn.Conv1d(128, 64, 15, padding=7), nn.BatchNorm1d(64), nn.GELU())
        
        self.up1 = nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1) # 250 -> 500
        self.dec1 = nn.Sequential(nn.Conv1d(64, 32, 15, padding=7), nn.BatchNorm1d(32), nn.GELU())
        
        self.final_conv = nn.Conv1d(32, in_channels, kernel_size=15, padding=7)

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)
        d1 = self.down1(e1)
        
        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        
        # Bottleneck
        b = self.bottleneck(d2)
        
        # Decode + Concatenate Skip Connections
        u2 = self.up2(b)
        u2 = torch.cat([u2, e2], dim=1) # Concatenate along channels
        dec2_out = self.dec2(u2)
        
        u1 = self.up1(dec2_out)
        u1 = torch.cat([u1, e1], dim=1)
        dec1_out = self.dec1(u1)
        
        return self.final_conv(dec1_out)