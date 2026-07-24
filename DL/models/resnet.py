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

class ResBlock1DReflect(nn.Module):

    def __init__(self, channels, kernel_size=15):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            # Reflect padding mirrors the signal at the 0.1 Hz boundary
            nn.ReflectionPad1d(padding),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=0),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.ReflectionPad1d(padding),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=0),
            nn.BatchNorm1d(channels),
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


class SpectralResNet1DReflect(nn.Module):

    def __init__(
        self, in_channels=1, hidden_dim=64, num_blocks=4, kernel_size=15
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2

        # 1. Input Projection with Reflection Padding
        self.in_proj = nn.Sequential(
            nn.ReflectionPad1d(padding),
            nn.Conv1d(
                in_channels,hidden_dim,kernel_size=kernel_size,padding=0,  # Manual padding via ReflectionPad1d
            ),
            nn.GELU(),
        )

        # 2. Reflection Residual Blocks
        self.res_blocks = nn.ModuleList(
            [ResBlock1DReflect(hidden_dim, kernel_size=kernel_size)for _ in range(num_blocks)]
        )

        # 3. Output Projection with Reflection Padding
        self.out_proj = nn.Sequential(
            nn.ReflectionPad1d(padding),
            nn.Conv1d(
                hidden_dim,in_channels,kernel_size=kernel_size,padding=0,  # Manual padding via ReflectionPad1d
            ),
        )

    def forward(self, x):
        # Global Skip Connection: Output = Input + Residual Correction
        h = self.in_proj(x)
        for block in self.res_blocks:
            h = block(h)
        residual_correction = self.out_proj(h)

        return x + residual_correction