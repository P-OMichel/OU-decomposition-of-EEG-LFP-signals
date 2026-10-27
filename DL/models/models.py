
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

# =====================================================================
# SHARED CONV / BOTTLENECK BLOCKS
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
        self.b0 = nn.Sequential(nn.Conv1d(in_c, branch_c, kernel_size=1), nn.BatchNorm1d(branch_c), nn.GELU())
        self.b1 = nn.Sequential(nn.Conv1d(in_c, branch_c, kernel_size=3, padding=2, dilation=2), nn.BatchNorm1d(branch_c), nn.GELU())
        self.b2 = nn.Sequential(nn.Conv1d(in_c, branch_c, kernel_size=3, padding=4, dilation=4), nn.BatchNorm1d(branch_c), nn.GELU())
        self.b3 = nn.Sequential(nn.Conv1d(in_c, branch_c, kernel_size=3, padding=8, dilation=8), nn.BatchNorm1d(branch_c), nn.GELU())
        self.project = nn.Sequential(nn.Conv1d(branch_c * 4, out_c, kernel_size=1), nn.BatchNorm1d(out_c), nn.GELU())

    def forward(self, x):
        feat = torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return self.project(feat)


# =====================================================================
# 1. ORIGINAL MODEL: ResMultiTaskUNet1D
# =====================================================================
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
# 2. MODEL 2: DEFORMABLE CONV 1D UNET
# =====================================================================
class DeformConv1D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.offset_conv = nn.Conv1d(in_c, 2 * kernel_size, kernel_size=kernel_size, padding=padding)
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)
        self.weight = nn.Parameter(torch.empty(out_c, in_c, 1, kernel_size))
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_c)) if bias else None

    def forward(self, x):
        B, C, L = x.shape
        x_2d = x.unsqueeze(2)
        offsets = self.offset_conv(x).unsqueeze(2)
        offsets[:, 0::2, :, :] = 0.0  # Zero out height offset (strict 1D frequency deformation)
        out_2d = deform_conv2d(x_2d, offsets, self.weight, self.bias, padding=(0, self.padding))
        return out_2d.squeeze(2)


class ResDeformConvBlock1D(nn.Module):
    def __init__(self, in_c, out_c, drop_prob=0.1):
        super().__init__()
        self.conv1 = DeformConv1D(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.act1 = nn.GELU()
        self.drop = nn.Dropout1d(p=drop_prob)
        self.conv2 = DeformConv1D(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.shortcut = nn.Conv1d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()
        self.act2 = nn.GELU()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return self.act2(out + res)


class DeformResMultiTaskUNet1D(nn.Module):
    def __init__(self, in_channels=1, base_filters=32):
        super().__init__()
        self.enc1 = ResDeformConvBlock1D(in_channels, base_filters)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = ResDeformConvBlock1D(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = ResDeformConvBlock1D(base_filters * 2, base_filters * 4)
        self.bottleneck = ResDeformConvBlock1D(base_filters * 4, base_filters * 4)

        self.up2 = nn.ConvTranspose1d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ResDeformConvBlock1D(base_filters * 4, base_filters * 2)
        self.up1 = nn.ConvTranspose1d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ResDeformConvBlock1D(base_filters * 2, base_filters)

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
        e3 = self.bottleneck(e3)

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
# 3. MODEL 3: TASK TOKEN BOTTLENECK + FiLM UNET
# =====================================================================
class TaskTokenBottleneck1D(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=2, drop=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.task_tokens = nn.Parameter(torch.randn(1, 3, embed_dim) * 0.02)
        self.norm_tokens = nn.LayerNorm(embed_dim)
        self.norm_feats = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.norm_inter = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=drop, batch_first=True)

        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(drop)
        )

        self.film_baseline = nn.Linear(embed_dim, 32 * 2)
        self.film_peaks = nn.Linear(embed_dim, 32 * 2)
        self.film_widths = nn.Linear(embed_dim, 32 * 2)

    def forward(self, x):
        B, C, L = x.shape
        x_seq = self.norm_feats(x.permute(0, 2, 1))
        tokens = self.task_tokens.expand(B, -1, -1)

        c_out, _ = self.cross_attn(self.norm_tokens(tokens), x_seq, x_seq)
        tokens = tokens + c_out

        s_out, _ = self.self_attn(self.norm_inter(tokens), self.norm_inter(tokens), self.norm_inter(tokens))
        tokens = tokens + s_out
        tokens = tokens + self.ffn(self.norm_ffn(tokens))

        def make_film(proj, tok):
            p = proj(tok).unsqueeze(-1)
            gamma, beta = p.chunk(2, dim=1)
            return gamma, beta

        film = {
            "baseline": make_film(self.film_baseline, tokens[:, 0, :]),
            "peaks": make_film(self.film_peaks, tokens[:, 1, :]),
            "widths": make_film(self.film_widths, tokens[:, 2, :]),
        }
        return tokens, film


class TokenResMultiTaskUNet1D(nn.Module):
    def __init__(self, in_channels=1, base_filters=32):
        super().__init__()
        self.enc1 = ResConvBlock1D(in_channels, base_filters, use_se=True)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = ResConvBlock1D(base_filters, base_filters * 2, use_se=True)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = ResConvBlock1D(base_filters * 2, base_filters * 4, use_se=True)
        self.aspp = ASPP1D(base_filters * 4, base_filters * 4)

        self.task_bottleneck = TaskTokenBottleneck1D(embed_dim=base_filters * 4, num_heads=4)

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

    def _apply_film(self, feat, film_tuple):
        gamma, beta = film_tuple
        return (1.0 + gamma) * feat + beta

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e3 = self.aspp(e3)

        tokens, film = self.task_bottleneck(e3)

        d2 = self.up2(e3)
        if d2.shape[-1] != e2.shape[-1]:
            d2 = F.interpolate(d2, size=e2.shape[-1], mode="linear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[-1] != e1.shape[-1]:
            d1 = F.interpolate(d1, size=e1.shape[-1], mode="linear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        d1_denoise = self._apply_film(d1, film["baseline"])
        f_denoise = self.denoise_feat(d1_denoise)
        noise_residual = self.denoise_out(f_denoise)
        clean_psd = x - noise_residual

        d1_heat = self._apply_film(d1, film["peaks"])
        f_heat = self.heatmap_feat(torch.cat([d1_heat, f_denoise], dim=1))
        heatmap_logits = self.heatmap_out(f_heat)

        peak_prob = torch.sigmoid(heatmap_logits)
        d1_width = self._apply_film(d1, film["widths"])
        f_width = self.width_feat(torch.cat([d1_width, f_denoise, peak_prob], dim=1))
        widths = self.width_out(f_width)

        return clean_psd, heatmap_logits, widths