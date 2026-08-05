import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

# Standard default install path for Graphviz on Windows
graphviz_path = r"C:\Program Files\Graphviz\bin"
if os.path.exists(graphviz_path):
    os.environ["PATH"] += os.pathsep + graphviz_path
# =====================================================================
# 1. YOUR MODEL DEFINITIONS
# =====================================================================
class ResConvBlock1D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_c),
            nn.GELU(),
            nn.Conv1d(out_c, out_c, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_c),
        )
        self.shortcut = (
            nn.Conv1d(in_c, out_c, kernel_size=1) 
            if in_c != out_c 
            else nn.Identity()
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.block(x) + self.shortcut(x))


class ImprovedMultiTaskUNet1D(nn.Module):
    def __init__(self, in_channels=1, base_filters=32):
        super().__init__()
        
        # ENCODER
        self.enc1 = ResConvBlock1D(in_channels, base_filters)
        self.pool1 = nn.MaxPool1d(2)
        
        self.enc2 = ResConvBlock1D(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool1d(2)
        
        # BOTTLENECK
        self.enc3 = ResConvBlock1D(base_filters * 2, base_filters * 4)
        
        # DECODER
        self.up2 = nn.ConvTranspose1d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ResConvBlock1D(base_filters * 4, base_filters * 2)
        
        self.up1 = nn.ConvTranspose1d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ResConvBlock1D(base_filters * 2, base_filters)
        
        # TASK HEADS
        self.head_denoise = nn.Sequential(
            nn.Conv1d(base_filters, base_filters // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(base_filters // 2, 1, kernel_size=1)
        )
        
        self.head_heatmap = nn.Sequential(
            nn.Conv1d(base_filters, base_filters // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(base_filters // 2, 1, kernel_size=1)
        )
        
        self.head_width = nn.Sequential(
            nn.Conv1d(base_filters, base_filters // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(base_filters // 2, 2, kernel_size=1),
            nn.Softplus()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        d2 = self.up2(e3)
        if d2.shape[-1] != e2.shape[-1]:
            d2 = F.interpolate(d2, size=e2.shape[-1], mode='linear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape[-1] != e1.shape[-1]:
            d1 = F.interpolate(d1, size=e1.shape[-1], mode='linear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        clean_psd = self.head_denoise(d1)
        heatmap_logits = self.head_heatmap(d1)
        widths = self.head_width(d1)
        
        return clean_psd, heatmap_logits, widths


import torch
from torchview import draw_graph
# Import your model class here
# from your_module import ImprovedMultiTaskUNet1D

model = ImprovedMultiTaskUNet1D(in_channels=1, base_filters=32)

# Pass input shape: (Batch Size, Channels, Sequence Length)
model_graph = draw_graph(
    model, 
    input_size=(1, 1, 1024),
    expand_nested=True,        # Set to True to see inside ResConvBlock1D
    depth=3,                   # Controls how deep into nested sub-modules to render
    device='cpu'
)

# Render and save the figure
model_graph.visual_graph.render("multitask_unet_architecture", format="png")


import torch
from torchviz import make_dot
# Import your model class here
# from your_module import ImprovedMultiTaskUNet1D

model = ImprovedMultiTaskUNet1D(in_channels=1, base_filters=32)
x = torch.randn(1, 1, 1024)

# Run model
clean_psd, heatmap_logits, widths = model(x)

# Create visualization dictionary for all 3 task outputs
dot = make_dot(
    (clean_psd, heatmap_logits, widths), 
    params=dict(list(model.named_parameters()) + [('input', x)]),
    show_attrs=True,
    show_saved=True
)

# Save as PNG image
dot.format = 'png'
dot.render('torchviz_multitask_unet')


import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_compact_pub_unet():
    # Set publication figure dimensions (Width: 7 inches, suitable for single/double column)
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=300)
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.5, 5.5)
    ax.axis("off")

    # Publication-friendly color palette (Colorblind safe)
    c_enc = "#3182bd"    # Blue
    c_btn = "#756bb1"    # Purple
    c_dec = "#31a354"    # Green
    c_head = "#e6550D"   # Orange
    c_skip = "#636363"   # Dark Gray

    def draw_block(x, y, w, h, title, subtitle, color):
        """Draws a clean, compact module block."""
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.1",
            facecolor=color, edgecolor="black", linewidth=1.0, zorder=3
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + 0.12, title, color="white", weight="bold", 
                fontsize=8, ha="center", va="center", zorder=4)
        ax.text(x + w/2, y + h/2 - 0.18, subtitle, color="white", style="italic", 
                fontsize=7, ha="center", va="center", zorder=4)

    def draw_arrow(x1, y1, x2, y2, color="black", linestyle="-", label=""):
        ax.annotate(
            label, xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2, linestyle=linestyle),
            fontsize=6.5, ha="center", va="center", zorder=5
        )

    # 1. INPUT
    ax.text(-0.3, 4.2, "Input PSD\n[B, 1, N]", fontsize=8, weight="bold", ha="center")
    draw_arrow(0.2, 4.2, 0.7, 4.2)

    # 2. ENCODER PATH (Left)
    draw_block(0.7, 3.7, 1.8, 1.0, "ResBlock 1", "32 ch | N", c_enc)
    draw_arrow(1.6, 3.7, 1.6, 2.7, color="red", label="Pool")
    
    draw_block(0.7, 1.7, 1.8, 1.0, "ResBlock 2", "64 ch | N/2", c_enc)
    draw_arrow(1.6, 1.7, 1.6, 0.7, color="red", label="Pool")

    # 3. BOTTLENECK (Bottom Middle)
    draw_block(3.8, 0.2, 2.2, 1.0, "Bottleneck (Enc 3)", "128 ch | N/4", c_btn)

    # 4. DECODER PATH (Right)
    draw_arrow(6.0, 0.7, 7.3, 1.7, color="green", label="UpConv")
    draw_block(7.3, 1.7, 2.0, 1.0, "Decoder 2", "64 ch | N/2", c_dec)
    
    draw_arrow(8.3, 2.7, 8.3, 3.7, color="green", label="UpConv")
    draw_block(7.3, 3.7, 2.0, 1.0, "Decoder 1", "32 ch | N", c_dec)

    # 5. SKIP CONNECTIONS
    draw_arrow(2.5, 4.2, 7.3, 4.2, color=c_skip, linestyle="--", label="Skip 1")
    draw_arrow(2.5, 2.2, 7.3, 2.2, color=c_skip, linestyle="--", label="Skip 2")

    # 6. MULTI-TASK HEADS (Branching Output)
    draw_arrow(9.3, 4.5, 10.3, 4.9, color=c_head)
    draw_block(10.3, 4.6, 1.8, 0.6, "Head: Denoise", "[B, 1, N]", c_head)

    draw_arrow(9.3, 4.2, 10.3, 4.2, color=c_head)
    draw_block(10.3, 3.9, 1.8, 0.6, "Head: Heatmap", "[B, 1, N]", c_head)

    draw_arrow(9.3, 3.9, 10.3, 3.5, color=c_head)
    draw_block(10.3, 3.2, 1.8, 0.6, "Head: Widths", "[B, 2, N]", c_head)

    plt.tight_layout()
    
    # Save directly as PDF/PNG for publication insertion
    plt.savefig("unet_publication_schematic.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("unet_publication_schematic.png", format="png", dpi=300, bbox_inches="tight")
    print("Saved 'unet_publication_schematic.pdf' and '.png'!")
    plt.show()

if __name__ == "__main__":
    generate_compact_pub_unet()