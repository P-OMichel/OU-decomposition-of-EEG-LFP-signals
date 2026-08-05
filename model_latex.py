import os
import sys

# Append path to cloned PlotNeuralNet repo
sys.path.append("PlotNeuralNet")

from pycore.blocks import *
from pycore.tikzeng import *


def generate_pytorch_unet_tikz():
    arch = [
        to_head(".."),
        to_cor(),
        to_begin(),
        # -----------------------------------------------------------------
        # INPUT & ENCODER (Assuming Sequence Length N = 1024)
        # -----------------------------------------------------------------
        # Input: [1, 1024]
        to_Conv(
            "input",
            s_filer=1024,
            n_filer=1,
            offset="(0,0,0)",
            to="(0,0,0)",
            height=12,
            depth=12,
            width=1,
            caption="Input PSD",
        ),
        # Enc1: ResConvBlock1D (1 -> 32 filters, Length 1024)
        to_Conv(
            "enc1",
            s_filer=1024,
            n_filer=32,
            offset="(1.2,0,0)",
            to="(input-east)",
            height=12,
            depth=12,
            width=2.5,
            caption="Enc 1 (32)",
        ),
        to_connection("input", "enc1"),
        # Pool1: MaxPool1d(2) -> Length 512
        to_Pool(
            "pool1",
            offset="(0,0,0)",
            to="(enc1-east)",
            height=8,
            depth=8,
            width=1,
        ),
        # Enc2: ResConvBlock1D (32 -> 64 filters, Length 512)
        to_Conv(
            "enc2",
            s_filer=512,
            n_filer=64,
            offset="(1.5,0,0)",
            to="(pool1-east)",
            height=8,
            depth=8,
            width=4,
            caption="Enc 2 (64)",
        ),
        # Pool2: MaxPool1d(2) -> Length 256
        to_Pool(
            "pool2",
            offset="(0,0,0)",
            to="(enc2-east)",
            height=5,
            depth=5,
            width=1,
        ),
        # -----------------------------------------------------------------
        # BOTTLENECK
        # -----------------------------------------------------------------
        # Enc3: ResConvBlock1D (64 -> 128 filters, Length 256)
        to_Conv(
            "enc3",
            s_filer=256,
            n_filer=128,
            offset="(1.5,0,0)",
            to="(pool2-east)",
            height=5,
            depth=5,
            width=6,
            caption="Bottleneck (128)",
        ),
        # -----------------------------------------------------------------
        # DECODER
        # -----------------------------------------------------------------
        # Up2: ConvTranspose1d (128 -> 64, Length 512)
        to_UnPool(
            "up2",
            offset="(1.5,0,0)",
            to="(enc3-east)",
            height=8,
            depth=8,
            width=1,
        ),
        # Dec2: Concat(64+64=128) + ResConvBlock1D -> 64 filters, Length 512
        to_ConvRes(
            "dec2",
            s_filer=512,
            n_filer=64,
            offset="(0,0,0)",
            to="(up2-east)",
            height=8,
            depth=8,
            width=4,
            caption="Dec 2 (64)",
        ),
        # Up1: ConvTranspose1d (64 -> 32, Length 1024)
        to_UnPool(
            "up1",
            offset="(1.5,0,0)",
            to="(dec2-east)",
            height=12,
            depth=12,
            width=1,
        ),
        # Dec1: Concat(32+32=64) + ResConvBlock1D -> 32 filters, Length 1024
        to_ConvRes(
            "dec1",
            s_filer=1024,
            n_filer=32,
            offset="(0,0,0)",
            to="(up1-east)",
            height=12,
            depth=12,
            width=2.5,
            caption="Dec 1 (32)",
        ),
        # -----------------------------------------------------------------
        # SKIP CONNECTIONS
        # -----------------------------------------------------------------
        to_Skip("enc2", "dec2", pos=1.25),
        to_Skip("enc1", "dec1", pos=1.25),
        # -----------------------------------------------------------------
        # MULTI-TASK HEADS
        # -----------------------------------------------------------------
        # 1. Denoise Head (32 -> 16 -> 1 channel)
        to_Conv(
            "head_denoise",
            s_filer=1024,
            n_filer=1,
            offset="(2.5,3.0,0)",
            to="(dec1-east)",
            height=12,
            depth=12,
            width=1,
            caption="Clean PSD (1ch)",
        ),
        to_connection("dec1", "head_denoise"),
        # 2. Heatmap Head (32 -> 16 -> 1 channel)
        to_Conv(
            "head_heatmap",
            s_filer=1024,
            n_filer=1,
            offset="(2.5,0,0)",
            to="(dec1-east)",
            height=12,
            depth=12,
            width=1,
            caption="Heatmap (1ch)",
        ),
        to_connection("dec1", "head_heatmap"),
        # 3. Width Head (32 -> 16 -> 2 channels + Softplus)
        to_Conv(
            "head_width",
            s_filer=1024,
            n_filer=2,
            offset="(2.5,-3.0,0)",
            to="(dec1-east)",
            height=12,
            depth=12,
            width=1.5,
            caption="Widths (2ch)",
        ),
        to_connection("dec1", "head_width"),
        to_end(),
    ]
    return arch


if __name__ == "__main__":
    file_name = "improved_multitask_unet1d"
    arch = generate_pytorch_unet_tikz()

    # Writes the LaTeX/TikZ code out to a file
    file_template_script(arch, f"{file_name}.tex")
    print(f"Successfully generated '{file_name}.tex'!")

    # Optionally compile to PDF directly if pdflatex is in system PATH
    os.system(f"pdflatex {file_name}.tex")