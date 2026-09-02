import matplotlib.pyplot as plt
import numpy as np
import torch
from DL.data.generate_synthetic_data import PSDSplitDataset
from DL.models.resnet import SpectralResNet1D
from DL.models.unet import SpectralUNet1D


# =====================================================================
# A. LOAD MODEL FROM CHECKPOINT
# =====================================================================
def load_trained_model(
    checkpoint_path="checkpoints/best_spectral_resnet.pth",
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint.get("model_type", "resnet").lower()

    if model_type == "unet":
        model = SpectralUNet1D().to(device)
    elif model_type == "resnet":
        model = SpectralResNet1D().to(device)
    else:
        raise ValueError(f"Unknown model type '{model_type}' in checkpoint.")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"Successfully loaded '{model_type}' from '{checkpoint_path}' (Val Loss was: {checkpoint['best_val_loss']:.6f})"
    )
    return model, device


# =====================================================================
# B. INFERENCE ON SYNTHETIC TEST SET SAMPLES
# =====================================================================
def predict_and_plot_test_samples(
    model,
    device,
    dataset_path="psd_dataset_splits.npz",
    num_samples=3,
):
    test_ds = PSDSplitDataset(filepath=dataset_path, split="test")
    f = test_ds.f

    indices = np.random.choice(len(test_ds), size=num_samples, replace=False)

    fig, axes = plt.subplots(
        num_samples, 1, figsize=(11, 3.8 * num_samples), sharex=True
    )
    if num_samples == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        noisy_x, clean_y = test_ds[idx]

        # Prepare batch input: shape (1, 1, n_freqs)
        input_tensor = noisy_x.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_log_psd = model(input_tensor).squeeze().cpu().numpy()

        noisy_log = noisy_x.squeeze().numpy()
        clean_log = clean_y.squeeze().numpy()
        meta = test_ds.metadata[idx]

        # Plot comparison
        ax.plot(
            f,
            noisy_log,
            color="gray",
            alpha=0.5,
            linewidth=1.0,
            label="Noisy Observation",
        )
        ax.plot(
            f,
            clean_log,
            color="black",
            linestyle="--",
            linewidth=1.8,
            label="Ground Truth",
        )
        ax.plot(
            f,
            pred_log_psd,
            color="crimson",
            linewidth=2.0,
            label="Model Prediction",
        )

        for p_info in meta["peaks"]:
            ax.axvline(
                x=p_info["f0"],
                color="navy",
                linestyle=":",
                alpha=0.6,
                label=f"f0={p_info['f0']:.1f}Hz",
            )

        ax.set_ylabel("Log Power")
        ax.set_title(
            f"Test Sample #{idx} | Peaks: {meta['n_peaks']} | Noise Std: {meta['noise_level']:.2f}"
        )
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()


# =====================================================================
# C. INFERENCE ON RAW REAL-WORLD EEG/LFP DATA
# =====================================================================
def predict_raw_psd(
    model,
    device,
    f_real,
    psd_real,
    target_n_freqs=500,
    target_f_max=100.0,
):
    """Interpolates real empirical PSD to standard frequency bin length,

    runs inference, and maps prediction back to linear power scale.
    """
    f_grid = np.linspace(0.1, target_f_max, target_n_freqs)

    # 1. Transform real data to log domain & resample to network grid
    log_psd_real = np.log(psd_real + 1e-12)
    resampled_log_psd = np.interp(f_grid, f_real, log_psd_real)

    # 2. Format tensor
    input_tensor = (
        torch.tensor(resampled_log_psd, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    # 3. Model pass
    with torch.no_grad():
        pred_log_psd = model(input_tensor).squeeze().cpu().numpy()

    # 4. Map back to original frequency grid and linear power
    recovered_log_psd_real = np.interp(f_real, f_grid, pred_log_psd)
    recovered_psd_real = np.exp(recovered_log_psd_real)

    return recovered_psd_real, recovered_log_psd_real


# =====================================================================
# EXECUTION
# =====================================================================
if __name__ == "__main__":
    CHECKPOINT = "checkpoints/best_spectral_resnet.pth"

    # 1. Load model
    model, device = load_trained_model(CHECKPOINT)

    # 2. Test set predictions
    predict_and_plot_test_samples(
        model, device, dataset_path="psd_dataset_splits.npz", num_samples=10
    )