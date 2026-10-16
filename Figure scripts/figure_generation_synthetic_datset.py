import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Optional dependency: used if present in your environment, otherwise falls back to basic FWHM
try:
    from Functions.extract_psd_bumps import extract_intervals_method1_detrended_seed
    HAS_METHOD1 = True
except ImportError:
    HAS_METHOD1 = False


def generate_and_plot_stages_svg(
    f_min: float = 0.1,
    f_max: float = 50.0,
    n_freqs: int = 250,
    n_obs_range: tuple = (80, 350),
    k_welch_range: tuple = (4, 32),
    save_filename: str = "psd_generation_stages.svg",
    seed: int = 42,
):
    """
    Simulates a single sample through each discrete stage of the synthesis pipeline
    and saves a clean, layered vector SVG plot for editing in Illustrator/Inkscape.
    """
    np.random.seed(seed)
    f_canonical = np.linspace(f_min, f_max, n_freqs, dtype=np.float32)

    # =========================================================================
    # STAGE 1: 1/f Background Generation
    # =========================================================================
    chi = np.random.uniform(0.8, 3.0)
    b_amp = np.random.uniform(0.5, 3.0)
    has_knee = True  # Enforce knee for clear visual illustration
    f_knee = np.random.uniform(4.0, 15.0)

    f_safe = np.maximum(f_canonical, 0.05)
    if has_knee:
        bg = b_amp / (f_knee**chi + f_safe**chi)
    else:
        bg = b_amp / (f_safe**chi)

    # =========================================================================
    # STAGE 2: Individual Resonant Peaks Generation
    # =========================================================================
    # Fix n_peaks = 3 to give a rich visual demonstration
    n_peaks = 3
    shape_pool = ["pseudo_voigt", "pearson_iv", "student_t", "ornstein_uhlenbeck"]
    peaks_meta = []
    peak_curves = []

    # Well-separated center frequencies for visual clarity
    f0_choices = [8.0, 20.0, 35.0]

    for f0 in f0_choices:
        shape_type = np.random.choice(shape_pool)
        A = np.random.uniform(2.0, 6.0)
        gamma = np.random.uniform(0.8, 2.2)  # Yields FWHM in [1.5, 4.5] Hz

        f = f_canonical
        if shape_type == "pseudo_voigt":
            eta = np.random.uniform(0.2, 0.8)
            sigma = gamma / np.sqrt(2.0 * np.log(2.0))
            f_pos, f_neg = f - f0, f + f0
            L_pos = (1.0 / np.pi) * (gamma / (f_pos**2 + gamma**2))
            L_neg = (1.0 / np.pi) * (gamma / (f_neg**2 + gamma**2))
            G_pos = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * (f_pos / sigma)**2)
            G_neg = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * (f_neg / sigma)**2)
            peak_signal = A**2 * (eta * L_pos + (1.0 - eta) * G_pos + eta * L_neg + (1.0 - eta) * G_neg)

        elif shape_type == "ornstein_uhlenbeck":
            f_pos, f_neg = f - f0, f + f0
            peak_signal = (A**2 / np.pi) * (
                (gamma / (gamma**2 + f_pos**2)) + (gamma / (gamma**2 + f_neg**2))
            )

        elif shape_type == "pearson_iv":
            m = np.random.uniform(1.2, 3.0)
            nu_p = np.random.uniform(-1.5, 1.5)
            y_pos, y_neg = (f - f0) / gamma, (f + f0) / gamma
            term_pos = (1.0 + y_pos**2)**(-m) * np.exp(-nu_p * np.arctan(y_pos))
            term_neg = (1.0 + y_neg**2)**(-m) * np.exp(-nu_p * np.arctan(y_neg))
            peak_signal = A**2 * (term_pos + term_neg)

        elif shape_type == "student_t":
            alpha = np.random.uniform(1.5, 3.5)
            term_pos = 1.0 + ((f - f0) / gamma)**2
            term_neg = 1.0 + ((f + f0) / gamma)**2
            peak_signal = (A**2 / np.pi) * (
                (1.0 / term_pos**(alpha / 2.0)) + (1.0 / term_neg**(alpha / 2.0))
            )

        # Optional tail
        if f0 > 2.0:
            beta_decay = np.random.uniform(1.2, 2.5)
            f_tail = np.maximum(f, 1e-6)
            post_decay = (
                (A**2 * 0.3)
                * ((f_tail / f0) ** (-beta_decay))
                * np.maximum(0.0, 1.0 - (f0 / f_tail))
            )
            peak_signal += np.where(f > f0, post_decay, 0.0)

        peak_curves.append(peak_signal)
        peaks_meta.append({"f0": f0, "A": A, "gamma": gamma, "shape": shape_type})

    summed_peaks = np.sum(peak_curves, axis=0)

    # =========================================================================
    # STAGE 3: Summation (Clean Synthetic Spectrum)
    # =========================================================================
    psd_clean = bg + summed_peaks
    log_clean = np.log(np.maximum(psd_clean, 1e-12)).astype(np.float32)

    # =========================================================================
    # STAGE 4: Peak Width / Interval Estimation
    # =========================================================================
    intervals = []
    f0_list = [p["f0"] for p in peaks_meta]

    if HAS_METHOD1:
        intervals = extract_intervals_method1_detrended_seed(
            f_canonical, log_clean, f0_list, lam=1e2, max_high_ratio=0.01, min_boundary_ratio=0.5
        )
    else:
        # Standalone FWHM-like fallback boundaries if Functions folder is not in path
        for p in peaks_meta:
            intervals.append({
                "f0": p["f0"],
                "peak_edges": [p["f0"] - p["gamma"], p["f0"] + p["gamma"]],
                "segment_bounds": [p["f0"] - 1.5 * p["gamma"], p["f0"] + 1.5 * p["gamma"]]
            })

    # =========================================================================
    # STAGE 5: Observation Grid & Welch Averaging Noise Addition
    # =========================================================================
    n_obs = int(np.random.randint(n_obs_range[0], n_obs_range[1] + 1))
    f_obs = np.linspace(f_min, f_max, n_obs, dtype=np.float32)

    clean_interp = interp1d(f_canonical, log_clean, kind="linear", fill_value="extrapolate")
    log_clean_obs = clean_interp(f_obs)

    k_seg = int(np.random.randint(k_welch_range[0], k_welch_range[1] + 1))
    noise_mult = np.random.gamma(shape=k_seg, scale=1.0 / k_seg, size=n_obs).astype(np.float32)
    linear_noisy_obs = np.exp(log_clean_obs) * noise_mult
    log_noisy_obs = np.log(np.maximum(linear_noisy_obs, 1e-12))

    recon_interp = interp1d(f_obs, log_noisy_obs, kind="linear", fill_value="extrapolate")
    log_noisy = recon_interp(f_canonical).astype(np.float32)

    # =========================================================================
    # FIGURE COMPOSITION (Exported as vector SVG)
    # =========================================================================
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["svg.fonttype"] = "none"  # Keep fonts as editable text in Illustrator/Inkscape

    fig, axes = plt.subplots(5, 1, figsize=(9, 13), sharex=True, constrained_layout=True)

    # Stage 1: Background
    axes[0].plot(f_canonical, np.log(bg), color="#1f77b4", lw=2, label=r"Background $\log(S_{bg})$")
    axes[0].set_title("1. Aperiodic Background Model ($1/f^{\\chi}$ with Knee)", fontsize=11, loc="left", fontweight="bold")
    axes[0].set_ylabel(r"Log Power", fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", frameon=True)

    # Stage 2: Individual Peaks
    colors = ["#2ca02c", "#d62728", "#9467bd"]
    for i, (pk, meta) in enumerate(zip(peak_curves, peaks_meta)):
        axes[1].plot(
            f_canonical, pk, lw=1.8, color=colors[i % len(colors)],
            label=f"{meta['shape']} ($f_0$={meta['f0']:.1f} Hz, $\\gamma$={meta['gamma']:.2f})"
        )
    axes[1].set_title("2. Resonant Peaks Generation (Diverse Profiles & Decays)", fontsize=11, loc="left", fontweight="bold")
    axes[1].set_ylabel("Linear Amplitude", fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", frameon=True)

    # Stage 3: Summation
    axes[2].plot(f_canonical, log_clean, color="#111111", lw=2, label=r"$\log(S_{bg} + \sum \mathrm{Peaks})$")
    axes[2].set_title("3. Composite Clean Signal (Log-Transformed)", fontsize=11, loc="left", fontweight="bold")
    axes[2].set_ylabel("Log Power", fontsize=9)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper right", frameon=True)

    # Stage 4: Interval / Width Estimation
    axes[3].plot(f_canonical, log_clean, color="#333333", lw=1.5, alpha=0.7, label="Clean Ground Truth")
    for info in intervals:
        f0 = info["f0"]
        f_left = info["peak_edges"][0] if info["peak_edges"][0] is not None else info["segment_bounds"][0]
        f_right = info["peak_edges"][1] if info["peak_edges"][1] is not None else info["segment_bounds"][1]

        axes[3].axvline(f0, color="#d62728", linestyle="--", lw=1.2, label=r"Center $f_0$" if f0 == f0_choices[0] else "")
        axes[3].axvspan(f_left, f_right, color="#ff7f0e", alpha=0.25, label="Estimated Interval" if f0 == f0_choices[0] else "")
    axes[3].set_title("4. Peak Centers and Interval Boundary Extraction", fontsize=11, loc="left", fontweight="bold")
    axes[3].set_ylabel("Log Power", fontsize=9)
    axes[3].grid(True, linestyle="--", alpha=0.5)
    axes[3].legend(loc="upper right", frameon=True)

    # Stage 5: Grid Observation & Noise Addition
    axes[4].plot(f_canonical, log_clean, color="#444444", lw=1.2, linestyle="--", label="Clean Baseline")
    axes[4].plot(f_canonical, log_noisy, color="#004c6d", lw=1.2, alpha=0.85, label=f"Welch Noise ($K$={k_seg}, $N_{{obs}}$={n_obs})")
    axes[4].set_title("5. Welch Multiplicative Gamma Noise & Canonical Grid Interpolation", fontsize=11, loc="left", fontweight="bold")
    axes[4].set_ylabel("Log Power", fontsize=9)
    axes[4].set_xlabel("Frequency (Hz)", fontsize=10)
    axes[4].set_xlim(f_min, f_max)
    axes[4].grid(True, linestyle="--", alpha=0.5)
    axes[4].legend(loc="upper right", frameon=True)

    plt.savefig(save_filename, format="svg", bbox_inches="tight")
    plt.close()
    print(f"SVG figure successfully saved as: {save_filename}")


if __name__ == "__main__":
    generate_and_plot_stages_svg()