import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from Functions.extract_psd_bumps import whittaker_smooth_asymmetric
from Functions.generate_OU import get_analytical_psd, get_mixed_OU_signals_exact

# --- Parameters & Synthesis
T, dt, fs = 1000, 0.001, 1000
lbda_list, omega_list = [1, 2, 1], [2*np.pi*1, 2*np.pi*10, 2*np.pi*30]
sigma_list, factor_list = [3, 5, 10], [1, 1, 0.5]
t, y = get_mixed_OU_signals_exact(T, dt, lbda_list, omega_list, sigma_list, factor_list)

# --- Welch PSD & Artificial Noise
f_psd, psd = signal.welch(y, fs=fs, nperseg=int(fs * 16))
mask = (f_psd > 0) & (f_psd < 45.0)
f_psd, psd = f_psd[mask], np.log(psd[mask] + 1e-11)
k = 15
psd[k:] += -0 * np.log(f_psd[k:] + 1e-11) + np.log(psd[k])
noise_level = 1
log_noise = np.random.normal(loc=0.0, scale=noise_level, size=len(f_psd))
psd_noisy = psd + log_noise 

# --- Analytical PSD
f_a, psd_a = get_analytical_psd(N=len(psd), f_stop=45, lbda_list=lbda_list, omega_list=omega_list, sigma_list=sigma_list, factor_list=factor_list)
psd_a = np.log(psd_a + 1e-11)
psd_a[k:] += -0 * np.log(f_a[k:] + 1e-11) + np.log(psd_a[k])

# --- Baselines (Low vs High Lambda)
lam_low, lam_high, p_val = 1e2, 1e6, 0.01
bl_a_low = whittaker_smooth_asymmetric(psd_a, lam=lam_low, p=p_val)
bl_a_high = whittaker_smooth_asymmetric(psd_a, lam=lam_high, p=p_val)
bl_low = whittaker_smooth_asymmetric(psd, lam=lam_low, p=p_val)
bl_high = whittaker_smooth_asymmetric(psd, lam=lam_high, p=p_val)
bl_n_low = whittaker_smooth_asymmetric(psd_noisy, lam=lam_low, p=p_val)
bl_n_high = whittaker_smooth_asymmetric(psd_noisy, lam=lam_high, p=p_val)

# --- Plotting via Loop
datasets = [
    (f_a, psd_a, bl_a_low, bl_a_high, "Analytical"),
    (f_psd, psd, bl_low, bl_high, "Clean Welch"),
    (f_psd, psd_noisy, bl_n_low, bl_n_high, "Noisy Welch")
]

fig, axes = plt.subplots(3, 2, figsize=(11, 8), constrained_layout=True)

for i, (f, raw, bl_l, bl_h, title) in enumerate(datasets):
    # Left Column: Raw & Baselines
    axes[i, 0].plot(f, raw, "k", alpha=0.6, label="PSD")
    axes[i, 0].plot(f, bl_l, "r--", label=r"$\lambda=10^2$")
    axes[i, 0].plot(f, bl_h, "b-.", label=r"$\lambda=10^6$")
    axes[i, 0].set(title=f"{title}: PSD & Baselines", ylabel="Log Power")
    axes[i, 0].legend(loc="upper right")
    
    # Right Column: Detrended Signals
    axes[i, 1].plot(f, np.abs(raw - bl_l), "r", alpha=0.7, label=r"$\lambda=10^2$")
    axes[i, 1].plot(f, np.abs(raw - bl_h), "b", alpha=0.7, label=r"$\lambda=10^6$")
    axes[i, 1].set(title=f"{title}: Detrended", ylabel="Absolute Excess Power")
    axes[i, 1].legend(loc="upper right")

axes[2, 0].set_xlabel("Frequency (Hz)")
axes[2, 1].set_xlabel("Frequency (Hz)")
plt.show()