'''
File to eavluate the fit using different mixture models of a simulated via sum of OU process or real EEG signal
'''

import numpy as np
import scipy as sc
import scipy.signal as signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from Functions.generate_OU import get_mixed_OU_signals_exact, get_analytical_psd
from Functions.fit_OU import fit_ou_mixture_psd, multi_ou_psd
from Functions.fit_student_t import fit_student_t_mixture_psd, multi_student_t_psd
from Functions.fit_student_t import fit_dual_student_t_mixture_psd, multi_dual_student_t_psd
from Functions.fit_pearson import fit_pearson_iv_mixture_psd, multi_pearson_iv_psd
from Functions.fit_pseudo_voigt import fit_pseudo_voigt_mixture_psd, multi_pseudo_voigt_psd
from Functions.time_frequency import spectrogram

from specparam import SpectralModel # for FOOOF method


USE_simulated = False
EVALUATE_time = False

if USE_simulated:
    # --- Parameters
    T = 1000 # desired signal duration (s)
    dt = 0.001
    fs = 1 / dt

    lbda_list = [1, 2, 1]
    omega_list = [2*np.pi*1, 2*np.pi*10, 2*np.pi*14]
    sigma_list = [3, 2, 10]
    factor_list = [1, 1, 0.5]

    # --- Generate EEG data
    t, y = get_mixed_OU_signals_exact(T, dt, lbda_list, omega_list, sigma_list, factor_list)

else: 
    file = r'c:\Users\holcman\Documents\GitHub\EEG-labellisation-app---Spectrogram\anesthesia_database\rec_20240321_085300.npy'

    fs = 128
    y = np.load(file)
    y = y[2100 * fs: 2250 * fs]  #[0 * fs: 200 * fs] #[2100 * fs: 2250 * fs]
    t = np.arange(len(y)) / fs

# --- Get PSD
f_psd, psd = signal.welch(y, fs=fs, nperseg=int(fs * 16))
fit_mask = (f_psd > 0) & (f_psd < 45.0)
f_psd = f_psd[fit_mask]
psd = psd[fit_mask]
freq_range = [0.1, 45]

fm = SpectralModel(aperiodic_mode='fixed', periodic_mode='cauchy', peak_width_limits=[0.2, 15.0], max_n_peaks=4, min_peak_height=0.1)
fm.fit(f_psd, psd, freq_range)
fm.report(f_psd, psd, freq_range)

# --- get various fits

# OU Mixture
best_K_OU, fitted_components_OU, popt_OU = fit_ou_mixture_psd(f_psd, psd, prominence = 1, max_components=4)
psd_model_OU = multi_ou_psd(f_psd, *popt_OU)

# Student-t mixture
best_K_St, fitted_components_St, popt_St = fit_student_t_mixture_psd(f_psd, psd, prominence = 1, max_components=4)
psd_model_St = multi_student_t_psd(f_psd, *popt_St)

# Student-t mixture
best_K_DSt, fitted_components_DSt, popt_DSt = fit_dual_student_t_mixture_psd(f_psd, psd, prominence = 1, max_components=4)
psd_model_DSt = multi_dual_student_t_psd(f_psd, *popt_DSt)

# Pearson mixture
best_K_Pe, fitted_components_Pe, popt_Pe = fit_pearson_iv_mixture_psd(f_psd, psd, prominence = 1, max_components=4)
psd_model_Pe = multi_pearson_iv_psd(f_psd, *popt_Pe)

# Pseudo Voigt mixture
best_K_PV, fitted_components_PV, popt_PV = fit_pseudo_voigt_mixture_psd(f_psd, psd, prominence = 1, max_components=4)
psd_model_PV = multi_pseudo_voigt_psd(f_psd, *popt_PV)

# --- Display

fig, axis = plt.subplots(1, constrained_layout = True)
axis.semilogy(f_psd, psd, label = 'Welch PSD')
axis.semilogy(f_psd, psd_model_OU, label = f'OU | {len(fitted_components_OU)} components')
axis.semilogy(f_psd, psd_model_St, label = f'Student-t | {len(fitted_components_St)} components')
axis.semilogy(f_psd, psd_model_DSt, label = f'Dual Student-t| {len(fitted_components_St)} components')
axis.semilogy(f_psd, psd_model_Pe, label = f'Pearson | {len(fitted_components_Pe)} components')
axis.semilogy(f_psd, psd_model_PV, label = f'Pseudo-Voigt | {len(fitted_components_PV)} components')
axis.legend()

plt.show()

if EVALUATE_time:

    import timeit

    models = {
        "OU Mixture": lambda: fit_ou_mixture_psd(f_psd, psd, prominence=1, max_components=4),
        "Student-t Mixture": lambda: fit_student_t_mixture_psd(f_psd, psd, prominence=1, max_components=4),
        "Pearson Mixture": lambda: fit_pearson_iv_mixture_psd(f_psd, psd, prominence=1, max_components=4),
        "Pseudo Voigt Mixture": lambda: fit_pseudo_voigt_mixture_psd(f_psd, psd, prominence=1, max_components=4),
    }

    # Run each function 10 times and get average duration per run
    for name, func in models.items():
        runs = 10
        total_time = timeit.timeit(func, number=runs)
        avg_time = total_time / runs
        print(f"{name:25s}: {avg_time * 1000:.2f} ms per run (averaged over {runs} runs)")



