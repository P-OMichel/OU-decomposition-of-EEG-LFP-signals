import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from Functions.generate_OU import get_mixed_OU_signals_exact, get_analytical_psd

USE_simulated = True

if USE_simulated:
    # --- Parameters
    T = 1000 # desired signal duration (s)
    dt = 0.001
    fs = 1 / dt

    lbda_list = [1, 2, 1]
    omega_list = [2*np.pi*1, 2*np.pi*10, 2*np.pi*30]
    sigma_list = [3, 2, 10]
    factor_list = [1, 1, 0.01]

    # --- Generate EEG data
    t, y = get_mixed_OU_signals_exact(T, dt, lbda_list, omega_list, sigma_list, factor_list)

else: 
    file = r'c:\Users\holcman\Documents\GitHub\EEG-labellisation-app---Spectrogram\anesthesia_database\rec_20240321_085300.npy'

    fs = 128
    y = np.load(file)
    y = y[2100 * 128: 2250 * 128]
    t = np.arange(len(y)) / fs

# --- Get PSD
f_psd, psd = signal.welch(y, fs=fs, nperseg=int(fs * 16))
fit_mask = (f_psd > 0) & (f_psd < 45.0)
freqs = f_psd[fit_mask]
pxx = psd[fit_mask]
# -------------------------------------------------------------
# Method 1: Savitzky-Golay Filter (Recommended for PSD)
# -------------------------------------------------------------
# window_length: Number of points in the filter window (must be an odd integer)
# polyorder: Polynomial order used to fit the samples (must be less than window_length)
window_length = 51  
polyorder = 3

pxx_savgol = signal.savgol_filter(pxx, window_length=window_length, polyorder=polyorder)


# -------------------------------------------------------------
# Method 2: Zero-Phase Butterworth Low-Pass Filter
# -------------------------------------------------------------
# Cutoff frequency normalized to Nyquist rate of the PSD array index (0.0 to 1.0)
cutoff = 0.05  # Lower value = stronger smoothing
order = 2

b, a = signal.butter(order, cutoff, btype='low')
pxx_lowpass = signal.filtfilt(b, a, pxx)


# -------------------------------------------------------------
# Visualization
# -------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.semilogy(freqs, pxx, alpha=0.4, label='Original PSD (Welch)')
plt.semilogy(freqs, pxx_savgol, label='Savitzky-Golay Filter', linewidth=2)
plt.semilogy(freqs, pxx_lowpass, '--', label='Butterworth Low-Pass', linewidth=2)

plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('PSD Smoothing Comparison')
plt.legend()
plt.grid(True)
plt.show()