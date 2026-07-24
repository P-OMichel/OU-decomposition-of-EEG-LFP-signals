import numpy as np
import scipy.signal as signal
from scipy.optimize import curve_fit

# ---------------------------------------------------------
# 1. Single and Multi-Component Pseudo-Voigt PSD Model
# ---------------------------------------------------------
def single_pseudo_voigt_psd(f, A, f0, gamma, eta):
    """
    One-sided Pseudo-Voigt PSD profile.
    - A: Amplitude scaling
    - f0: Peak center frequency
    - gamma: Half-width parameter
    - eta: Mixing factor (0 = Pure Gaussian, 1 = Pure Lorentzian)
    """
    # Equivalent Gaussian sigma from Lorentzian gamma for matched FWHM
    sigma = gamma / np.sqrt(2 * np.log(2))
    
    # Positive and negative frequency terms for real signals
    f_pos = f - f0
    f_neg = f + f0
    
    # Lorentzian parts (Heavy tails)
    L_pos = (1.0 / np.pi) * (gamma / (f_pos**2 + gamma**2))
    L_neg = (1.0 / np.pi) * (gamma / (f_neg**2 + gamma**2))
    
    # Gaussian parts (Sharp peak core)
    G_pos = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (f_pos / sigma)**2)
    G_neg = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (f_neg / sigma)**2)
    
    # Linear combination
    PV_pos = eta * L_pos + (1.0 - eta) * G_pos
    PV_neg = eta * L_neg + (1.0 - eta) * G_neg
    
    return A**2 * (PV_pos + PV_neg)

def multi_pseudo_voigt_psd(f, *params):
    """
    params layout: [A_0, f0_0, gamma_0, eta_0, ..., A_K, f0_K, gamma_K, eta_K]
    4 parameters per peak component.
    """
    psd = np.zeros_like(f)
    num_components = len(params) // 4
    for i in range(num_components):
        A, f0, gamma, eta = params[4*i : 4*i + 4]
        psd += single_pseudo_voigt_psd(f, A, f0, gamma, eta)
    return psd

def multi_pseudo_voigt_log_psd(f, *params):
    return np.log(multi_pseudo_voigt_psd(f, *params) + 1e-12)

# ---------------------------------------------------------
# 2. Pseudo-Voigt Fitting Pipeline
# ---------------------------------------------------------
def fit_pseudo_voigt_mixture_psd(f_emp, psd_emp, prominence=1.0, max_components=5):
    """
    Fits Pseudo-Voigt mixture (Gaussian peak top + Lorentzian tail mixture).
    Returns: best_K, components, best_fit_results
    """
    log_psd = np.log(psd_emp + 1e-12)
    peaks, _ = signal.find_peaks(
        log_psd, 
        prominence=prominence, 
        distance=max(1, int(len(f_emp) * 0.02))
    )
    
    if len(peaks) == 0:
        peaks = [np.argmax(psd_emp)]
        
    detected_freqs = f_emp[peaks]
    detected_powers = psd_emp[peaks]
    sorted_idx = np.argsort(detected_powers)[::-1]
    detected_freqs = detected_freqs[sorted_idx]
    
    max_K = min(max_components, max(len(detected_freqs), max_components))
    best_aic = np.inf
    best_fit_results = None
    best_K = 0
    N_data = len(f_emp)
    
    for K in range(1, max_K + 1):
        p0 = []
        bounds_lower = []
        bounds_upper = []
        
        for k in range(K):
            if k < len(detected_freqs):
                f0_guess = detected_freqs[k]
                A_guess = np.sqrt(psd_emp[peaks[sorted_idx[k]]])
            else:
                f0_guess = (k + 1) * (f_emp[-1] / (K + 1))
                A_guess = np.sqrt(np.mean(psd_emp))
                
            gamma_guess = 1.0
            eta_guess = 0.5  # Equal mix of Gaussian and Lorentzian
            
            p0.extend([A_guess, gamma_guess, f0_guess, eta_guess])
            
            bounds_lower.extend([0.0, 1e-3, 0.0, 0.0])
            bounds_upper.extend([np.inf, 100.0, f_emp[-1], 1.0])

        try:
            popt, _ = curve_fit(
                multi_pseudo_voigt_log_psd, 
                f_emp, 
                np.log(psd_emp + 1e-12), 
                p0=p0, 
                bounds=(bounds_lower, bounds_upper),
                maxfev=5000
            )
            
            residuals = np.log(psd_emp + 1e-12) - multi_pseudo_voigt_log_psd(f_emp, *popt)
            rss = np.sum(residuals**2)
            
            num_params = 4 * K
            aic = N_data * np.log(rss / N_data) + 2 * num_params
            
            if aic < best_aic:
                best_aic = aic
                best_K = K
                best_fit_results = popt
                
        except RuntimeError:
            continue

    components = []
    for k in range(best_K):
        components.append({
            'component': k + 1,
            'A_est': best_fit_results[4*k],
            'gamma_est (width)': best_fit_results[4*k + 1],
            'f0_est_Hz': best_fit_results[4*k + 2],
            'eta_est (mixing factor)': best_fit_results[4*k + 3]
        })

    return best_K, components, best_fit_results