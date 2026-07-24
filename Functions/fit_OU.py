import numpy as np
from scipy.optimize import curve_fit
import scipy.signal as signal


def single_ou_psd(f, A, lbda, f0):
    Omega = 2.0 * np.pi * f
    omega0 = 2.0 * np.pi * f0
    return (A**2 / np.pi) * (
        1.0 / (lbda**2 + (Omega - omega0)**2) + 
        1.0 / (lbda**2 + (Omega + omega0)**2)
    )

def multi_ou_psd(f, *params):
    psd = np.zeros_like(f)
    num_components = len(params) // 3
    for i in range(num_components):
        psd += single_ou_psd(f, params[3*i], params[3*i + 1], params[3*i + 2])
    return psd

def multi_ou_log_psd(f, *params):
    """Log-transformed PSD for curve_fit to treat small/large peaks equally."""
    return np.log(multi_ou_psd(f, *params) + 1e-12)

def fit_ou_mixture_psd(f_emp, psd_emp, prominence = 1, max_components=5):

    # 1. Lower prominence & use log scale for peak finding to catch small bumps
    log_psd = np.log(psd_emp + 1e-12)
    peaks, _ = signal.find_peaks(
        log_psd, 
        prominence=prominence, # Fixed log-prominence threshold catches weak peaks
        distance=max(int(len(f_emp) * 0.02), 1)
    )
    
    if len(peaks) == 0:
        peaks = [np.argmax(psd_emp)]
        
    detected_freqs = f_emp[peaks]
    detected_powers = psd_emp[peaks]
    
    # Sort peaks by power descending
    sorted_idx = np.argsort(detected_powers)[::-1]
    detected_freqs = detected_freqs[sorted_idx]
    
    # Allow testing up to max_components even if find_peaks detected fewer
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
                A_guess = np.sqrt(psd_emp[peaks[sorted_idx[k]]] * np.pi * 0.5)
            else:
                # If testing higher K than peaks found, space out remaining guesses
                f0_guess = (k + 1) * (f_emp[-1] / (K + 1))
                A_guess = np.sqrt(np.mean(psd_emp) * np.pi * 0.5)
                
            p0.extend([A_guess, 1.0, f0_guess])
            bounds_lower.extend([0.0, 1e-3, 0.0])
            bounds_upper.extend([np.inf, 100.0, f_emp[-1]])

        try:
            # Fit on LOG-PSD to give equal weight to weak peaks
            popt, _ = curve_fit(
                multi_ou_log_psd, 
                f_emp, 
                np.log(psd_emp + 1e-12), 
                p0=p0, 
                bounds=(bounds_lower, bounds_upper),
                maxfev=5000
            )
            
            # Compute residual sum of squares in log domain
            residuals = np.log(psd_emp + 1e-12) - multi_ou_log_psd(f_emp, *popt)
            rss = np.sum(residuals**2)
            
            # AIC in log domain
            num_params = 3 * K
            aic = N_data * np.log(rss / N_data) + 2 * num_params
            
            if aic < best_aic:
                best_aic = aic
                best_K = K
                best_fit_results = popt
                
        except RuntimeError:
            continue

    components = []
    for k in range(best_K):
        A_est = best_fit_results[3*k]
        lbda_est = best_fit_results[3*k + 1]
        f0_est = best_fit_results[3*k + 2]
        
        components.append({
            'component': k + 1,
            'A_est (c*sigma)': A_est,
            'lambda_est': lbda_est,
            'f0_est_Hz': f0_est
        })

    return best_K, components, best_fit_results