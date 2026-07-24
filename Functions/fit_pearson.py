import numpy as np
import scipy.signal as signal
from scipy.optimize import curve_fit

# ---------------------------------------------------------
# 1. Pearson Type IV Resonance Model
# ---------------------------------------------------------
def single_pearson_iv_psd(f, A, gamma, f0, m, nu_p):
    """
    Pearson Type IV Profile:
    - gamma: controls central peak width
    - m: tail decay exponent (asymptotic slope)
    - nu_p: tail asymmetry factor (allows slow decay on one side)
    """
    y_pos = (f - f0) / gamma
    y_neg = (f + f0) / gamma
    
    term_pos = (1.0 + y_pos**2)**(-m) * np.exp(-nu_p * np.arctan(y_pos))
    term_neg = (1.0 + y_neg**2)**(-m) * np.exp(-nu_p * np.arctan(y_neg))
    
    return A**2 * (term_pos + term_neg)

def multi_pearson_iv_psd(f, *params):
    """
    params layout: [A_0, gamma_0, f0_0, m_0, nu_p_0, ..., A_K, gamma_K, f0_K, m_K, nu_p_K]
    5 parameters per peak.
    """
    psd = np.zeros_like(f)
    num_components = len(params) // 5
    for i in range(num_components):
        A, gamma, f0, m, nu_p = params[5*i : 5*i + 5]
        psd += single_pearson_iv_psd(f, A, gamma, f0, m, nu_p)
    return psd

def multi_pearson_iv_log_psd(f, *params):
    return np.log(multi_pearson_iv_psd(f, *params)+ 1e-12)

# ---------------------------------------------------------
# 2. Pearson IV Fitting Pipeline
# ---------------------------------------------------------
def fit_pearson_iv_mixture_psd(f_emp, psd_emp, prominence=1.0, max_components=5):
    """
    Fits Pearson Type IV mixture (independent peak curvature and tail decay).
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
            m_guess = 1.5      # Asymptotic tail exponent
            nu_p_guess = 0.0   # Symmetric baseline
            
            p0.extend([A_guess, gamma_guess, f0_guess, m_guess, nu_p_guess])
            
            bounds_lower.extend([0.0, 1e-3, 0.0, 0.2, -5.0])
            bounds_upper.extend([np.inf, 100.0, f_emp[-1], 6.0, 5.0])

        try:
            popt, _ = curve_fit(
                multi_pearson_iv_log_psd, 
                f_emp, 
                np.log(psd_emp + 1e-12), 
                p0=p0, 
                bounds=(bounds_lower, bounds_upper),
                maxfev=500
            )
            
            residuals = np.log(psd_emp + 1e-12) - multi_pearson_iv_log_psd(f_emp, *popt)
            rss = np.sum(residuals**2)
            
            num_params = 5 * K
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
            'A_est': best_fit_results[5*k],
            'gamma_est (width)': best_fit_results[5*k + 1],
            'f0_est_Hz': best_fit_results[5*k + 2],
            'm_est (tail exponent)': best_fit_results[5*k + 3],
            'nu_p_est (asymmetry)': best_fit_results[5*k + 4]
        })

    return best_K, components, best_fit_results