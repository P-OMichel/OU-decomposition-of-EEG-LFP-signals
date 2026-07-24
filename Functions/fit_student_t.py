import numpy as np
import scipy.signal as signal
from scipy.optimize import curve_fit


# ---------------------------------------------------------
# 1. Student's t / Generalized Lorentzian PSD Model
# ---------------------------------------------------------
def single_student_t_psd(f, A, gamma, f0, alpha):
    """
    One-sided double Student's t / Generalized Lorentzian spectrum.
    Asymptotic tail decay scales as 1 / f^alpha.
    """
    # Double-sided peak formulation (positive and negative frequencies)
    term_pos = 1.0 + ((f - f0) / gamma)**2
    term_neg = 1.0 + ((f + f0) / gamma)**2
    
    return (A**2 / np.pi) * (
        (1.0 / term_pos**(alpha / 2.0)) + 
        (1.0 / term_neg**(alpha / 2.0))
    )

def multi_student_t_psd(f, *params):
    """
    Sum of K Student's t components.
    params layout: 
    [A_0, gamma_0, f0_0, alpha_0, ..., A_K, gamma_K, f0_K, alpha_K]
    """
    psd = np.zeros_like(f)
    num_components = len(params) // 4
    for i in range(num_components):
        A = params[4*i]
        gamma = params[4*i + 1]
        f0 = params[4*i + 2]
        alpha = params[4*i + 3]
        psd += single_student_t_psd(f, A, gamma, f0, alpha)
    return psd

def multi_student_t_log_psd(f, *params):
    """Log-transformed PSD for curve_fit to treat small/large peaks equally."""
    return np.log(multi_student_t_psd(f, *params) + 1e-12)

# ---------------------------------------------------------
# 2. Inverse Fitting Pipeline
# ---------------------------------------------------------
def fit_student_t_mixture_psd(f_emp, psd_emp, prominence = 1, max_components=5):
    """
    Fits empirical PSD to find optimal number K of Student's t / Generalized
    Lorentzian components and their parameters (A_i, gamma_i, f0_i, alpha_i).
    """
    # 1. Peak Detection on log-scale
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
    
    # Sort peaks by power descending
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
                A_guess = np.sqrt(psd_emp[peaks[sorted_idx[k]]] * np.pi * 0.5)
            else:
                f0_guess = (k + 1) * (f_emp[-1] / (K + 1))
                A_guess = np.sqrt(np.mean(psd_emp) * np.pi * 0.5)
                
            gamma_guess = 1.0   # Initial bandwidth guess
            alpha_guess = 2.0   # Default to standard Lorentzian tail exponent (2.0)
            
            p0.extend([A_guess, gamma_guess, f0_guess, alpha_guess])
            
            # Parameter bounds:
            # A >= 0, gamma > 0, 0 <= f0 <= f_max
            # alpha between 0.2 (extremely heavy tail) and 6.0 (steeper decay)
            bounds_lower.extend([0.0, 1e-3, 0.0, 0.2])
            bounds_upper.extend([np.inf, 100.0, f_emp[-1], 6.0])

        try:
            # Fit on LOG-PSD
            popt, _ = curve_fit(
                multi_student_t_log_psd, 
                f_emp, 
                np.log(psd_emp + 1e-12), 
                p0=p0, 
                bounds=(bounds_lower, bounds_upper),
                maxfev=30000
            )
            
            # Residual sum of squares in log domain
            residuals = np.log(psd_emp + 1e-12) - multi_student_t_log_psd(f_emp, *popt)
            rss = np.sum(residuals**2)
            
            # AIC with 4 parameters per component (4 * K)
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
        A_est = best_fit_results[4*k]
        gamma_est = best_fit_results[4*k + 1]
        f0_est = best_fit_results[4*k + 2]
        alpha_est = best_fit_results[4*k + 3]
        
        components.append({
            'component': k + 1,
            'A_est': A_est,
            'gamma_est (width)': gamma_est,
            'f0_est_Hz': f0_est,
            'alpha_est (tail exponent)': alpha_est
        })

    return best_K, components, best_fit_results



#==================================================================================#
#                           two Student-t for one peak                             #
#==================================================================================#

def dual_student_t_peak_psd(f, A_sharp, gamma_sharp, alpha_sharp,
                              A_broad, gamma_broad, alpha_broad, f0):
    """
    Combines a sharp central Student's t core with a broad Student's t wing 
    sharing the exact same center frequency f0.
    """
    psd_sharp = single_student_t_psd(f, A_sharp, gamma_sharp, f0, alpha_sharp)
    psd_broad = single_student_t_psd(f, A_broad, gamma_broad, f0, alpha_broad)
    return psd_sharp + psd_broad

def multi_dual_student_t_psd(f, *params):
    """
    params layout for K dual peaks:
    [A_s0, g_s0, a_s0, A_b0, g_b0, a_b0, f0_0, ..., A_sK, g_sK, a_sK, A_bK, g_bK, a_bK, f0_K]
    7 parameters per peak structure.
    """
    psd = np.zeros_like(f)
    num_peaks = len(params) // 7
    for i in range(num_peaks):
        A_s, g_s, a_s = params[7*i : 7*i + 3]
        A_b, g_b, a_b = params[7*i + 3 : 7*i + 6]
        f0 = params[7*i + 6]
        psd += dual_student_t_peak_psd(f, A_s, g_s, a_s, A_b, g_b, a_b, f0)
    return psd

def multi_dual_student_t_log_psd(f, *params):
    return np.log(multi_dual_student_t_psd(f, *params) + 1e-12)

# ---------------------------------------------------------
# 2. Dual-Component Fitting Pipeline
# ---------------------------------------------------------
def fit_dual_student_t_mixture_psd(f_emp, psd_emp, prominence=1.0, max_components=5):
    """
    Fits Dual Student's t mixture (sharp core + broad decay tail per peak frequency).
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
                pwr = psd_emp[peaks[sorted_idx[k]]]
            else:
                f0_guess = (k + 1) * (f_emp[-1] / (K + 1))
                pwr = np.mean(psd_emp)
                
            A_sharp_guess = np.sqrt(pwr * 0.7 * np.pi * 0.5)
            gamma_sharp_guess = 0.5
            alpha_sharp_guess = 3.0
            
            A_broad_guess = np.sqrt(pwr * 0.3 * np.pi * 0.5)
            gamma_broad_guess = 4.0
            alpha_broad_guess = 1.5
            
            p0.extend([A_sharp_guess, gamma_sharp_guess, alpha_sharp_guess,
                        A_broad_guess, gamma_broad_guess, alpha_broad_guess, f0_guess])
            
            # Enforce gamma_sharp <= 2.0 Hz and gamma_broad >= 2.0 Hz for separation
            bounds_lower.extend([0.0, 1e-3, 0.2,  0.0, 2.0, 0.2, 0.0])
            bounds_upper.extend([np.inf, 2.0, 6.0, np.inf, 50.0, 6.0, f_emp[-1]])

        try:
            popt, _ = curve_fit(
                multi_dual_student_t_log_psd, 
                f_emp, 
                np.log(psd_emp + 1e-12), 
                p0=p0, 
                bounds=(bounds_lower, bounds_upper),
                maxfev=5000
            )
            
            residuals = np.log(psd_emp + 1e-12) - multi_dual_student_t_log_psd(f_emp, *popt)
            rss = np.sum(residuals**2)
            
            num_params = 7 * K
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
            'A_sharp_est': best_fit_results[7*k],
            'gamma_sharp_est': best_fit_results[7*k + 1],
            'alpha_sharp_est': best_fit_results[7*k + 2],
            'A_broad_est': best_fit_results[7*k + 3],
            'gamma_broad_est': best_fit_results[7*k + 4],
            'alpha_broad_est': best_fit_results[7*k + 5],
            'f0_est_Hz': best_fit_results[7*k + 6]
        })

    return best_K, components, best_fit_results