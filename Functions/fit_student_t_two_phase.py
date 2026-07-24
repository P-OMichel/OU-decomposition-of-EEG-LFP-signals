import numpy as np
import scipy.signal as signal
from scipy.optimize import least_squares


# ---------------------------------------------------------
# 1. Student's t / Generalized Lorentzian PSD Model
# ---------------------------------------------------------
def single_student_t_psd(f, A, gamma, f0, alpha):
    """
    One-sided double Student's t / Generalized Lorentzian spectrum.
    Asymptotic tail decay scales as 1 / f^alpha.
    """
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
    """Log-transformed PSD for least_squares to treat small/large peaks equally."""
    return np.log(multi_student_t_psd(f, *params) + 1e-12)


# ---------------------------------------------------------
# 2. Two-Phase Decoupled Inverse Fitting Pipeline
# ---------------------------------------------------------
def fit_student_t_mixture_psd(
    f_emp, 
    psd_emp, 
    prominence=1, 
    max_components=5, 
    criterion='aic'  # Select 'aic' or 'bic'
):
    """
    Fits empirical PSD to find optimal number K of Student's t components.
    
    Uses Two-Phase Decoupling:
    - Phase 1: Rapid search over K with fixed alpha = 2.0 (standard Lorentzian)
    - Phase 2: Refines all parameters (including tail exponent alpha) only on winning K.
    
    Parameters
    ----------
    criterion : str, optional ('aic' or 'bic')
        Model selection criterion. 'aic' retains more components for tighter fits,
        while 'bic' penalizes extra components more strictly.
    """
    log_psd_emp = np.log(psd_emp + 1e-12)
    N_data = len(f_emp)
    
    # 1. Peak Detection on log scale
    peaks, _ = signal.find_peaks(
        log_psd_emp, 
        prominence=prominence, 
        distance=max(1, int(N_data * 0.02))
    )
    
    if len(peaks) == 0:
        peaks = [np.argmax(psd_emp)]
        
    detected_freqs = f_emp[peaks]
    detected_powers = psd_emp[peaks]
    sorted_idx = np.argsort(detected_powers)[::-1]
    detected_freqs = detected_freqs[sorted_idx]
    
    max_K = min(max_components, max(len(detected_freqs), max_components))
    
    best_score = np.inf
    best_phase1_results = None
    best_K = 0

    # =========================================================
    # PHASE 1: Model Selection with Fixed alpha = 2.0
    # =========================================================
    for K in range(1, max_K + 1):
        p0_phase1 = []
        bounds_lower_phase1 = []
        bounds_upper_phase1 = []
        
        for k in range(K):
            if k < len(detected_freqs):
                f0_guess = detected_freqs[k]
                A_guess = np.sqrt(psd_emp[peaks[sorted_idx[k]]] * np.pi * 0.5)
            else:
                f0_guess = (k + 1) * (f_emp[-1] / (K + 1))
                A_guess = np.sqrt(np.mean(psd_emp) * np.pi * 0.5)
                
            gamma_guess = 1.0
            
            p0_phase1.extend([A_guess, gamma_guess, f0_guess])
            bounds_lower_phase1.extend([0.0, 1e-3, 0.0])
            bounds_upper_phase1.extend([np.inf, 100.0, f_emp[-1]])

        def phase1_residuals(params):
            full_params = []
            for i in range(K):
                full_params.extend([params[3*i], params[3*i + 1], params[3*i + 2], 2.0])
            return log_psd_emp - multi_student_t_log_psd(f_emp, *full_params)

        try:
            res_p1 = least_squares(
                phase1_residuals, 
                p0_phase1, 
                bounds=(bounds_lower_phase1, bounds_upper_phase1),
                method='trf', 
                ftol=1e-5
            )
            
            rss = np.sum(res_p1.fun**2)
            num_params = 3 * K
            
            # --- Model Selection Criterion Switch ---
            if criterion.lower() == 'aic':
                # AIC Penalty = 2 * num_params
                score = N_data * np.log(rss / N_data) + 2 * num_params
            elif criterion.lower() == 'bic':
                # BIC Penalty = num_params * ln(N)
                score = N_data * np.log(rss / N_data) + num_params * np.log(N_data)
            else:
                raise ValueError("criterion must be either 'aic' or 'bic'")
            
            if score < best_score:
                best_score = score
                best_K = K
                
                best_phase1_results = []
                for i in range(K):
                    best_phase1_results.extend([
                        res_p1.x[3*i], 
                        res_p1.x[3*i + 1], 
                        res_p1.x[3*i + 2], 
                        2.0
                    ])
                    
        except Exception:
            continue

    if best_K == 0 or best_phase1_results is None:
        return 0, [], None

    # =========================================================
    # PHASE 2: Tail Exponent Refinement on Best K Only
    # =========================================================
    bounds_lower_phase2 = []
    bounds_upper_phase2 = []
    
    for k in range(best_K):
        bounds_lower_phase2.extend([0.0, 1e-3, 0.0, 0.2])
        bounds_upper_phase2.extend([np.inf, 100.0, f_emp[-1], 6.0])

    def phase2_residuals(params):
        return log_psd_emp - multi_student_t_log_psd(f_emp, *params)

    res_p2 = least_squares(
        phase2_residuals,
        best_phase1_results,
        bounds=(bounds_lower_phase2, bounds_upper_phase2),
        method='trf',
        ftol=1e-6
    )
    
    best_fit_results = res_p2.x

    components = []
    for k in range(best_K):
        components.append({
            'component': k + 1,
            'A_est': best_fit_results[4*k],
            'gamma_est (width)': best_fit_results[4*k + 1],
            'f0_est_Hz': best_fit_results[4*k + 2],
            'alpha_est (tail exponent)': best_fit_results[4*k + 3]
        })

    return best_K, components, best_fit_results


def fit_student_t_mixture_psd_1(
    f_emp, 
    psd_emp, 
    prominence=1, 
    max_components=5, 
    criterion='aic',
    peak_weight_boost=5.0
):
    log_psd_emp = np.log(psd_emp + 1e-12)
    N_data = len(f_emp)
    
    # 1. Peak Detection
    peaks, _ = signal.find_peaks(
        log_psd_emp, 
        prominence=prominence, 
        distance=max(1, int(N_data * 0.02))
    )
    if len(peaks) == 0:
        peaks = [np.argmax(psd_emp)]
        
    detected_freqs = f_emp[peaks]
    detected_powers = psd_emp[peaks]
    sorted_idx = np.argsort(detected_powers)[::-1]
    detected_freqs = detected_freqs[sorted_idx]
    
    max_K = min(max_components, max(len(detected_freqs), max_components))
    
    # 2. Build Peak-Weighted Vector
    weights = np.ones_like(psd_emp)
    norm_psd = (psd_emp - np.min(psd_emp)) / (np.max(psd_emp) - np.min(psd_emp) + 1e-12)
    weights += (peak_weight_boost - 1.0) * norm_psd
    
    d2_log_psd = -np.gradient(np.gradient(log_psd_emp))
    d2_log_psd = np.clip(d2_log_psd, 0, None)
    if np.max(d2_log_psd) > 0:
        weights += (peak_weight_boost * 0.5) * (d2_log_psd / np.max(d2_log_psd))

    best_score = np.inf
    best_phase1_results = None
    best_K = 0

    # =========================================================
    # PHASE 1: Fast Weighted Geometry Search (Fixed alpha = 2.0)
    # =========================================================
    for K in range(1, max_K + 1):
        p0_p1, lower_p1, upper_p1 = [], [], []
        for k in range(K):
            if k < len(detected_freqs):
                f0_g, pwr = detected_freqs[k], psd_emp[peaks[sorted_idx[k]]]
            else:
                f0_g, pwr = (k + 1) * (f_emp[-1] / (K + 1)), np.mean(psd_emp)
                
            p0_p1.extend([np.sqrt(pwr * np.pi * 0.5), 1.0, f0_g])
            lower_p1.extend([0.0, 1e-3, 0.0])
            upper_p1.extend([np.inf, 100.0, f_emp[-1]])

        def weighted_p1_residuals(params):
            full_params = []
            for i in range(K):
                full_params.extend([params[3*i], params[3*i+1], params[3*i+2], 2.0])
            return weights * (log_psd_emp - multi_student_t_log_psd(f_emp, *full_params))

        try:
            res_p1 = least_squares(weighted_p1_residuals, p0_p1, bounds=(lower_p1, upper_p1), method='trf', ftol=1e-5)
            
            # Evaluate unweighted RSS for model selection
            unweighted_res = log_psd_emp - multi_student_t_log_psd(
                f_emp, *[val for i in range(K) for val in (res_p1.x[3*i], res_p1.x[3*i+1], res_p1.x[3*i+2], 2.0)]
            )
            rss = np.sum(unweighted_res**2)
            num_params = 3 * K
            
            score = (N_data * np.log(rss / N_data) + 2 * num_params) if criterion.lower() == 'aic' else (N_data * np.log(rss / N_data) + num_params * np.log(N_data))
            
            if score < best_score:
                best_score = score
                best_K = K
                best_phase1_results = []
                for i in range(K):
                    best_phase1_results.extend([res_p1.x[3*i], res_p1.x[3*i+1], res_p1.x[3*i+2], 2.0])
        except Exception:
            continue

    if best_K == 0 or best_phase1_results is None:
        return 0, [], None

    # =========================================================
    # PHASE 2: Weighted Tail Refinement on Winning K Only
    # =========================================================
    lower_p2, upper_p2 = [], []
    for k in range(best_K):
        lower_p2.extend([0.0, 1e-3, 0.0, 0.2])
        upper_p2.extend([np.inf, 100.0, f_emp[-1], 6.0])

    def weighted_p2_residuals(params):
        return weights * (log_psd_emp - multi_student_t_log_psd(f_emp, *params))

    res_p2 = least_squares(weighted_p2_residuals, best_phase1_results, bounds=(lower_p2, upper_p2), method='trf', ftol=1e-7)
    best_fit_results = res_p2.x

    components = []
    for k in range(best_K):
        components.append({
            'component': k + 1,
            'A_est': best_fit_results[4*k],
            'gamma_est (width)': best_fit_results[4*k + 1],
            'f0_est_Hz': best_fit_results[4*k + 2],
            'alpha_est (tail exponent)': best_fit_results[4*k + 3]
        })

    return best_K, components, best_fit_results