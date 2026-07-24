import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve

# ==============================================================================
# 1. ADVANCED SMOOTHING METHODS
# ==============================================================================

def smooth_savitzky_golay(f_grid, psd_emp, window_hz=1.0, poly_order=3):
    df = f_grid[1] - f_grid[0]
    window_pts = int(window_hz / df)
    if window_pts % 2 == 0:
        window_pts += 1
    window_pts = max(window_pts, poly_order + 2)
    
    log_emp = np.log10(psd_emp)
    log_smoothed = savgol_filter(log_emp, window_length=window_pts, polyorder=poly_order)
    return 10**log_smoothed

def smooth_adaptive_whittaker(psd_emp, lam_base=1e4, p=0.01, niter=10):
    log_emp = np.log10(psd_emp)
    L = len(log_emp)
    
    D = diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    D = csc_matrix(D)
    
    d2 = np.abs(log_emp[:-2] - 2 * log_emp[1:-1] + log_emp[2:])
    d2_norm = d2 / (np.max(d2) + 1e-12)
    
    lam_vector = lam_base / (1.0 + 50.0 * d2_norm)
    LAM = diags([lam_vector], [0], shape=(L - 2, L - 2))
    
    w = np.ones(L)
    for _ in range(niter):
        W = diags([w], [0], shape=(L, L))
        Z = W + D.T * LAM * D
        z = spsolve(Z, w * log_emp)
        w = p * (log_emp > z) + (1 - p) * (log_emp <= z)
        
    return 10**z

def smooth_log_octave(f_grid, psd_emp, fraction=1/12):
    log_emp = np.log10(psd_emp)
    valid_mask = f_grid > 0
    f_val = f_grid[valid_mask]
    log_val = log_emp[valid_mask]
    
    log_f = np.log10(f_val)
    log_f_uniform = np.linspace(log_f[0], log_f[-1], len(log_f))
    
    interp_func = interp1d(log_f, log_val, kind='linear', fill_value="extrapolate")
    log_val_uniform = interp_func(log_f_uniform)
    
    df_log = log_f_uniform[1] - log_f_uniform[0]
    win_pts = max(3, int(fraction / df_log))
    if win_pts % 2 == 0:
        win_pts += 1
        
    smoothed_uniform = savgol_filter(log_val_uniform, window_length=win_pts, polyorder=1)
    
    back_func = interp1d(log_f_uniform, smoothed_uniform, kind='linear', fill_value="extrapolate")
    log_smoothed = np.copy(log_emp)
    log_smoothed[valid_mask] = back_func(log_f)
    
    return 10**log_smoothed