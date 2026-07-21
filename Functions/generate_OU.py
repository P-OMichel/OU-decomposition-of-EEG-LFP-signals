import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def get_OU_signal(T, dt, lbda, omega, sigma):
    """
    Generates a 2D Ornstein-Uhlenbeck signal using the unconditionally stable 
    Implicit Backward Euler scheme.
    """
    N = int(T / dt)          # Number of time steps
    t = np.linspace(0, T, N)
    x = np.zeros(N)
    y = np.zeros(N)

    x[0] = 0.0
    y[0] = 0.0    

    A = np.array([[-lbda,  omega],
                  [-omega, -lbda]])
    I = np.eye(2)
    implicit_operator = np.linalg.inv(I - A * dt)

    for i in range(1, N):
        dW_x = np.random.normal(0, np.sqrt(dt))
        dW_y = np.random.normal(0, np.sqrt(dt))
        
        current_state_with_noise = np.array([
            x[i-1] + np.sqrt(2) * sigma * dW_x,
            y[i-1] + np.sqrt(2) * sigma * dW_y
        ])
        
        next_state = implicit_operator @ current_state_with_noise
        
        x[i] = next_state[0]
        y[i] = next_state[1]

    return t, x


def get_mixed_OU_signals(T, dt, lbda_list, omega_list, sigma_list, factor_list):
    """
    Generates mixed OU signals and extracts ground-truth peak segmentation limits
    empirically from the computed PSD to account for Welch window smearing.
    """
    N = int(T / dt)
    t = np.linspace(0, T, N)

    mixed_OU = np.zeros(N)

    # 1. Generate and sum the OU signals
    for i in range(len(lbda_list)):
        x = get_OU_signal(T, dt, lbda_list[i], omega_list[i], sigma_list[i])[-1]
        mixed_OU += x * factor_list[i]

    return t, mixed_OU

def get_mixed_OU_signals_psd_peaks(T, dt, lbda_list, omega_list, sigma_list, factor_list):
    """
    Generates mixed OU signals and extracts ground-truth peak segmentation limits
    empirically from the computed PSD to account for Welch window smearing.
    """
    N = int(T / dt)
    t = np.linspace(0, T, N)
    fs = 1.0 / dt

    mixed_OU = np.zeros(N)

    # 1. Generate and sum the OU signals
    for i in range(len(lbda_list)):
        x = get_OU_signal(T, dt, lbda_list[i], omega_list[i], sigma_list[i])[-1]
        mixed_OU += x * factor_list[i]

    # 2. Compute the empirical PSD of the mixture
    f, psd = signal.welch(mixed_OU, fs=fs, nperseg=int(fs * 2))

    peak_limits = []

    # 3. Empirically find the boundaries for each peak on the calculated PSD
    for i in range(len(omega_list)):
        f_target = omega_list[i] / (2 * np.pi)
        
        # Find the index in the frequency array 'f' closest to our theoretical target
        target_idx = np.argmin(np.abs(f - f_target))
        
        # Find the local peak maximum around the target frequency (within a +/- 2 Hz buffer)
        search_radius = int(2.0 / (f[1] - f[0])) 
        start_search = max(0, target_idx - search_radius)
        end_search = min(len(f), target_idx + search_radius)
        
        local_peak_idx = start_search + np.argmax(psd[start_search:end_search])
        peak_max_power = psd[local_peak_idx]
        half_max_power = peak_max_power / 2.0
        
        # Scan left from the peak to find where it drops below half-maximum
        left_idx = local_peak_idx
        while left_idx > 0 and psd[left_idx] > half_max_power:
            left_idx -= 1
            
        # Scan right from the peak to find where it drops below half-maximum
        right_idx = local_peak_idx
        while right_idx < len(f) - 1 and psd[right_idx] > half_max_power:
            right_idx += 1
            
        peak_limits.append({
            'peak_index': i,
            'center_freq': f[local_peak_idx],
            'left_bound': f[left_idx],
            'right_bound': f[right_idx],
            'empirical_fwhm': f[right_idx] - f[left_idx]
        })

    return t, mixed_OU, f, psd, peak_limits


def get_analytical_psd(N, f_stop, lbda_list, omega_list, sigma_list, factor_list):
    f = np.linspace(0, f_stop, N)
    psd = np.zeros(N)

    for i in range(len(lbda_list)):
        lbda = lbda_list[i]
        omega = omega_list[i]
        sigma = sigma_list[i]
        c = factor_list[i]  # Linear factor

        # Conversion to continuous-time angular frequency
        Omega = 2 * np.pi * f
        
        # Single double-sided lorentzian spectrum for s_1(t)
        S_two_sided = (sigma**2) * (
            1.0 / (lbda**2 + (Omega - omega)**2) + 
            1.0 / (lbda**2 + (Omega + omega)**2)
        )
        
        # Multiply by 2 for one-sided PSD (Welch standard), and by factor^2 for signal scaling
        psd += (c**2) * 2 * S_two_sided
    
    return f, psd


import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def get_OU_signal_exact(T, dt, lbda, omega, sigma):
    """
    Generates a 2D Ornstein-Uhlenbeck signal using the exact discrete-time step 
    (matrix exponential solution) with zero numerical damping.
    """
    N = int(T / dt)
    t = np.linspace(0, T, N)
    
    x = np.zeros(N)
    y = np.zeros(N)

    # 1. Exact transition matrix M = exp(A * dt)
    decay = np.exp(-lbda * dt)
    cos_w = np.cos(omega * dt)
    sin_w = np.sin(omega * dt)
    
    M = decay * np.array([
        [ cos_w,  sin_w],
        [-sin_w,  cos_w]
    ])

    # 2. Exact standard deviation for discrete noise step
    sigma_eta = sigma * np.sqrt((1.0 - np.exp(-2.0 * lbda * dt)) / lbda)

    # 3. Simulate exact step
    for i in range(1, N):
        # Generate independent Gaussian noise
        eta_x = np.random.normal(0, sigma_eta)
        eta_y = np.random.normal(0, sigma_eta)
        
        # Matrix multiplication for deterministic state transition
        prev_state = np.array([x[i-1], y[i-1]])
        next_state = M @ prev_state + np.array([eta_x, eta_y])
        
        x[i] = next_state[0]
        y[i] = next_state[1]

    return t, x

def get_mixed_OU_signals_exact(T, dt, lbda_list, omega_list, sigma_list, factor_list):
    """
    Generates mixed OU signals using the exact solver.
    """
    N = int(T / dt)
    t = np.linspace(0, T, N)
    mixed_OU = np.zeros(N)

    for i in range(len(lbda_list)):
        _, x = get_OU_signal_exact(T, dt, lbda_list[i], omega_list[i], sigma_list[i])
        mixed_OU += x * factor_list[i]

    return t, mixed_OU