import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Define the Spectral Models
# -------------------------------------------------------------------------

def p_tail(f, A=10.0, f0=20.0, beta_decay=2.0):
    """
    Asymmetric spectral tail with smooth onset:
    P_tail(f) = 0.3 * A^2 * (f / f0)^(-beta_decay) * (1 - f0 / f) * I(f > f0)
    """
    f = np.asarray(f, dtype=float)
    out = np.zeros_like(f)
    mask = f > f0
    out[mask] = (
        0.3 * (A**2) 
        * ((f[mask] / f0) ** (-beta_decay)) 
        * (1.0 - f0 / f[mask])
    )
    return out


def p_bg_without_knee(f, b_amp=100.0, chi=1.5):
    """
    Pure scale-free power-law background:
    P_bg(f) = b_amp / (f^chi)
    """
    f = np.asarray(f, dtype=float)
    return b_amp / (f ** chi)


def p_bg_with_knee(f, b_amp=100.0, f_knee=10.0, chi=1.5):
    """
    Knee-bounded aperiodic background (Lorentzian-like / FOOOF style):
    P_bg(f) = b_amp / (f_knee^chi + f^chi)
    """
    f = np.asarray(f, dtype=float)
    return b_amp / ((f_knee ** chi) + (f ** chi))


# -------------------------------------------------------------------------
# Computation & Plotting
# -------------------------------------------------------------------------

# Global parameters
A = 15.0
f0 = 20.0
beta_decay = 2.0
b_amp = 200.0
chi = 1.5
f_knee = 10.0

# 1. Broad frequency range for Log-Log plot
f_broad = np.logspace(-1, 3, 2000)  # 0.1 Hz to 1000 Hz

y_tail_broad = p_tail(f_broad, A=A, f0=f0, beta_decay=beta_decay)
y_no_knee_broad = p_bg_without_knee(f_broad, b_amp=b_amp, chi=chi)
y_with_knee_broad = p_bg_with_knee(f_broad, b_amp=b_amp, f_knee=f_knee, chi=chi)

# Mask zero values for clean log-scale display
y_tail_broad_masked = np.where(y_tail_broad > 0, y_tail_broad, np.nan)

# 2. Local frequency range for Semi-Log / Log-Y plot (linear x-axis, log y-axis)
f_lin = np.linspace(0.5, 100, 1000)

y_tail_lin = p_tail(f_lin, A=A, f0=f0, beta_decay=beta_decay)
y_no_knee_lin = p_bg_without_knee(f_lin, b_amp=b_amp, chi=chi)
y_with_knee_lin = p_bg_with_knee(f_lin, b_amp=b_amp, f_knee=f_knee, chi=chi)

y_tail_lin_masked = np.where(y_tail_lin > 0, y_tail_lin, np.nan)

# Create Subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Subplot 1: Log-Log Scale
ax1 = axes[0]
ax1.plot(f_broad, y_no_knee_broad, 
         label=r'$P_{\mathrm{bg}}$ without knee: $\frac{b_{\mathrm{amp}}}{f^\chi}$', 
         color='#d95f02', linestyle='--', linewidth=2)
ax1.plot(f_broad, y_with_knee_broad, 
         label=r'$P_{\mathrm{bg}}$ with knee: $\frac{b_{\mathrm{amp}}}{f_{\mathrm{knee}}^\chi + f^\chi}$', 
         color='#1b9e77', linewidth=2)
ax1.plot(f_broad, y_tail_broad_masked, 
         label=r'$P_{\mathrm{tail}}$: Onset at $f_0$', 
         color='#7570b3', linewidth=2)

ax1.axvline(f0, color='gray', linestyle=':', alpha=0.7, label=r'Threshold $f_0$')
ax1.axvline(f_knee, color='gray', linestyle='-.', alpha=0.7, label=r'Knee $f_{\mathrm{knee}}$')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title('Power Spectrum (Log-Log Scale)', fontsize=13)
ax1.set_xlabel('Frequency $f$ (Hz)', fontsize=11)
ax1.set_ylabel('Power Spectral Density $P(f)$', fontsize=11)
ax1.set_ylim(1e-4, 5e3)
ax1.grid(True, which='both', linestyle='--', alpha=0.4)
ax1.legend(loc='lower left', fontsize=9.5)

# Subplot 2: Semi-Log / Log-Y Scale (Linear X, Log Y)
ax2 = axes[1]
ax2.plot(f_lin, y_no_knee_lin, 
         label=r'$P_{\mathrm{bg}}$ without knee: $\frac{b_{\mathrm{amp}}}{f^\chi}$', 
         color='#d95f02', linestyle='--', linewidth=2)
ax2.plot(f_lin, y_with_knee_lin, 
         label=r'$P_{\mathrm{bg}}$ with knee: $\frac{b_{\mathrm{amp}}}{f_{\mathrm{knee}}^\chi + f^\chi}$', 
         color='#1b9e77', linewidth=2)
ax2.plot(f_lin, y_tail_lin_masked, 
         label=r'$P_{\mathrm{tail}}$: Onset at $f_0$', 
         color='#7570b3', linewidth=2)

ax2.set_yscale('log')
ax2.set_title('Power Spectrum (Log-Y Axis Scale)', fontsize=13)
ax2.set_xlabel('Frequency $f$ (Hz)', fontsize=11)
ax2.set_ylabel('Power Spectral Density $P(f)$ (Log Scale)', fontsize=11)
ax2.set_xlim(0.5, 100)
ax2.set_ylim(1e-2, 5e2)
ax2.grid(True, which='both', linestyle='--', alpha=0.4)
ax2.legend(loc='upper right', fontsize=9.5)

plt.tight_layout()
plt.show()