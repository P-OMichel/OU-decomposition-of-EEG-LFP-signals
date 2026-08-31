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

# Frequency range spanning from low frequencies to the far tail
f = np.logspace(-1, 3, 2000)  # 0.1 Hz to 1000 Hz

# Parameters
A = 15.0
f0 = 20.0
beta_decay = 2.0

b_amp = 200.0
chi = 1.5
f_knee = 10.0

# Evaluate formulas
y_tail = p_tail(f, A=A, f0=f0, beta_decay=beta_decay)
y_no_knee = p_bg_without_knee(f, b_amp=b_amp, chi=chi)
y_with_knee = p_bg_with_knee(f, b_amp=b_amp, f_knee=f_knee, chi=chi)

# Mask zero values for clean log-scale display
y_tail_masked = np.where(y_tail > 0, y_tail, np.nan)

# Create Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 1. Log-Log Spectrum
ax = axes[0]
ax.plot(f, y_no_knee, label=r'$P_{\mathrm{bg}}$ without knee: $\frac{b_{\mathrm{amp}}}{f^\chi}$', 
        color='#d95f02', linestyle='--', linewidth=2)
ax.plot(f, y_with_knee, label=r'$P_{\mathrm{bg}}$ with knee: $\frac{b_{\mathrm{amp}}}{f_{\mathrm{knee}}^\chi + f^\chi}$', 
        color='#1b9e77', linewidth=2)
ax.plot(f, y_tail_masked, label=r'$P_{\mathrm{tail}}$: Onset at $f_0$', 
        color='#7570b3', linewidth=2)

ax.axvline(f0, color='gray', linestyle=':', alpha=0.7, label=r'Threshold $f_0$')
ax.axvline(f_knee, color='gray', linestyle='-.', alpha=0.7, label=r'Knee $f_{\mathrm{knee}}$')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('Power Spectrum (Log-Log Scale)', fontsize=13)
ax.set_xlabel('Frequency $f$ (Hz)', fontsize=11)
ax.set_ylabel('Power Spectral Density $P(f)$', fontsize=11)
ax.set_ylim(1e-4, 5e3)
ax.grid(True, which='both', linestyle='--', alpha=0.4)
ax.legend(loc='lower left', fontsize=9.5)

# 2. Linear Scale Focus (around the onset/knee region)
ax2 = axes[1]
f_lin = np.linspace(0.1, 100, 1000)


ax2.plot(f_lin, p_bg_without_knee(f_lin, b_amp, chi), 
         label=r'$P_{\mathrm{bg}}$ with knee', color='#1b9e77', linewidth=2)
ax2.plot(f_lin, p_bg_with_knee(f_lin, b_amp, f_knee, chi), 
         label=r'$P_{\mathrm{bg}}$ with knee', color='#1b9e77', linewidth=2)
ax2.plot(f_lin, p_tail(f_lin, A, f0, beta_decay), 
         label=r'$P_{\mathrm{tail}}$', color='#7570b3', linewidth=2)

ax2.set_title('Linear Scale (Peak & Transition Focus)', fontsize=13)
ax2.set_xlabel('Frequency $f$ (Hz)', fontsize=11)
ax2.set_ylabel('Power $P(f)$', fontsize=11)
ax2.set_xlim(0, 100)
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()