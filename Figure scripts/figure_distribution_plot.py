import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Parameter Definitions & Setup
# -----------------------------------------------------------------------------
f_0 = 100.0       # Peak center frequency (Hz)
gamma = 15.0      # Half-width at half-maximum (HWHM) parameter
A = 1.0           # Amplitude scaling factor

f = np.logspace(0, 3, 2000)

# -----------------------------------------------------------------------------
# 2. Mathematical Model Functions
# -----------------------------------------------------------------------------
def pseudo_voigt(f, f_0, gamma, A, eta=0.5):
    sigma = gamma / np.sqrt(2 * np.log(2))
    L = lambda df: (1 / np.pi) * (gamma / (df**2 + gamma**2))
    G = lambda df: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (df / sigma)**2)
    df_pos, df_neg = f - f_0, f + f_0
    return (A**2) * (eta * L(df_pos) + (1 - eta) * G(df_pos) + eta * L(df_neg) + (1 - eta) * G(df_neg))

def ornstein_uhlenbeck(f, f_0, gamma, A):
    Omega, omega_0 = 2 * np.pi * f, 2 * np.pi * f_0
    lam = max(0.1, gamma)
    pos = 1 / (lam**2 + (Omega - omega_0)**2)
    neg = 1 / (lam**2 + (Omega + omega_0)**2)
    return (A**2 / np.pi) * (pos + neg)

def pearson_iv(f, f_0, gamma, A, nu_p=1.2, m=1.5):
    peak = lambda df: np.exp(-nu_p * np.arctan(df / gamma)) / ((1 + (df / gamma)**2)**m)
    return (A**2) * (peak(f - f_0) + peak(f + f_0))

def student_t(f, f_0, gamma, A, alpha=2.0):
    pos = 1 / (1 + ((f - f_0) / gamma)**2)**(alpha / 2)
    neg = 1 / (1 + ((f + f_0) / gamma)**2)**(alpha / 2)
    return (A**2 / np.pi) * (pos + neg)

# Compute & Normalize
p_voigt = pseudo_voigt(f, f_0, gamma, A, eta=0.5)
p_ou = ornstein_uhlenbeck(f, f_0, gamma, A)
p_pearson = pearson_iv(f, f_0, gamma, A, nu_p=1.2, m=1.5)
p_student = student_t(f, f_0, gamma, A, alpha=2.0)

p_voigt /= np.max(p_voigt)
p_ou /= np.max(p_ou)
p_pearson /= np.max(p_pearson)
p_student /= np.max(p_student)

# -----------------------------------------------------------------------------
# 3. Compact & Responsive Layout Settings
# -----------------------------------------------------------------------------
# Reset default rcParams to avoid lingering large-font overrides
plt.rcdefaults()

# Create figure using constrained layout engine
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2), dpi=300, constrained_layout=True)

FONT_SIZE = 7

colors = {
    'Pseudo-Voigt': '#1f77b4',
    'Ornstein-Uhlenbeck': '#ff7f0e',
    'Pearson IV': '#2ca02c',
    'Student-t': '#d62728'
}

models = [
    ('Pseudo-Voigt ($\\eta=0.5$)', p_voigt, colors['Pseudo-Voigt'], '-'),
    ('Ornstein-Uhlenbeck', p_ou, colors['Ornstein-Uhlenbeck'], '--'),
    ('Pearson IV ($\\nu_p=1.2, m=1.5$)', p_pearson, colors['Pearson IV'], '-.'),
    ('Student-$t$ ($\\alpha=2.0$)', p_student, colors['Student-t'], ':')
]

# Panel A: Log-Log Scale
for label, data, color, ls in models:
    ax1.loglog(f, data, label=label, color=color, linestyle=ls, linewidth=1.2)

ax1.axvline(f_0, color='gray', linestyle=':', alpha=0.6, linewidth=1.0, label=f'$f_0 = {f_0:.0f}$ Hz')
ax1.set_xlabel('Frequency $f$ (Hz)', fontsize=FONT_SIZE)
ax1.set_ylabel('Normalized PSD $P_{\\text{peak}}(f)$', fontsize=FONT_SIZE)
ax1.set_title('(a) Log-Log Scale', fontsize=FONT_SIZE + 1, pad=4)
ax1.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 1)
ax1.grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.5)
ax1.set_ylim(1e-5, 2.0)

# Move legend outside the plot area above panel A
ax1.legend(loc='lower left', frameon=True, fontsize=FONT_SIZE - 1, framealpha=0.9)

# Panel B: Semi-Log X Scale
for label, data, color, ls in models:
    ax2.semilogy(f, data, label=label, color=color, linestyle=ls, linewidth=1.2)

ax2.axvline(f_0, color='gray', linestyle=':', alpha=0.6, linewidth=1.0)
ax2.set_xlabel('Frequency $f$ (Hz)', fontsize=FONT_SIZE)
ax2.set_ylabel('Normalized PSD $P_{\\text{peak}}(f)$', fontsize=FONT_SIZE)
ax2.set_title('(b) Semi-Log Y Scale', fontsize=FONT_SIZE + 1, pad=4)
ax2.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 1)
ax2.grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.5)
ax2.set_xlim(10, 500)
ax2.set_ylim(-0.02, 1.08)

plt.show()