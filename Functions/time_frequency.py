import numpy as np
import scipy as sc

def spectrogram(y,fs,nperseg_factor=1,noverlap_factor=0.9,nfft_factor=1,detrend=False,scaling="psd",f_cut=45):
    nperseg = int(nperseg_factor * fs)
    noverlap = int(noverlap_factor * nperseg)
    nfft = int(nfft_factor * nperseg)
    window = sc.signal.windows.hamming(nperseg, sym=True)

    f_spectro, t_spectro, stft = sc.signal.stft(y,fs=fs,window=window,nperseg=nperseg,noverlap=noverlap,nfft=nfft,detrend=detrend,scaling=scaling)

    Sxx = np.abs(stft) ** 2

    if len(f_spectro) > 1:
        df = f_spectro[1] - f_spectro[0]
        j = int(f_cut / df)
        j = max(1, min(j, len(f_spectro)))
        f_spectro = f_spectro[:j]
        Sxx = Sxx[:j, :]

    return f_spectro, t_spectro, Sxx
