"""
float_ref.py — floating-point reference log-mel with the SAME definition as the
integer model (periodic Hann, 512-pt FFT scaled by 1/512, peak-normalised Slaney
mel triangles, 16*log2). Used only for verification / error reporting.
"""
import numpy as np
from . import fe_config as C
from .frontend_ref import make_mel_float, frame

def logmel_float(pcm24, n_mels=C.N_MELS, win=C.WIN, hop=C.HOP, data_bits=C.FFT_DATA_BITS,
                 frac_bits=C.LOG_FRAC_BITS, return_all=False):
    """pcm24 [..., n] -> log2-domain features [..., T, n_mels] in Q.frac units (float),
    on the same numeric scale as frontend_ref.log2_int output (before offset/clip)."""
    x = np.asarray(pcm24, dtype=np.float64) / (1 << (C.PCM_BITS - data_bits))   # DB-bit units
    fr = frame(x, win, hop)
    n = np.arange(win)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * n / win)
    xw = fr * w
    X = np.fft.rfft(xw, n=C.N_FFT, axis=-1) / C.N_FFT
    P = (X.real ** 2 + X.imag ** 2)
    fb = make_mel_float(n_mels) * ((1 << C.MEL_W_BITS) - 1)   # same 0..255 weight scale
    M = P @ fb.T
    L = (1 << frac_bits) * np.log2(np.maximum(M, 1e-30))
    if return_all:
        return dict(frames=fr, windowed=xw, fft=X, power=P, mel=M, log2q=L)
    return L
