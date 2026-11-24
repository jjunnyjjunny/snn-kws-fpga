"""quick FFT self-test: integer radix-2 vs numpy float FFT (SNR)."""
import sys, numpy as np
sys.path.insert(0, __file__.rsplit('/snn/', 1)[0])
from snn.frontend import fe_config as C
from snn.frontend.frontend_ref import fft512_int, make_twiddle_q15

def snr_db(ref, est):
    return 10 * np.log10(np.sum(np.abs(ref) ** 2) / max(np.sum(np.abs(ref - est) ** 2), 1e-30))

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    for db in (16, 18):
        amp = (1 << (db - 1)) - 1
        tw = make_twiddle_q15()
        for name, x in [('white', rng.uniform(-1, 1, (64, 512))),
                        ('sine', np.sin(2 * np.pi * 1000 / 16000 * np.arange(512))[None] * np.ones((4, 1))),
                        ('quiet -40dB', rng.uniform(-1, 1, (64, 512)) * 0.01)]:
            xi = np.round(x * amp).astype(np.int64)
            Xr, Xi = fft512_int(xi, None, *tw, data_bits=db)
            ref = np.fft.fft(xi.astype(np.float64), axis=-1) / 512
            est = Xr + 1j * Xi
            print(f"DB={db} {name:12s} SNR={snr_db(ref, est):6.2f} dB  max|ref|={np.abs(ref).max():.1f}")
