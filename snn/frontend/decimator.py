"""
decimator.py — 48 kHz -> 16 kHz polyphase FIR decimator golden model (fallback
path, used only if the ADAU1761 cannot be configured for 16 kHz output).

Design: scipy.signal.remez, passband 0..7 kHz, stopband 9 kHz..24 kHz, minimum
odd tap count (multiple of 3 rounded up for 3-phase polyphase) that reaches
>= 60 dB stopband attenuation AFTER Q1.15 quantisation.

Integer arithmetic: y[n] = rshift_round( sum_k h_q[k] * x48[3n - k], 15 ), saturate 24 bit.
Accumulator width: 24 (x) + 16 (h) + ceil(log2(taps)) <= 47 bits.

usage: python snn/frontend/decimator.py   (designs, verifies SNR on dataset wavs, prints taps)
"""
import os, sys, json
import numpy as np
from scipy import signal
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

FS_IN, FS_OUT, DECIM = 48000, 16000, 3
F_PASS, F_STOP = 7000.0, 9000.0
ATT_DB = 60.0
COEF_Q = 15
PCM_BITS = 24

def design(n_taps):
    h = signal.remez(n_taps, [0, F_PASS, F_STOP, FS_IN / 2], [1, 0], weight=[1, 10], fs=FS_IN)
    return h

def quantise(h):
    s = 1 << COEF_Q
    return np.clip(np.round(h * s), -s, s - 1).astype(np.int64)

def stopband_atten(hq):
    w, H = signal.freqz(hq / (1 << COEF_Q), worN=8192, fs=FS_IN)
    sb = w >= F_STOP
    pb = w <= F_PASS
    return -20 * np.log10(np.abs(H[sb]).max()), np.ptp(20 * np.log10(np.abs(H[pb])))

def find_min_taps():
    for n in range(49, 300, 6):            # N = 6j+1 -> group delay (N-1)/2 is a multiple of 3 (integer 16 kHz delay)
        hq = quantise(design(n))
        att, ripple = stopband_atten(hq)
        if att >= ATT_DB:
            return n, hq, att, ripple
    raise RuntimeError

def decimate_int(x48, hq):
    """x48: int64 24-bit samples at 48 kHz -> int64 24-bit at 16 kHz (direct form; polyphase gives
    identical results since it is the same sum split by phase)."""
    x = np.asarray(x48, dtype=np.int64)
    n_out = (len(x) - 1) // DECIM + 1
    y = np.zeros(n_out, dtype=np.int64)
    # full convolution restricted to output phases 0, 3, 6, ...
    acc = np.convolve(x, hq[::-1] * 0 + hq, mode='full')[:len(x)]  # acc[n] = sum_k h[k] x[n-k]
    acc = acc[::DECIM]
    y = (acc + (1 << (COEF_Q - 1))) >> COEF_Q
    return np.clip(y, -(1 << (PCM_BITS - 1)), (1 << (PCM_BITS - 1)) - 1)

def decimate_polyphase_int(x48, hq):
    """explicit 3-phase polyphase form (what the RTL implements): y[n] = sum_p sum_j E_p[j] x_p[n-j]
    with E_p[j] = h[3j+p], x_p[n] = x[3n - p]."""
    x = np.asarray(x48, dtype=np.int64)
    n_out = len(x) // DECIM
    acc = np.zeros(n_out, dtype=np.int64)
    L = int(np.ceil(len(hq) / DECIM))
    hp = np.zeros(L * DECIM, dtype=np.int64); hp[:len(hq)] = hq
    for p in range(DECIM):
        E = hp[p::DECIM]
        xp = np.zeros(n_out, dtype=np.int64)
        idx = DECIM * np.arange(n_out) - p
        ok = idx >= 0
        xp[ok] = x[idx[ok]]
        acc += np.convolve(xp, E, mode='full')[:n_out]
    y = (acc + (1 << (COEF_Q - 1))) >> COEF_Q
    return np.clip(y, -(1 << (PCM_BITS - 1)), (1 << (PCM_BITS - 1)) - 1)

def verify(hq, n_files=30):
    import soundfile as sf
    from snn.frontend.frontend_ref import wav_float_to_pcm24
    DATA = os.path.join(ROOT, 'data', 'speech_commands_v0.02')
    lst = [l.strip() for l in open(os.path.join(DATA, 'testing_list.txt')) if l.strip()][:n_files]
    snrs = []
    delay = (len(hq) - 1) // 2 // DECIM        # group delay in output samples (linear phase)
    for f in lst:
        x, sr = sf.read(os.path.join(DATA, f), dtype='float32')
        x = np.pad(x, (0, max(0, 16000 - len(x))))[:16000]
        x16 = wav_float_to_pcm24(x)
        # upsample x3 with a high quality float filter (simulates a 48 kHz codec stream)
        x48 = signal.resample_poly(x16.astype(np.float64), DECIM, 1, window=('kaiser', 14.0))
        x48 = np.clip(np.round(x48), -(1 << 23), (1 << 23) - 1).astype(np.int64)
        y = decimate_polyphase_int(x48, hq)
        y2 = decimate_int(x48, hq)
        assert np.array_equal(y[:len(y2)], y2[:len(y)]), "polyphase != direct form"
        ref = x16[: len(y) - delay].astype(np.float64)
        est = y[delay: delay + len(ref)].astype(np.float64)
        full = 10 * np.log10(np.sum(ref ** 2) / np.sum((ref - est) ** 2))
        # passband-only SNR: same float low-pass (0..6.5 kHz) on both signals
        lp = signal.firwin(255, 6500, fs=FS_OUT)
        r2, e2 = signal.filtfilt(lp, [1], ref), signal.filtfilt(lp, [1], est)
        pb = 10 * np.log10(np.sum(r2 ** 2) / np.sum((r2 - e2) ** 2))
        snrs.append((full, pb))
    s = np.array(snrs)
    return dict(full_mean=float(s[:,0].mean()), full_min=float(s[:,0].min()), pass_mean=float(s[:,1].mean()), pass_min=float(s[:,1].min()))

if __name__ == '__main__':
    n, hq, att, ripple = find_min_taps()
    print(f"min taps = {n}  stopband atten (Q1.15) = {att:.1f} dB  passband ripple = {ripple:.3f} dB")
    print(f"sum(h_q) = {hq.sum()}  (DC gain {hq.sum()/32768:.5f}), max |h_q| = {np.abs(hq).max()}")
    r = verify(hq)
    print(f"decimator SNR vs original 16 kHz (30 wavs): full-band mean {r['full_mean']:.1f} / min {r['full_min']:.1f} dB ; "
          f"passband(<6.5k) mean {r['pass_mean']:.1f} / min {r['pass_min']:.1f} dB")
    out = os.path.join(ROOT, 'snn', 'runs', 'decimator.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(n_taps=n, stop_att_db=att, pass_ripple_db=ripple, snr=r, group_delay_16k_samples=(n-1)//2//DECIM,
                   coef_q15=[int(v) for v in hq]), open(out, 'w'), indent=1)
    np.save(os.path.join(ROOT, 'snn', 'runs', 'decim_coef_q15.npy'), hq)
    print('saved', out)
