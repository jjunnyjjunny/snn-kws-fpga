"""
frontend_ref.py — bit-exact integer front-end golden model (RTL reference).

pcm24 (int, 16 kHz) -> uint8 log-mel features [T, N_MELS]

Every block is an independent function operating on numpy int64 arrays so an RTL
testbench can compare intermediate outputs block by block:

  1. scale_input     : 24 bit -> DB bit            (rshift_round + saturate)
  2. frame           : [n] -> [T, WIN]              (hop 160, no padding)
  3. window          : * Hann Q1.15 -> DB bit       (rshift_round(.,15) + sat)
  4. fft512_int      : radix-2 DIT, DB-bit data, Q1.15 twiddles, 1/2 per stage
  5. power           : re^2 + im^2, bins 0..256     (2*DB bit unsigned, exact)
  6. mel_int         : sparse 8-bit triangle weights -> MEL_ACC_BITS accumulator
  7. log2_int        : leading-one + 5-bit mantissa LUT -> Q6.4
  8. to_feature      : clip(log2q - LOG_OFFSET, 0, 255) -> uint8

All constants come from fe_config.py.  A `FrontendInt` object bundles the ROMs
(window, twiddles, mel table, log LUT) and exposes `run()` (features only) and
`run_debug()` (all intermediates, for test-vector dumps).
"""
import numpy as np
from . import fe_config as C

# ----------------------------------------------------------------------------- helpers
def rshift_round(x, s):
    """round-half-up arithmetic right shift (x int64)."""
    if s == 0:
        return x
    return (x + (1 << (s - 1))) >> s

def sat(x, bits):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return np.clip(x, lo, hi)

def bitrev_indices(n_bits):
    n = 1 << n_bits
    idx = np.arange(n)
    rev = np.zeros(n, dtype=np.int64)
    for b in range(n_bits):
        rev |= ((idx >> b) & 1) << (n_bits - 1 - b)
    return rev

# ----------------------------------------------------------------------------- ROM generators
def make_hann_q15(win=C.WIN):
    """periodic Hann (librosa / scipy fftbins=True), Q1.15 unsigned, max 32767."""
    n = np.arange(win)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * n / win)
    q = np.clip(np.round(w * (1 << C.WIN_Q)), 0, (1 << C.WIN_Q) - 1).astype(np.int64)
    return q

def make_twiddle_q15(n_fft=C.N_FFT):
    """W_N^k = cos(2pi k/N) - j sin(2pi k/N), k=0..N/2-1, Q1.15 signed."""
    k = np.arange(n_fft // 2)
    re = np.cos(2 * np.pi * k / n_fft)
    im = -np.sin(2 * np.pi * k / n_fft)
    s = 1 << C.TW_Q
    re_q = np.clip(np.round(re * s), -s, s - 1).astype(np.int64)
    im_q = np.clip(np.round(im * s), -s, s - 1).astype(np.int64)
    return re_q, im_q

def hz_to_mel(f):
    """Slaney mel scale (librosa default, htk=False)."""
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = f >= min_log_hz
    mels = np.where(log_t, min_log_mel + np.log(np.maximum(f, 1e-9) / min_log_hz) / logstep, mels)
    return mels

def mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = m >= min_log_mel
    freqs = np.where(log_t, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)
    return freqs

def make_mel_float(n_mels=C.N_MELS, sr=C.SR, n_fft=C.N_FFT, fmin=C.FMIN, fmax=C.FMAX):
    """Peak-normalised triangular filters on Slaney mel scale (== librosa norm=None, htk=False).
    Returns float matrix [n_mels, n_bins]."""
    n_bins = n_fft // 2 + 1
    fftfreqs = np.arange(n_bins) * sr / n_fft
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    fb = np.zeros((n_mels, n_bins))
    for m in range(n_mels):
        lo, ce, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        up = (fftfreqs - lo) / (ce - lo)
        dn = (hi - fftfreqs) / (hi - ce)
        fb[m] = np.maximum(0.0, np.minimum(up, dn))
    return fb

def make_mel_sparse(n_mels=C.N_MELS, **kw):
    """Quantise to 8-bit unsigned and convert to sparse table.
    Returns list of (start_bin, length, weights[int64]) and the dense int matrix."""
    fb = make_mel_float(n_mels, **kw)
    wmax = (1 << C.MEL_W_BITS) - 1
    fq = np.clip(np.round(fb * wmax), 0, wmax).astype(np.int64)
    table = []
    for m in range(n_mels):
        nz = np.nonzero(fq[m])[0]
        if len(nz) == 0:
            raise ValueError(f"mel band {m} has no non-zero 8-bit weight")
        s, e = int(nz[0]), int(nz[-1])
        table.append((s, e - s + 1, fq[m, s:e + 1].copy()))
    return table, fq

def make_log_lut():
    i = np.arange(1 << C.LOG_MANT_BITS)
    lut = np.round((1 << C.LOG_FRAC_BITS) * np.log2(1 + (i + 0.5) / (1 << C.LOG_MANT_BITS)))
    return lut.astype(np.int64)

# ----------------------------------------------------------------------------- blocks
def scale_input(pcm24, in_shift=C.IN_SHIFT, out_bits=C.FFT_DATA_BITS):
    """block 1: 24-bit signed -> DB-bit signed, round-half-up, saturate."""
    x = np.asarray(pcm24, dtype=np.int64)
    return sat(rshift_round(x, in_shift), out_bits)

def frame(x, win=C.WIN, hop=C.HOP):
    """block 2: [..., n] -> [..., T, win], T = (n - win)//hop + 1, no padding."""
    x = np.asarray(x)
    n = x.shape[-1]
    T = (n - win) // hop + 1
    idx = np.arange(win)[None, :] + hop * np.arange(T)[:, None]
    return x[..., idx]

def window(frames, hann_q15, out_bits=C.FFT_DATA_BITS):
    """block 3: frames * Hann(Q1.15) -> DB bits."""
    y = frames.astype(np.int64) * hann_q15[None, :]
    return sat(rshift_round(y, C.WIN_Q), out_bits)

def fft512_int(xr, xi=None, tw_re=None, tw_im=None, data_bits=C.FFT_DATA_BITS, n_fft=C.N_FFT):
    """block 4: radix-2 DIT in-place integer FFT, 1/2 scaling per stage.
    xr, xi: [..., n_fft] int64 (xi None -> zeros). Returns (Xr, Xi) [..., n_fft]."""
    n_bits = int(np.log2(n_fft))
    if tw_re is None:
        tw_re, tw_im = make_twiddle_q15(n_fft)
    rev = bitrev_indices(n_bits)
    ar = np.asarray(xr, dtype=np.int64)[..., rev]
    ai = np.zeros_like(ar) if xi is None else np.asarray(xi, dtype=np.int64)[..., rev]
    lead = ar.shape[:-1]
    for s in range(1, n_bits + 1):
        m = 1 << s
        half = m >> 1
        ar = ar.reshape(*lead, n_fft // m, m)
        ai = ai.reshape(*lead, n_fft // m, m)
        a_r, a_i = ar[..., :half], ai[..., :half]
        b_r, b_i = ar[..., half:], ai[..., half:]
        k = np.arange(half) * (n_fft // m)
        wr, wi = tw_re[k], tw_im[k]
        pr = b_r * wr - b_i * wi          # <= 2^(DB+16) magnitude
        pi = b_r * wi + b_i * wr
        a15r, a15i = a_r << C.TW_Q, a_i << C.TW_Q
        nar = sat(rshift_round(a15r + pr, C.TW_Q + 1), data_bits)
        nai = sat(rshift_round(a15i + pi, C.TW_Q + 1), data_bits)
        nbr = sat(rshift_round(a15r - pr, C.TW_Q + 1), data_bits)
        nbi = sat(rshift_round(a15i - pi, C.TW_Q + 1), data_bits)
        ar = np.concatenate([nar, nbr], axis=-1).reshape(*lead, n_fft)
        ai = np.concatenate([nai, nbi], axis=-1).reshape(*lead, n_fft)
    return ar, ai

def power(Xr, Xi, n_bins=C.N_BINS):
    """block 5: |X|^2 for bins 0..n_bins-1, unsigned 2*DB bits (exact, no truncation)."""
    r = Xr[..., :n_bins].astype(np.int64)
    i = Xi[..., :n_bins].astype(np.int64)
    return r * r + i * i

def mel_int(P, mel_table):
    """block 6: sparse 8-bit triangular filters. P [..., n_bins] -> [..., n_mels] int64."""
    out = np.zeros(P.shape[:-1] + (len(mel_table),), dtype=np.int64)
    for m, (s, L, w) in enumerate(mel_table):
        out[..., m] = (P[..., s:s + L] * w).sum(axis=-1)
    return out

def leading_one_pos(x):
    """floor(log2(x)) for x>0 (0 for x==0), vectorised binary search on int64."""
    x = np.asarray(x, dtype=np.int64)
    pos = np.zeros_like(x)
    y = x.copy()
    for b in (32, 16, 8, 4, 2, 1):
        t = (y >> b) > 0
        pos += t * b
        y = np.where(t, y >> b, y)
    return pos

def log2_int(x, lut, mant_bits=C.LOG_MANT_BITS, frac_bits=C.LOG_FRAC_BITS):
    """block 7: log2 in Q.frac via leading-one position + mantissa LUT. log2(0) := 0."""
    x = np.asarray(x, dtype=np.int64)
    pos = leading_one_pos(x)
    sh = pos - mant_bits
    mant = np.where(sh >= 0,
                    np.right_shift(x, np.maximum(sh, 0)),
                    np.left_shift(x, np.maximum(-sh, 0))) & ((1 << mant_bits) - 1)
    out = (pos << frac_bits) + lut[mant]
    return np.where(x > 0, out, 0)

def to_feature(log2q, offset=C.LOG_OFFSET, bits=C.FEAT_BITS):
    """block 8: global offset subtract + saturate to unsigned FEAT_BITS."""
    return np.clip(log2q - offset, 0, (1 << bits) - 1).astype(np.uint8)

# ----------------------------------------------------------------------------- bundle
class FrontendInt:
    def __init__(self, n_mels=C.N_MELS, win=C.WIN, hop=C.HOP, data_bits=C.FFT_DATA_BITS,
                 log_offset=C.LOG_OFFSET, in_shift=None):
        self.n_mels, self.win, self.hop, self.data_bits = n_mels, win, hop, data_bits
        self.in_shift = C.PCM_BITS - data_bits if in_shift is None else in_shift
        self.log_offset = log_offset
        self.hann = make_hann_q15(win)
        self.tw_re, self.tw_im = make_twiddle_q15()
        self.mel_table, self.mel_dense = make_mel_sparse(n_mels)
        self.lut = make_log_lut()

    def run_debug(self, pcm24):
        """pcm24: [..., n] int. Returns dict of all intermediates."""
        d = {}
        d['x_db'] = scale_input(pcm24, self.in_shift, self.data_bits)
        d['frames'] = frame(d['x_db'], self.win, self.hop)
        d['windowed'] = window(d['frames'], self.hann, self.data_bits)
        T = d['windowed'].shape[-2]
        pad = np.zeros(d['windowed'].shape[:-1] + (C.N_FFT - self.win,), dtype=np.int64)
        xin = np.concatenate([d['windowed'], pad], axis=-1)
        d['fft_re'], d['fft_im'] = fft512_int(xin, None, self.tw_re, self.tw_im, self.data_bits)
        d['power'] = power(d['fft_re'], d['fft_im'])
        d['mel'] = mel_int(d['power'], self.mel_table)
        d['log2q'] = log2_int(d['mel'], self.lut)
        d['feat'] = to_feature(d['log2q'], self.log_offset)
        return d

    def run(self, pcm24):
        return self.run_debug(pcm24)['feat']

    def run_log2q(self, pcm24):
        return self.run_debug(pcm24)['log2q']

def wav_float_to_pcm24(x):
    """dataset float32 [-1,1) (16-bit wav) -> 24-bit signed int: int16 << 8."""
    x16 = np.clip(np.round(np.asarray(x, dtype=np.float64) * 32768.0), -32768, 32767).astype(np.int64)
    return x16 << 8

def apply_gain_pcm24(pcm24, gain_db):
    """gain in dB with 24-bit saturation (for augmentation / robustness tests)."""
    g = 10 ** (gain_db / 20.0)
    y = np.round(np.asarray(pcm24, dtype=np.float64) * g)
    return np.clip(y, -(1 << 23), (1 << 23) - 1).astype(np.int64)
