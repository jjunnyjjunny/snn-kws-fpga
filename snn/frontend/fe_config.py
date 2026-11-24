"""
Integer front-end configuration (single source of truth for Python golden model
and RTL). Every bit width / rounding / saturation rule used by frontend_ref.py is
defined here.

Rounding convention everywhere: ROUND-HALF-UP on arithmetic right shift,
    rshift_round(x, s) = (x + 2^(s-1)) >> s      (>> is arithmetic / floor)
Saturation convention: symmetric two's complement clip to [-2^(n-1), 2^(n-1)-1].
"""
SR          = 16000      # input sample rate (codec configured to 16 kHz)
PCM_BITS    = 24         # signed PCM from i2s_data_interface (audio_l_in[23:0])

# ---- block 1: input scaling 24 bit -> FFT data width -------------------------
FFT_DATA_BITS = 18       # DB. 18 chosen (DSP48E1 25x18 friendly, +12 dB SNR vs 16); 16 also supported
IN_SHIFT      = PCM_BITS - FFT_DATA_BITS   # right shift w/ round-half-up, then saturate to DB bits

# ---- block 2: framing --------------------------------------------------------
HOP   = 160              # 10 ms
WIN   = 480              # 30 ms (400 = 25 ms also supported)
N_FFT = 512
N_BINS = N_FFT // 2 + 1  # 257 (bins 0..256 used)
CLIP_LEN = 16000         # 1 s clip
N_FRAMES = (CLIP_LEN - WIN) // HOP + 1   # = 98 for WIN=480 and WIN=400. No padding:
                                         # frame t = x[160 t : 160 t + WIN], t = 0..97

# ---- block 3: window ---------------------------------------------------------
WIN_Q = 15               # Hann, periodic, Q1.15 unsigned (0..32767), ROM depth WIN
# windowed[n] = sat_DB( rshift_round(x[n] * w[n], 15) )

# ---- block 4: FFT ------------------------------------------------------------
TW_Q = 15                # twiddle Q1.15 signed, ROM depth N_FFT/2 = 256 (cos, -sin)
FFT_STAGES = 9           # radix-2 DIT, in-place, bit-reversed input
# butterfly (per stage, scaling 1/2):
#   p  = b * W                              (33-bit complex product, Q1.15 * DB)
#   a' = sat_DB( rshift_round(a<<15 + p, 16) )     == round((a + p/2^15)/2)
#   b' = sat_DB( rshift_round(a<<15 - p, 16) )
# Overall gain = 1/512 relative to the unscaled DFT.

# ---- block 5: power ----------------------------------------------------------
POW_BITS = 2 * FFT_DATA_BITS      # re^2+im^2 <= 2^(2DB-1) -> unsigned 2DB bits, NO truncation

# ---- block 6: mel filterbank -------------------------------------------------
N_MELS   = 40            # 32 also supported (both exported)
FMIN     = 0.0
FMAX     = 8000.0
MEL_W_BITS = 8           # unsigned weight 0..255 (peak-normalised triangles, Slaney mel scale)
# acc[m] = sum_k w[m,k] * P[k]  -> MEL_ACC_BITS unsigned (no truncation)
MEL_ACC_BITS = POW_BITS + MEL_W_BITS + 6     # 6 = ceil(log2(max band length <= 64))

# ---- block 7: log2 -----------------------------------------------------------
LOG_MANT_BITS = 5        # LUT index = 5 bits right below the leading one (32 entries)
LOG_FRAC_BITS = 4        # output fraction bits -> log2 in Q6.4 (0..~46*16)
# log2q(x) = (lead_pos(x) << 4) + LUT[mant5(x)],  log2q(0) = 0
# LUT[i] = round( 16 * log2(1 + (i + 0.5) / 32) ),  i = 0..31  (values 0..16)

# ---- block 8: feature --------------------------------------------------------
FEAT_BITS  = 8
LOG_OFFSET = 332         # feature = clip(log2q - LOG_OFFSET, 0, 255). Set from train stats (see report); valid for DB=18
