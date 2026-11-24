"""
t0_bound.py — B-1: provable threshold T0 on the time-domain frame energy E0 = sum |x18[n]| over the 480-sample
window such that E0 < T0 guarantees all 40 features of the frame are 0 (so the front-end can be skipped losslessly).

Chain of bounds (integer front-end, frontend_ref.py):
  |xw[n]| <= |x[n]|                              (Hann Q1.15 < 1, round-half-up cannot exceed |x|)
  |X_int[k]| <= S1/512 * (1+2^-15)^9 + 9*0.5*sqrt(2)   S1 = sum|xw| <= sum|x| = E0 ; twiddle |W_q| <= 1+2^-15 ;
                                                 one round-half-up per stage (|err| <= 0.5 per re/im), scaling 1/2 per stage keeps errors from growing
  P[k] = |X|^2 <= Pmax ;  acc[m] <= 255 * Lmax * Pmax  (Lmax = longest mel band)
  feature = 0  <=>  log2q(acc) < 332  <=>  acc < A_thr (exact, scanned from log2_int)
  =>  T0 = largest E0 with (E0/512*(1+2^-15)^9 + 6.364)^2 * 255 * Lmax < A_thr
Also the Parseval variant on E2 = sum x^2: max_k |X|^2 <= (1/512) sum|xw|^2 <= E2/512 (plus the rounding term).
Empirical check: all test-set frames -> (E0, any feature != 0); report the smallest E0 with a non-zero feature and the
gating ratio of frames under the proven T0 and under the empirical threshold.
"""
import os, sys, json, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, ROOT)
from snn.frontend import fe_config as C
from snn.frontend.frontend_ref import log2_int, make_log_lut, make_mel_sparse, FrontendInt, wav_float_to_pcm24, scale_input, frame
from snn.data.make_dataset import load_item
lut = make_log_lut()
# exact A_thr: smallest acc with log2q >= LOG_OFFSET (log2q is monotone in acc)
lo, hi = 1, 1 << 40
while lo < hi:
    mid = (lo + hi) // 2
    if int(log2_int(np.array([mid]), lut)[0]) >= C.LOG_OFFSET: hi = mid
    else: lo = mid + 1
A_thr = lo
table, _ = make_mel_sparse(C.N_MELS); Lmax = max(L for _, L, _ in table); wsum_max = max(int(w.sum()) for _, _, w in table)
g = (1 + 2 ** -15) ** 9; err = 9 * 0.5 * np.sqrt(2)
Pmax_allowed = A_thr / (255.0 * Lmax)              # bound with max weight 255 on every bin (loose)
Pmax_allowed_w = A_thr / wsum_max                   # tighter: acc <= (sum of the band's weights) * Pmax
T0_sum = int(np.floor((np.sqrt(Pmax_allowed) - err) * 512 / g)) - 1          # on E0 = sum|x|
T0_sum_w = int(np.floor((np.sqrt(Pmax_allowed_w) - err) * 512 / g)) - 1
# Parseval variant on E2 = sum x^2 : |X|^2 <= E2/512*g^2 (+ cross term with rounding): (sqrt(E2/512)*g + err)^2 < Pmax
T2 = int(np.floor(((np.sqrt(Pmax_allowed_w) - err) / g) ** 2 * 512)) - 1
print(f"A_thr (smallest acc with feature>0) = {A_thr}  (2^{np.log2(A_thr):.3f}) ; Lmax = {Lmax} bins ; max band weight sum = {wsum_max}")
print(f"Pmax allowed: 255*Lmax bound {Pmax_allowed:.1f}, weight-sum bound {Pmax_allowed_w:.1f} ; FFT rounding term {err:.3f}")
print(f"T0 (E0 = sum|x18| over 480 samples): loose {T0_sum}, with weight-sum {T0_sum_w}  -> mean |x18| < {T0_sum_w/480:.2f} LSB  (~{20*np.log10(max(T0_sum_w/480,1e-9)/2**17):.0f} dBFS avg)")
print(f"T2 (E2 = sum x18^2): {T2}")
# ---- empirical: test set clips
fl = json.load(open(os.path.join(ROOT, 'snn', 'data', 'filelist_c12.json')))['test']
fe = FrontendInt(); E0s = []; nz = []; E2s = []
for i in range(0, len(fl), 256):
    pcm = np.stack([wav_float_to_pcm24(load_item(r)) for r, _ in fl[i:i + 256]])
    d = fe.run_debug(pcm); x = d['x_db']; fr = frame(x)
    E0s.append(np.abs(fr).sum(-1)); E2s.append((fr.astype(np.int64) ** 2).sum(-1)); nz.append((d['feat'] != 0).any(-1))
E0 = np.concatenate(E0s).ravel(); E2 = np.concatenate(E2s).ravel(); NZ = np.concatenate(nz).ravel()
emp = int(E0[NZ].min()) if NZ.any() else None; emp2 = int(E2[NZ].min())
res = dict(A_thr=int(A_thr), Lmax=int(Lmax), wsum_max=int(wsum_max), T0_proven=T0_sum_w, T0_loose=T0_sum, T2_proven=T2,
           empirical_min_E0_nonzero=emp, empirical_min_E2_nonzero=emp2, n_frames=int(len(E0)),
           gated_ratio_proven=float((E0 < T0_sum_w).mean()), gated_ratio_empirical=float((E0 < emp).mean()),
           zero_feature_frames_ratio=float((~NZ).mean()), violations_proven=int(((E0 < T0_sum_w) & NZ).sum()))
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(ROOT, 'snn', 'runs', 't0_bound.json'), 'w'), indent=1)
