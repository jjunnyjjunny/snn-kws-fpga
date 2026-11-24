"""
verify_frontend.py — integer front-end vs float reference on N wav files.
Reports correlation, SNR, per-band error, dynamic range / saturation, input-gain
precision loss, FFT SNR, and a librosa cross-check of the float reference.

usage: python snn/frontend/verify_frontend.py --n 100 --out snn/runs/frontend_verify.json
"""
import os, sys, json, argparse, glob
import numpy as np
import soundfile as sf
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from snn.frontend import fe_config as C
from snn.frontend.frontend_ref import (FrontendInt, wav_float_to_pcm24, apply_gain_pcm24,
                                       fft512_int, make_twiddle_q15, make_mel_float)
from snn.frontend.float_ref import logmel_float

KEYWORDS = ['yes', 'no', 'stop', 'go', 'up', 'down', 'left', 'right', 'on', 'off']
DATA = os.path.join(ROOT, 'data', 'speech_commands_v0.02')

def load_wavs(n_per_kw):
    test_list = [l.strip() for l in open(os.path.join(DATA, 'testing_list.txt')) if l.strip()]
    out, names = [], []
    for kw in KEYWORDS:
        files = [f for f in test_list if f.startswith(kw + '/')][:n_per_kw]
        for f in files:
            x, sr = sf.read(os.path.join(DATA, f), dtype='float32')
            assert sr == 16000
            x = np.pad(x, (0, max(0, 16000 - len(x))))[:16000]
            out.append(x); names.append(f)
    return np.stack(out), names

def snr_db(ref, est):
    return float(10 * np.log10(np.sum(ref ** 2) / max(np.sum((ref - est) ** 2), 1e-30)))

def compare(Lf, Li, label):
    """Lf, Li: [N,T,M] log2-domain (Q.4 units). Mask out float values below the int floor."""
    err = (Li - Lf)
    r = {}
    r['label'] = label
    r['corr'] = float(np.corrcoef(Lf.ravel(), Li.ravel())[0, 1])
    r['snr_db'] = snr_db(Lf.ravel() - Lf.mean(), Li.ravel() - Lf.mean())
    r['mae_q4'] = float(np.abs(err).mean())
    r['mae_dB'] = float(np.abs(err).mean() * 6.0206 / 16)
    r['rmse_dB'] = float(np.sqrt((err ** 2).mean()) * 6.0206 / 16)
    r['bias_dB'] = float(err.mean() * 6.0206 / 16)
    r['per_band_mae_dB'] = [float(v) for v in np.abs(err).mean(axis=(0, 1)) * 6.0206 / 16]
    # feature domain: both sides through offset + clip (what the SNN actually sees)
    Ff = np.clip(Lf - C.LOG_OFFSET, 0, 255); Fi = np.clip(Li - C.LOG_OFFSET, 0, 255)
    ef = Fi - Ff
    r['feat_corr'] = float(np.corrcoef(Ff.ravel(), Fi.ravel())[0, 1])
    r['feat_snr_db'] = snr_db(Ff.ravel() - Ff.mean(), Fi.ravel() - Ff.mean())
    r['feat_mae_dB'] = float(np.abs(ef).mean() * 6.0206 / 16)
    r['feat_per_band_mae_dB'] = [float(v) for v in np.abs(ef).mean(axis=(0, 1)) * 6.0206 / 16]
    r['feat_exact_ratio'] = float((np.round(Ff) == Fi).mean())
    # error on "loud" cells only (float value within 60 dB of its max) — excludes noise floor
    thr = Lf.max() - 60 * 16 / 6.0206
    m = Lf > thr
    r['mae_dB_top60dB'] = float(np.abs(err[m]).mean() * 6.0206 / 16)
    r['corr_top60dB'] = float(np.corrcoef(Lf[m], Li[m])[0, 1])
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--out', default=os.path.join(ROOT, 'snn', 'runs', 'frontend_verify.json'))
    a = ap.parse_args()
    wav, names = load_wavs(a.n // 10)
    pcm = wav_float_to_pcm24(wav)
    print(f"loaded {len(wav)} wavs, peak |pcm24| = {np.abs(pcm).max()} ({20*np.log10(np.abs(pcm).max()/2**23):.1f} dBFS)")
    report = {'n_files': len(wav), 'results': [], 'fft_snr': {}, 'gain_sweep': [], 'dynamic_range': {}}

    # ---- FFT SNR on real windowed frames (DB 16 / 18) --------------------------------------
    tw = make_twiddle_q15()
    for db in (16, 18):
        fe = FrontendInt(data_bits=db)
        d = fe.run_debug(pcm[:20])
        xin = np.concatenate([d['windowed'], np.zeros(d['windowed'].shape[:-1] + (C.N_FFT - fe.win,), np.int64)], -1)
        ref = np.fft.fft(xin.astype(np.float64), axis=-1) / C.N_FFT
        est = d['fft_re'] + 1j * d['fft_im']
        s = snr_db(np.abs(ref).ravel() * 0 + ref.real.ravel(), est.real.ravel())
        s2 = float(10 * np.log10(np.sum(np.abs(ref) ** 2) / np.sum(np.abs(ref - est) ** 2)))
        report['fft_snr'][f'DB{db}'] = s2
        print(f"FFT SNR (DB={db}, real speech frames): {s2:.2f} dB")

    # ---- main comparison grid ------------------------------------------------------------
    for db in (16, 18):
        for win in (480, 400):
            for nm in (40, 32):
                fe = FrontendInt(n_mels=nm, win=win, data_bits=db)
                Li = fe.run_log2q(pcm).astype(np.float64)
                Lf = logmel_float(pcm, n_mels=nm, win=win, data_bits=db)
                r = compare(Lf, Li, f"DB={db} win={win} n_mels={nm}")
                report['results'].append(r)
                print(f"{r['label']:28s} raw: corr={r['corr']:.4f} | feat: corr={r['feat_corr']:.5f} snr={r['feat_snr_db']:.1f}dB mae={r['feat_mae_dB']:.3f}dB exact={r['feat_exact_ratio']:.3f} "
                      f"| top60dB: mae={r['mae_dB_top60dB']:.3f}dB corr={r['corr_top60dB']:.5f}")

    # ---- dynamic range / saturation with default config ----------------------------------
    fe = FrontendInt()
    d = fe.run_debug(pcm)
    L = d['log2q']
    feat = d['feat']
    pct = np.percentile(L, [0, 0.1, 1, 50, 99, 99.9, 100])
    report['dynamic_range'] = dict(config=f"DB={fe.data_bits} win={fe.win} n_mels={fe.n_mels} offset={fe.log_offset}",
        log2q_percentiles={str(p): float(v) for p, v in zip([0, 0.1, 1, 50, 99, 99.9, 100], pct)},
        feat_min=int(feat.min()), feat_max=int(feat.max()), feat_mean=float(feat.mean()),
        sat_low_ratio=float((L - fe.log_offset < 0).mean()), sat_high_ratio=float((L - fe.log_offset > 255).mean()),
        mel_acc_max_bits=int(np.ceil(np.log2(d['mel'].max() + 1))), power_max_bits=int(np.ceil(np.log2(d['power'].max() + 1))))
    print("dynamic range:", json.dumps(report['dynamic_range'], indent=1))

    # ---- input gain sweep: precision loss of 24->DB truncation -----------------------------
    for db in (16, 18):
        fe = FrontendInt(data_bits=db)
        for g in (0, -20, -40, -60):
            p = apply_gain_pcm24(pcm[:30], g)
            Li = fe.run_log2q(p).astype(np.float64)
            # float reference computed from the FULL 24-bit signal (no truncation) on the same scale
            Lf = logmel_float(p, data_bits=db)
            r = compare(Lf, Li, f"DB={db} gain={g:+d}dB")
            r['sat_low_ratio_feat'] = float((Li - C.LOG_OFFSET < 0).mean())
            report['gain_sweep'].append(r)
            print(f"{r['label']:22s} feat: corr={r['feat_corr']:.4f} mae={r['feat_mae_dB']:.3f}dB | top60: mae={r['mae_dB_top60dB']:.3f}dB corr={r['corr_top60dB']:.5f}  feat<0 ratio={r['sat_low_ratio_feat']:.3f}")

    # ---- librosa cross-check of float reference ------------------------------------------
    try:
        import librosa
        x = wav[0].astype(np.float32)
        S = librosa.feature.melspectrogram(y=x, sr=16000, n_fft=512, hop_length=160, win_length=480,
                                           window='hann', center=False, power=2.0, n_mels=40, norm=None, htk=False)
        Lr = 16 * np.log2(np.maximum(S.T, 1e-30))
        Lf = logmel_float(pcm[:1], n_mels=40, win=480)[0]
        # scale difference: librosa uses float wav (|x|<1) and unscaled FFT; ours DB-bit units and /512.
        # => constant offset in log domain. Check std of difference.
        n = min(Lr.shape[0], Lf.shape[0])      # librosa center=False frames with n_fft -> 97 frames
        m = (Lr[:n] > Lr.max() - 60 * 16 / 6.0206)
        diff = (Lr[:n] - Lf[:n])[m]
        report['librosa_check'] = dict(diff_mean_q4=float(diff.mean()), diff_std_dB=float(diff.std() * 6.0206 / 16),
                                       corr=float(np.corrcoef(Lr[:n][m], Lf[:n][m])[0, 1]))
        print("librosa cross-check (float ref vs librosa, const offset expected):", report['librosa_check'])
    except Exception as e:
        print("librosa check skipped:", e)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(report, open(a.out, 'w'), indent=1)
    print("saved", a.out)

if __name__ == '__main__':
    main()
