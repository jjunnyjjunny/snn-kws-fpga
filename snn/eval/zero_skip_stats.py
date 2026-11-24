"""
zero_skip_stats.py — B-6: input-zero ratio, input-layer MAC savings and accuracy of the integer SNN on the
test set, (a) as is, (b) with a simulated microphone noise floor: white / pink noise at -60/-70/-80 dBFS RMS
added to the 24-bit waveform before the integer front-end.
usage: python snn/eval/zero_skip_stats.py --run snn/runs/M3k3e60_c12_q8_kd --workers 16
"""
import os, sys, json, argparse, time
import numpy as np
from multiprocessing import Pool
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from snn.frontend.frontend_ref import FrontendInt, wav_float_to_pcm24
from snn.data.make_dataset import load_item
from snn.golden.snn_int_ref import run_int
_G = {}

def pink_noise(n, rng):
    """1/f noise via FFT shaping, unit RMS."""
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(X)); f[0] = 1
    X = X / np.sqrt(f)
    x = np.fft.irfft(X, n)
    return x / x.std()

def _init():
    _G['fe'] = FrontendInt()

def _work(args):
    rels, kind, dbfs, seed = args
    rng = np.random.default_rng(seed)
    pcm = np.stack([wav_float_to_pcm24(load_item(r)) for r in rels])
    if kind != 'none':
        rms = (1 << 23) * 10 ** (dbfs / 20.0)
        nz = np.stack([(rng.standard_normal(16000) if kind == 'white' else pink_noise(16000, rng)) for _ in rels]) * rms
        pcm = np.clip(pcm + np.round(nz).astype(np.int64), -(1 << 23), (1 << 23) - 1)
    return _G['fe'].run(pcm)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=os.path.join(ROOT, 'snn', 'runs', 'M3k3e60_c12_q8_kd')); ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--out', default=os.path.join(ROOT, 'snn', 'runs', 'zero_skip_stats.json'))
    a = ap.parse_args()
    import torch; torch.backends.cuda.matmul.allow_tf32 = False
    from snn.golden.export_snn_mem import load_run, extract
    model, cfg, res = load_run(a.run); P = extract(model)
    fl = json.load(open(os.path.join(ROOT, 'snn', 'data', 'filelist_c12.json')))['test']
    rels = [r for r, _ in fl]; y = np.array([l for _, l in fl])
    K, n_in, n_hid = 3, 40, 64
    rows = []
    with Pool(a.workers, initializer=_init) as pool:
        for kind, dbfs in [('none', 0), ('white', -80), ('white', -70), ('white', -60), ('white', -50), ('white', -40), ('pink', -80), ('pink', -70), ('pink', -60), ('pink', -50), ('pink', -40)]:
            t0 = time.time()
            jobs = [(rels[i:i + 128], kind, dbfs, 7 + i) for i in range(0, len(rels), 128)]
            X = np.concatenate(pool.map(_work, jobs), 0)
            g = run_int(P, X, trace=False)
            zero = float((X == 0).mean())
            # input-layer rows = K taps x 40 features per frame; a row is skipped iff its feature is 0 (taps see the same frames)
            dense = K * n_in * n_hid
            # exact per-frame count: taps j read frame t-j -> average over taps ~ zero ratio; compute exactly
            T = X.shape[1]
            nz_rows = 0
            for j in range(K):
                sh = np.zeros_like(X); sh[:, j:] = X[:, :T - j]
                nz_rows += (sh != 0).sum()
            mac_per_frame = nz_rows * n_hid / (X.shape[0] * T)
            rows.append(dict(noise=kind, dbfs=dbfs if kind != 'none' else None, feat_zero_ratio=zero, feat_mean=float(X.mean()),
                             input_mac_per_frame=float(mac_per_frame), input_mac_dense=dense, mac_saving=1 - mac_per_frame / dense,
                             acc=float((g['pred_sum'] == y).mean())))
            print(rows[-1], f"({time.time()-t0:.0f}s)", flush=True)
    json.dump(rows, open(a.out, 'w'), indent=1)

if __name__ == '__main__':
    main()
