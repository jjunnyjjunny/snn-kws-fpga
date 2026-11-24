"""
dump_vectors.py — RTL test vectors: one test clip per class (hex text) with
  front-end intermediates (pcm24, x18, windowed, fft re/im, power, mel acc, log2q, feat) and
  SNN traces (per layer / per timestep membrane + spikes, output membrane, prediction).

Output: snn/golden/vectors/<prefix>/<class>_<idx>/{*.hex, info.json}
usage: CUDA_VISIBLE_DEVICES=6 python snn/golden/dump_vectors.py --run snn/runs/M1_c12_q8 --prefix m1 --data snn/data/feat_c12_m40_w480.npz
"""
import os, sys, json, argparse
import numpy as np, soundfile as sf
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from snn.frontend import fe_config as C
from snn.frontend.frontend_ref import FrontendInt, wav_float_to_pcm24
from snn.golden.export_snn_mem import load_run, extract
from snn.golden.snn_int_ref import run_int
from snn.data.make_dataset import load_item, DATA

def hexdump(path, arr, bits, signed, comment):
    """one word per line; 2-D arrays flattened row-major (row = time/frame)."""
    a = np.asarray(arr, dtype=np.int64).reshape(-1)
    mask = (1 << bits) - 1
    with open(path, 'w') as f:
        f.write(f"// {comment}\n// width={bits} signed={int(signed)} shape={list(np.asarray(arr).shape)} (row-major)\n")
        for v in a:
            v = int(v)
            if signed and v < 0: v += 1 << bits
            assert 0 <= v <= mask, (v, bits, path)
            f.write(f"{v:0{(bits + 3) // 4}x}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--prefix', required=True); ap.add_argument('--data', required=True)
    ap.add_argument('--per_class', type=int, default=1)
    ap.add_argument('--regress', type=int, default=0, help='export N class-balanced regression samples instead')
    ap.add_argument('--stream', default='', help='comma-separated vector dir names to concatenate (streaming golden)')
    ap.add_argument('--stream_name', default='stream')
    a = ap.parse_args()
    if a.regress:
        return dump_regress(a.run, a.prefix, a.data, a.regress)
    if a.stream:
        return dump_stream(a.run, a.prefix, a.data, a.stream.split(','), a.stream_name)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    model, cfg, res = load_run(a.run)
    P = extract(model)
    model = model.cuda()
    z = np.load(a.data); meta = json.loads(str(z['meta'])); classes = meta['classes']
    n_cls = len(classes)
    import re as _re
    _tag = _re.search(r'_(U\d)', os.path.basename(a.data))      # week-4 variants have their own (re-shuffled) file lists
    fl = json.load(open(os.path.join(ROOT, 'snn', 'data', f'filelist_c{n_cls}{"_" + _tag.group(1) if _tag else ""}.json')))['test']
    X_test, y_test = z['X_test'], z['y_test']
    fe = FrontendInt(n_mels=meta['n_mels'], win=meta['win'])
    out_root = os.path.join(ROOT, 'snn', 'golden', 'vectors', a.prefix)
    os.makedirs(out_root, exist_ok=True)
    index = []
    DB = C.FFT_DATA_BITS
    for c in range(n_cls):
        idxs = np.nonzero(y_test == c)[0][: a.per_class]
        for i in idxs:
            rel = fl[i][0]
            wav = load_item(rel)
            pcm = wav_float_to_pcm24(wav)[None]
            d = fe.run_debug(pcm)
            assert np.array_equal(d['feat'][0], X_test[i]), "front-end output differs from dataset feature"
            g = run_int(P, X_test[i:i + 1], trace=True)
            with torch.no_grad():
                o = model(torch.from_numpy(X_test[i:i + 1]).cuda())
            pred_t = int(o['vout'].double().sum(1).argmax(1).item())
            name = f"{classes[c]}_{i}"
            dd = os.path.join(out_root, name); os.makedirs(dd, exist_ok=True)
            hexdump(os.path.join(dd, 'pcm24.hex'), pcm[0], 24, True, f"{rel} 16 kHz 24-bit signed input (int16<<8)")
            hexdump(os.path.join(dd, 'x_db.hex'), d['x_db'][0], DB, True, f"after input scaling >>{fe.in_shift}")
            hexdump(os.path.join(dd, 'windowed.hex'), d['windowed'][0], DB, True, "[T,WIN] after Hann (Q1.15, >>15 round)")
            hexdump(os.path.join(dd, 'fft_re.hex'), d['fft_re'][0], DB, True, "[T,512] FFT real (1/512 scaled)")
            hexdump(os.path.join(dd, 'fft_im.hex'), d['fft_im'][0], DB, True, "[T,512] FFT imag")
            hexdump(os.path.join(dd, 'power.hex'), d['power'][0], 2 * DB, False, "[T,257] re^2+im^2")
            hexdump(os.path.join(dd, 'mel_acc.hex'), d['mel'][0], C.MEL_ACC_BITS, False, "[T,n_mels] mel accumulator")
            hexdump(os.path.join(dd, 'log2q.hex'), d['log2q'][0], 10, False, "[T,n_mels] log2 Q6.4")
            hexdump(os.path.join(dd, 'feat.hex'), d['feat'][0], 8, False, f"[T,n_mels] feature = clip(log2q-{C.LOG_OFFSET},0,255)")
            for L in P['layers']:
                if L['type'] == 'out': continue
                hexdump(os.path.join(dd, f"{L['name']}_v.hex"), g['traces'][L['name']]['v'][0], P['v_bits_hidden'], True, f"[T,N] membrane after reset, layer {L['name']}")
                hexdump(os.path.join(dd, f"{L['name']}_s.hex"), g['traces'][L['name']]['s'][0], 1, False, f"[T,N] spikes, layer {L['name']}")
            hexdump(os.path.join(dd, 'out_v.hex'), g['vout'][0], P['v_bits_out'], True, "[T,C] output membrane")
            info = dict(file=rel, label=int(c), label_name=classes[c], pred_sum=int(g['pred_sum'][0]), pred_last=int(g['pred_last'][0]),
                        pred_max=int(g['pred_max'][0]), pred_torch=pred_t, vout_sum=[int(v) for v in g['vout'][0].sum(0)],
                        vout_last=[int(v) for v in g['vout'][0, -1]], T=int(X_test.shape[1]), n_mels=int(X_test.shape[2]))
            json.dump(info, open(os.path.join(dd, 'info.json'), 'w'), indent=1)
            index.append(info | dict(dir=name))
            print(name, 'pred', info['pred_sum'], 'torch', pred_t, 'label', c)
    json.dump(index, open(os.path.join(out_root, 'index.json'), 'w'), indent=1)
    with open(os.path.join(out_root, 'README.md'), 'w') as f:
        f.write(f"# Test vectors for `{a.prefix}` ({P['model']})\n\nOne directory per clip. Each `.hex` file: 2 comment lines then one hex word per line,\n")
        f.write("row-major flattening of the shape given in the header (row = frame/timestep). Signed values are two's complement.\n\n")
        f.write("| file | content |\n|---|---|\n")
        for fn, desc in [('pcm24.hex', '16000 x 24-bit input samples'), ('x_db.hex', f'after >>{fe.in_shift} scaling ({DB} bit)'),
                         ('windowed.hex', '[T,WIN] windowed frames'), ('fft_re.hex / fft_im.hex', '[T,512] FFT output'),
                         ('power.hex', '[T,257] power'), ('mel_acc.hex', '[T,n_mels] mel accumulator'), ('log2q.hex', '[T,n_mels] log2 Q6.4'),
                         ('feat.hex', '[T,n_mels] 8-bit feature (SNN input)'), ('<layer>_v.hex / _s.hex', '[T,N] membrane (after reset) / spikes'),
                         ('out_v.hex', '[T,C] output membrane'), ('info.json', 'label, predictions (sum / last / max), output sums')]:
            f.write(f"| `{fn}` | {desc} |\n")
    print('wrote', out_root)


# ----------------------------------------------------------------------------- regression export
def dump_regress(run, prefix, data, n_total=240, seed=0):
    """Class-balanced N test samples -> snn/golden/vectors/<prefix>_regress/{feat.hex, pred.hex, vsum.hex, index.json}
    feat.hex: [N,T,n_mels] 8-bit ; pred.hex: N x 4-bit expected class ; vsum.hex: [N,C] 32-bit signed sums."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    model, cfg, res = load_run(run); P = extract(model)
    z = np.load(data); meta = json.loads(str(z['meta'])); classes = meta['classes']
    X, y = z['X_test'], z['y_test']
    rng = np.random.default_rng(seed)
    per = int(np.ceil(n_total / len(classes)))
    idx = np.concatenate([rng.choice(np.nonzero(y == c)[0], per, replace=False) for c in range(len(classes))])[:n_total]
    idx = np.sort(idx)
    g = run_int(P, X[idx], trace=False)
    out = os.path.join(ROOT, 'snn', 'golden', 'vectors', f'{prefix}_regress'); os.makedirs(out, exist_ok=True)
    hexdump(os.path.join(out, 'feat.hex'), X[idx], 8, False, f"[N={len(idx)},T,n_mels] features")
    hexdump(os.path.join(out, 'pred.hex'), g['pred_sum'], 4, False, "[N] expected class (time-sum argmax)")
    hexdump(os.path.join(out, 'vsum.hex'), g['vout'].sum(1), 32, True, "[N,C] expected output sums")
    json.dump(dict(n=len(idx), T=int(X.shape[1]), n_mels=int(X.shape[2]), n_class=len(classes), test_idx=[int(i) for i in idx],
                   label=[int(v) for v in y[idx]], pred=[int(v) for v in g['pred_sum']], acc=float((g['pred_sum'] == y[idx]).mean())),
              open(os.path.join(out, 'index.json'), 'w'))
    print('regress:', out, 'n', len(idx), 'golden acc', float((g['pred_sum'] == y[idx]).mean()))

def dump_stream(run, prefix, data, names, out_name='stream'):
    """Concatenate the listed golden-vector clips (no reset between them) -> one long vector dir
    snn/golden/vectors/<prefix>_<out_name>/ with the same file layout as a single clip (T = 98 * n)."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    model, cfg, res = load_run(run); P = extract(model)
    z = np.load(data); meta = json.loads(str(z['meta'])); classes = meta['classes']
    vec_root = os.path.join(ROOT, 'snn', 'golden', 'vectors', prefix)
    idx = json.load(open(os.path.join(vec_root, 'index.json')))
    feats = []
    for nm in names:
        d = [e for e in idx if e['dir'] == nm][0]
        i = int(nm.rsplit('_', 1)[1]); feats.append(z['X_test'][i])
    x = np.concatenate(feats, 0)[None]            # [1, 98*n, 40]
    g = run_int(P, x, trace=True)
    out = os.path.join(ROOT, 'snn', 'golden', 'vectors', f'{prefix}_{out_name}'); os.makedirs(out, exist_ok=True)
    hexdump(os.path.join(out, 'feat.hex'), x[0], 8, False, "[T,n_mels] concatenated clips, no reset")
    for L in P['layers']:
        if L['type'] == 'out': continue
        hexdump(os.path.join(out, f"{L['name']}_v.hex"), g['traces'][L['name']]['v'][0], P['v_bits_hidden'], True, f"[T,N] membrane {L['name']}")
        hexdump(os.path.join(out, f"{L['name']}_s.hex"), g['traces'][L['name']]['s'][0], 1, False, f"[T,N] spikes {L['name']}")
    hexdump(os.path.join(out, 'out_v.hex'), g['vout'][0], P['v_bits_out'], True, "[T,C] output membrane")
    # per-clip argmax of the per-clip sum of vo (what a windowed decision with W=98 aligned to clips would give)
    T = 98; n = len(names)
    per_clip = [int(g['vout'][0, k*T:(k+1)*T].sum(0).argmax()) for k in range(n)]
    # expected RTL class after the last frame = argmax of the last WIN_SUM=98 frames (= last clip's window)
    info = dict(clips=names, T=int(x.shape[1]), n_mels=int(x.shape[2]), pred_sum=per_clip[-1], pred_total=int(g['pred_sum'][0]), per_clip_pred=per_clip,
                vout_sum=[int(v) for v in g['vout'][0].sum(0)], label=-1)
    json.dump(info, open(os.path.join(out, 'info.json'), 'w'), indent=1)
    print('stream:', out, 'T', x.shape[1], 'per-clip preds', per_clip)


if __name__ == '__main__':
    main()
