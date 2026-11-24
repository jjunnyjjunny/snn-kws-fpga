"""
level_sweep.py — what happens to the integer pipeline when the input level drops (board input-channel question):
  1. feature floor: 0.5 s sine at 300 / 1000 / 3000 / 6000 Hz, level -10 ... -80 dBFS -> max feature value (0 = below the
     log offset floor), i.e. the absolute level under which the front-end outputs nothing.
  2. dataset test clips (n per keyword) attenuated by 0 ... -42 dB -> clip-mode SNN prediction: keyword accuracy,
     per-class recall, where 'stop' goes (stop->up ...), feature zero ratio, E1 max, VAD trigger rate
     (longest run of E1 > T_on >= N_on), per gain.
  3. feature heat maps: dataset 'stop' clip at 0 / -12 / -24 / -36 dB next to board captures (near / far) if given.
usage: python snn/eval/level_sweep.py [--n 100] [--gains 0,-6,...] [--captures snn/data/board_capture/stop_near_02.csv,...]
"""
import os, sys, json, argparse
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from snn.frontend.frontend_ref import FrontendInt, wav_float_to_pcm24, apply_gain_pcm24
from snn.golden.snn_int_ref import run_int
from snn.data.make_dataset import KEYWORDS, load_item
SR = 16000; FS24 = 1 << 23; T_ON = 600; N_ON = 4
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']

def feats(fe, pcm, bs=100):
    return np.concatenate([fe.run(pcm[i:i + bs]) for i in range(0, len(pcm), bs)])

def vad_trigger(E):
    run = 0; best = 0
    for v in E: run = run + 1 if v > T_ON else 0; best = max(best, run)
    return best >= N_ON

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=os.path.join(ROOT, 'snn', 'runs', 'M3k3e60_U2_q8_kd_s0'))
    ap.add_argument('--data', default=os.path.join(ROOT, 'snn', 'data', 'feat_c12_m40_w480_U2.npz'))
    ap.add_argument('--n', type=int, default=100); ap.add_argument('--gains', default='0,-6,-12,-18,-24,-30,-36,-42')
    ap.add_argument('--captures', default=''); ap.add_argument('--out', default=os.path.join(ROOT, 'snn', 'runs', 'board_capture'))
    ap.add_argument('--fig', default=os.path.join(ROOT, 'snn', 'docs', 'figures'))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import torch; torch.backends.cuda.matmul.allow_tf32 = False
    from snn.golden.export_snn_mem import load_run, extract
    model, cfg, res = load_run(a.run); P = extract(model); classes = json.loads(str(np.load(a.data)['meta']))['classes']
    fe = FrontendInt(); out = {}
    # ---- 1. feature floor with sines
    t = np.arange(SR // 2) / SR; floor = {}
    for f in (300, 1000, 3000, 6000):
        row = {}
        for lv in range(-10, -85, -5):
            x = np.round(np.sin(2 * np.pi * f * t) * 10 ** (lv / 20) * FS24).astype(np.int64)
            ft = fe.run(x[None])[0]; row[lv] = int(ft.max())
        floor[f] = row
        z = [lv for lv, v in row.items() if v == 0]
        print(f"sine {f:4d} Hz: max feature per level " + ' '.join(f"{lv}:{v}" for lv, v in row.items()) + f"  -> first all-zero level {max(z) if z else None} dBFS")
    out['sine_floor'] = {str(k): {str(l): v for l, v in r.items()} for k, r in floor.items()}
    # ---- 2. gain sweep on test clips
    fl = json.load(open(os.path.join(ROOT, 'snn', 'data', 'filelist_c12.json')))['test']
    rng = np.random.default_rng(0); rels, ys = [], []
    for ci, w in enumerate(KEYWORDS):
        r_ = [r for r, l in fl if l == ci]; sel = rng.permutation(len(r_))[:a.n]; rels += [r_[i] for i in sel]; ys += [ci] * len(sel)
    ys = np.array(ys); pcm0 = np.stack([wav_float_to_pcm24(load_item(r)) for r in rels])
    gains = [int(g) for g in a.gains.split(',')]; sweep = []
    for g in gains:
        pcm = apply_gain_pcm24(pcm0, g) if g != 0 else pcm0
        X = feats(fe, pcm); E = X.astype(np.int64).sum(-1)
        gi = run_int(P, X, trace=False); pred = gi['vout'].sum(1).argmax(-1)
        rec = {w: float((pred[ys == ci] == ci).mean()) for ci, w in enumerate(KEYWORDS)}
        si = KEYWORDS.index('stop'); sp = pred[ys == si]; to = {classes[c]: int((sp == c).sum()) for c in np.unique(sp)}
        row = dict(gain_db=g, acc=float((pred == ys).mean()), recall=rec, stop_to=to, zero_ratio=float((X == 0).mean()), E1_max_median=float(np.median(E.max(1))),
                   vad_trigger=float(np.mean([vad_trigger(e) for e in E])), peak_dbfs=float(20 * np.log10(np.abs(pcm).max(1).mean() / FS24)))
        sweep.append(row)
        print(f"gain {g:4d} dB (peak {row['peak_dbfs']:6.1f} dBFS): acc {row['acc']:.3f} stop {rec['stop']:.2f} yes {rec['yes']:.2f} no {rec['no']:.2f} go {rec['go']:.2f} up {rec['up']:.2f} | "
              f"zero {row['zero_ratio']:.3f} E1max {row['E1_max_median']:.0f} VAD {row['vad_trigger']:.2f} | stop-> {to}")
    out['sweep'] = sweep
    json.dump(out, open(os.path.join(a.out, 'level_sweep.json'), 'w'), indent=1)
    L = ["| 감쇠 dB | peak dBFS(평균) | 키워드 정확도 | stop 재현율 | yes | no | go | up | 특징 0 비율 | E1 max 중앙값 | VAD 발동률 | stop → |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sweep:
        L.append(f"| {r['gain_db']} | {r['peak_dbfs']:.1f} | {r['acc']:.3f} | {r['recall']['stop']:.2f} | {r['recall']['yes']:.2f} | {r['recall']['no']:.2f} | {r['recall']['go']:.2f} | {r['recall']['up']:.2f} | {r['zero_ratio']:.3f} | {r['E1_max_median']:.0f} | {r['vad_trigger']:.2f} | "
                 + ', '.join(f"{k} {v}" for k, v in sorted(r['stop_to'].items(), key=lambda kv: -kv[1])) + " |")
    L.append("\n사인파 특징 바닥(최대 특징이 0이 되는 첫 레벨, dBFS): " + ', '.join(f"{f} Hz: {max([int(l) for l, v in r.items() if v == 0] or ['없음'])}" for f, r in out['sine_floor'].items()))
    open(os.path.join(a.out, 'level_sweep.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
    # ---- figures
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    gs = [r['gain_db'] for r in sweep]
    ax[0].plot(gs, [r['acc'] for r in sweep], '-o', color=PAL[0], lw=2, ms=5, label='keyword accuracy (10 classes)')
    ax[0].plot(gs, [r['recall']['stop'] for r in sweep], '-o', color=PAL[1], lw=2, ms=5, label="'stop' recall")
    ax[0].plot(gs, [r['vad_trigger'] for r in sweep], '-o', color=PAL[2], lw=2, ms=5, label='VAD trigger rate (E1>600, 4 frames)')
    ax[0].set_xlabel('attenuation (dB) of dataset test clips'); ax[0].set_ylim(0, 1.02); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7); ax[0].set_title('Integer pipeline vs input level')
    ax[1].plot(gs, [r['zero_ratio'] for r in sweep], '-o', color=PAL[0], lw=2, ms=5, label='feature zero ratio')
    ax[1].set_xlabel('attenuation (dB)'); ax[1].set_ylim(0, 1.02); ax[1].grid(alpha=.3); ax[1].legend(fontsize=7, loc='upper left'); ax[1].set_title('Features clipped to 0 by the log offset')
    fig.tight_layout(); fig.savefig(os.path.join(a.fig, 'board_level_sweep.png'), dpi=150); plt.close(fig)
    # ---- 3. feature heat maps
    caps = [c for c in a.captures.split(',') if c]
    if caps:
        from snn.eval.board_capture_check import read_capture
        si = KEYWORDS.index('stop'); idx = int(np.where(ys == si)[0][0]); x0 = pcm0[idx]
        panels = [(f"dataset 'stop' {g:+d} dB", fe.run(apply_gain_pcm24(x0[None], g) if g else x0[None])[0]) for g in (0, -12, -24, -36)]
        for c in caps:
            x18, _ = read_capture(c); panels.append((os.path.splitext(os.path.basename(c))[0] + ' (board)', fe.run((x18 << 6)[None])[0]))
        fig, axs = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.2), sharey=True)
        for ax_, (ttl, F) in zip(axs, panels):
            ax_.imshow(F.T, origin='lower', aspect='auto', cmap='Blues', vmin=0, vmax=160); ax_.set_title(ttl, fontsize=8); ax_.set_xlabel('frame (10 ms)')
        axs[0].set_ylabel('mel band'); fig.suptitle('8-bit features (0 = below the log offset floor)', fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(a.fig, 'board_feat.png'), dpi=150); plt.close(fig)

if __name__ == '__main__':
    main()
