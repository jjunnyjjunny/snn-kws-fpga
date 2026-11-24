"""
det_curve.py — system-level comparison: detection rate vs false alarms/hour as the decision margin sweeps, for
SNN base / SNN U2 and (optionally) DS-CNN 24ch (base / U2 training) run through the same VAD-segment pipeline
(segment features zero-padded / cropped to 98 frames). Streams: cached val/test x board/talk (vad_eval cache).
Margins are swept over each model's own margin scale (SNN: output-sum top1-top2; DS-CNN: logit top1-top2).
Operating points are chosen on val (FA/h <= 10 and <= 3) and reported on test.
usage: python snn/eval/det_curve.py --workers 16
"""
import os, sys, json, argparse, time
import numpy as np
from multiprocessing import Pool
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, ROOT)
from snn.eval import vad_eval as V
from snn.golden.snn_int_ref import run_int
N_KEYWORD = 10; T = 98
_G = {}
MODELS = [('SNN base', 'snn', 'M3k3e60_c12_q8_kd'), ('SNN U2', 'snn', 'M3k3e60_U2_q8_kd_s0'),
          ('DS-CNN 24ch base (segment, 0-pad)', 'dscnn', 'dscnn_c12_e100'), ('DS-CNN 24ch U2-style (segment, 0-pad)', 'dscnn', 'dscnn_c12_U2_s0'),
          ('DS-CNN 24ch base (last 98 frames)', 'dscnn_last98', 'dscnn_c12_e100'), ('DS-CNN 24ch U2-style (last 98 frames)', 'dscnn_last98', 'dscnn_c12_U2_s0')]

def _init(run, kind):
    import torch; torch.backends.cuda.matmul.allow_tf32 = False
    if kind == 'snn':
        from snn.golden.export_snn_mem import load_run, extract
        m, cfg, res = load_run(run); _G['P'] = extract(m); _G['kind'] = 'snn'
    else:
        from snn.baseline.train_dscnn import DSCNN
        c = json.load(open(os.path.join(run, 'config.json'))); d = DSCNN(12, c.get('ch', 24), c['mean'], c['std'])
        d.load_state_dict(torch.load(os.path.join(run, 'best.pt'), map_location='cpu')); d.eval(); _G['net'] = d; _G['kind'] = kind; torch.set_num_threads(2)

def _segments(args):
    """-> per stream: list of (end_frame, cls, margin) for every kept segment (margin on the model's scale)."""
    s, prm, energy = args
    segs = V.vad_segments(s[energy], prm); out = []
    for st, e, kept in segs:
        if not kept: continue
        x = s['feat'][st:e]
        if _G['kind'] == 'snn':
            v = run_int(_G['P'], x[None], trace=False)['vout'][0].sum(0).astype(np.float64)
        else:
            import torch
            xx = np.zeros((T, x.shape[1]), np.uint8)
            if _G['kind'] == 'dscnn_last98':                       # the 98 real frames ending at the VAD end (front-padded only if e < 98)
                w = s['feat'][max(e - T, 0):e]; xx[T - len(w):] = w
            else:
                L = min(len(x), T); xx[:L] = x[:L]                   # segment, zero-padded / cropped to 98 frames
            with torch.no_grad(): v = _G['net'](torch.from_numpy(xx[None])).numpy()[0].astype(np.float64)
        o = np.sort(v)[::-1]; out.append((e - 1, int(v.argmax()), float(o[0] - o[1])))
    return out

def score(streams, dets_per_stream, margin, tail=50):
    hit = miss = wrong = fa = 0; n_kw = 0; hours = 0.0
    for s, dets in zip(streams, dets_per_stream):
        hours += len(s['E1']) * 160 / 16000 / 3600
        d2 = [(t, c) for t, c, m in dets if c < N_KEYWORD and m >= margin]; used = np.zeros(len(d2), bool)
        for e in s['events']:
            if e['label'] >= N_KEYWORD:
                for j, (t, c) in enumerate(d2):
                    if not used[j] and e['onset'] <= t <= e['offset'] + tail: used[j] = True; fa += 1
                continue
            n_kw += 1; got = None
            for j, (t, c) in enumerate(d2):
                if not used[j] and e['onset'] <= t <= e['offset'] + tail: used[j] = True; got = (t, c); break
            if got is None: miss += 1
            elif got[1] == e['label']: hit += 1
            else: wrong += 1
        fa += int((~used).sum())
    return dict(margin=margin, det=hit / max(n_kw, 1), wrong=wrong / max(n_kw, 1), miss=miss / max(n_kw, 1), fa_per_h=fa / max(hours, 1e-9))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=16); ap.add_argument('--out', default=os.path.join(ROOT, 'snn', 'runs', 'det_curve'))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    rep = json.load(open(os.path.join(ROOT, 'snn', 'runs', 'vad_eval', 'report.json'))); bv = rep['best_val']
    prm = {k: bv[k] for k in ('T_on', 'T_off', 'N_on', 'H', 'P', 'L_min', 'L_max')}; energy = bv['energy']
    S = {}
    for split in ('val', 'test'):
        for cond in ('board', 'talk'):
            S[(split, cond)] = list(np.load(os.path.join(ROOT, 'snn', 'runs', 'vad_eval', f'streams_{split}_{cond}.npz'), allow_pickle=True)['streams'])
    results = {}
    for name, kind, run in MODELS:
        rp = os.path.join(ROOT, 'snn', 'runs', run)
        if not os.path.exists(os.path.join(rp, 'results.json')): print('skip (not trained yet):', name); continue
        cache = os.path.join(a.out, f'segs_{run}_{kind}.npz')
        if os.path.exists(cache): segs = np.load(cache, allow_pickle=True)['segs'].item()
        else:
            segs = {}
            with Pool(a.workers, initializer=_init, initargs=(rp, kind)) as pool:
                for key, streams in S.items(): segs[key] = pool.map(_segments, [(s, prm, energy) for s in streams])
            np.savez(cache, segs=np.array(segs, dtype=object))
        # margin grid: 0 + percentiles of the val margins of all kept segments
        allm = np.array([m for key in segs if key[0] == 'val' for st in segs[key] for _, _, m in st])
        grid = [0.0] + [float(v) for v in np.percentile(allm, np.linspace(2, 95, 40))]
        res = {}
        for key in S:
            res[f'{key[0]}_{key[1]}'] = [score(S[key], segs[key], m) for m in grid]
        # operating points chosen on val (per condition): largest det with FA/h <= 10 and <= 3 -> reported on test
        ops = {}
        for cond in ('board', 'talk'):
            for lim in (10, 3):
                cands = [r for r in res[f'val_{cond}'] if r['fa_per_h'] <= lim]
                if cands:
                    best = max(cands, key=lambda r: r['det']); tr = next(r for r in res[f'test_{cond}'] if r['margin'] == best['margin'])
                    ops[f'{cond}_fa<={lim}'] = dict(margin=best['margin'], val=best, test=tr)
        results[name] = dict(run=run, kind=kind, curves=res, ops=ops)
        print(f"{name}: margins {len(grid)}, test board @margin0 det {res['test_board'][0]['det']:.3f} FA/h {res['test_board'][0]['fa_per_h']:.1f} | ops {[(k, round(v['test']['det'],3), round(v['test']['fa_per_h'],1)) for k, v in ops.items()]} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(results, open(os.path.join(a.out, 'det_curve.json'), 'w'), indent=1)
    # ---- figure
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    BG, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e0'
    C = {'SNN base': '#eb6834', 'SNN U2': '#e34948', 'DS-CNN 24ch base (segment, 0-pad)': '#2a78d6', 'DS-CNN 24ch U2-style (segment, 0-pad)': '#6250d6',
         'DS-CNN 24ch base (last 98 frames)': '#2a78d6', 'DS-CNN 24ch U2-style (last 98 frames)': '#6250d6'}
    LS = lambda n: '--' if 'last 98' in n else '-'
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150); fig.patch.set_facecolor(BG)
    for ax, cond in zip(axs, ('board', 'talk')):
        for name, r in results.items():
            cv = r['curves'][f'test_{cond}']; fa = np.array([c['fa_per_h'] for c in cv]); det = np.array([100 * c['det'] for c in cv])
            ax.plot(np.maximum(fa, 0.3), det, LS(name), marker='o', ms=3.0, lw=1.6, color=C.get(name, INK2), label=name, markeredgecolor=BG, markeredgewidth=0.8)
            for k, op in r['ops'].items():
                if k.startswith(cond): ax.plot(max(op['test']['fa_per_h'], 0.3), 100 * op['test']['det'], marker='D', ms=7, color=C.get(name, INK2), markeredgecolor=BG)
        ax.set_xscale('log'); ax.set_xlabel('false alarms per hour (test, log; ◆ = operating points chosen on val for FA ≤ 10 and ≤ 3/h)', color=INK, fontsize=8)
        ax.set_ylabel('keyword detection rate (%)', color=INK); ax.set_title(f"{'board condition' if cond == 'board' else 'conversation-heavy'} test streams (VAD-segment pipeline)", fontsize=10, color=INK)
        ax.set_facecolor(BG); ax.grid(True, which='major', color=GRID, lw=0.8); ax.grid(False, which='minor')
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'): ax.spines[sp].set_color('#c3c2b7')
        ax.tick_params(colors=INK2); ax.legend(frameon=False, fontsize=7, loc='lower right')
    fig.tight_layout(); F = os.path.join(ROOT, 'snn', 'docs', 'figures'); os.makedirs(F, exist_ok=True)
    fig.savefig(os.path.join(F, 'det_curve.png')); fig.savefig(os.path.join(a.out, 'det_curve.png'))
    import csv
    with open(os.path.join(F, 'det_curve.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['model', 'split_cond', 'margin', 'det', 'wrong', 'miss', 'fa_per_h'])
        for name, r in results.items():
            for key, cv in r['curves'].items():
                for c in cv: w.writerow([name, key, f"{c['margin']:.2f}", f"{c['det']:.4f}", f"{c['wrong']:.4f}", f"{c['miss']:.4f}", f"{c['fa_per_h']:.2f}"])
    print('saved', os.path.join(F, 'det_curve.png'))

if __name__ == '__main__': main()
