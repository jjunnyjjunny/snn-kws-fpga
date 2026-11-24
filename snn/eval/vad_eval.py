"""
vad_eval.py — B: VAD-segment decision (replaces the sliding-window rule). Integer pipeline:
  E[t] (frame energy) -> VAD state machine (T_on/T_off hysteresis, N_on, hangover H, L_min, L_max)
  -> on trigger: SNN reset + pre-roll P frames + live frames -> at segment end: argmax of summed output
  membranes (silence/unknown = no detection, optional top1-top2 margin). SNN runs only inside segments.
Parameters are searched on VAL streams, reported on TEST streams, in two conditions:
  'w2'    : week-2 condition  (gain U(-20,+10) dB, background volume U(0, 0.1))
  'board' : board condition   (gain U(-6,+6) dB,  background volume U(0, 0.05))
Compared with the week-2 sliding rule (W=50, M=6000, R=100) and the oracle event window (reset at onset).
usage: python snn/eval/vad_eval.py --n_val 100 --n_test 200 --workers 16
"""
import os, sys, json, argparse, time, itertools
import numpy as np, soundfile as sf
from multiprocessing import Pool
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from snn.frontend.frontend_ref import FrontendInt, wav_float_to_pcm24, apply_gain_pcm24, log2_int, make_log_lut
from snn.data.make_dataset import load_item, DATA
from snn.golden.snn_int_ref import run_int
from snn.eval.stream_eval import decide as sliding_decide, score as sliding_score
SR = 16000; HOP = 160; N_KEYWORD = 10
COND = {'w2': dict(gain=(-20.0, 10.0), vol=(0.0, 0.1), unk=None, gap=(0.5, 3.0), dur=30.0),
        'board': dict(gain=(-6.0, 6.0), vol=(0.0, 0.05), unk=None, gap=(0.5, 3.0), dur=30.0),
        'talk': dict(gain=(-6.0, 6.0), vol=(0.0, 0.05), unk=0.6, gap=(0.5, 3.0), dur=30.0),      # 60 % non-keyword utterances
        'quiet': dict(gain=(-6.0, 6.0), vol=(0.0, 0.05), unk=0.5, gap=(8.0, 25.0), dur=60.0)}    # >= 90 % silence/noise
_G = {}

# ----------------------------------------------------------------------------- frame energy (integer, RTL-cheap)
def energy_sum_feat(feat):            # [T,40] uint8 -> [T] 14-bit sum of the 8-bit features
    return feat.astype(np.int64).sum(-1)
def energy_log2_mel(mel_acc):         # [T,40] mel accumulators -> log2q(sum of accumulators) (10 bit)
    return log2_int(mel_acc.sum(-1), make_log_lut())

# ----------------------------------------------------------------------------- VAD state machine (frame-level, integer)
class VAD:
    """step(E) -> event: None | ('start', feed_from_frame) | ('end', n_fed) | ('force', n_fed)
    Frames fed to the SNN: pre-roll P frames before the trigger frame, the trigger frame, then every live frame
    until the end. Segment is discarded at end if fewer than L_min frames were fed."""
    def __init__(self, T_on, T_off, N_on, H, P, L_min, L_max):
        self.T_on, self.T_off, self.N_on, self.H, self.P, self.L_min, self.L_max = T_on, T_off, N_on, H, P, L_min, L_max
        self.active = False; self.cnt_on = 0; self.cnt_off = 0; self.n_fed = 0; self.t = -1
    def step(self, E):
        self.t += 1
        if not self.active:
            self.cnt_on = self.cnt_on + 1 if E > self.T_on else 0
            if self.cnt_on >= self.N_on:
                self.active = True; self.cnt_on = 0; self.cnt_off = 0
                self.n_fed = self.P + 1                                  # pre-roll + trigger frame
                return ('start', max(self.t - self.P, 0))
            return None
        self.n_fed += 1
        self.cnt_off = self.cnt_off + 1 if E < self.T_off else 0
        if self.n_fed >= self.L_max:
            self.active = False; return ('force', self.n_fed)
        if self.cnt_off >= self.H:
            self.active = False; return ('end', self.n_fed)
        return None

def vad_segments(E, prm):
    """-> list of (feed_start, feed_end_exclusive, kept) over a whole stream."""
    v = VAD(**prm); segs = []; start = None
    for t in range(len(E)):
        ev = v.step(int(E[t]))
        if ev is None: continue
        if ev[0] == 'start': start = ev[1]
        else:
            n = ev[1]; segs.append((start, t + 1, n >= prm['L_min'])); start = None
    if start is not None: segs.append((start, len(E), (len(E) - start) >= prm['L_min']))
    return segs

def segment_decisions(P, feat, segs, margin=0):
    """SNN reset per segment; decision at segment end frame. -> list of (frame, class, margin, n_frames)"""
    dets = []
    for s, e, kept in segs:
        if not kept: continue
        g = run_int(P, feat[None, s:e], trace=False)
        v = g['vout'][0].sum(0); o = np.sort(v)[::-1]
        c = int(v.argmax()); m = int(o[0] - o[1])
        if c < N_KEYWORD and m >= margin: dets.append((e - 1, c, m, e - s))
    return dets

# ----------------------------------------------------------------------------- streams
def _init(run):
    import torch; torch.backends.cuda.matmul.allow_tf32 = False
    from snn.golden.export_snn_mem import load_run, extract
    model, cfg, res = load_run(run); _G['P'] = extract(model); _G['fe'] = FrontendInt()
    fl = json.load(open(os.path.join(ROOT, 'snn', 'data', 'filelist_c12.json')))
    _G['files'] = {sp: [(r, y) for r, y in fl[sp] if y != 10] for sp in ('val', 'test')}
    names = sorted(f for f in os.listdir(os.path.join(DATA, '_background_noise_')) if f.endswith('.wav'))
    noise = {'val': [], 'test': []}
    for f in names:
        x, _ = sf.read(os.path.join(DATA, '_background_noise_', f), dtype='float32')
        noise['val'].append(x[int(0.8 * len(x)): int(0.9 * len(x))]); noise['test'].append(x[int(0.9 * len(x)):])
    _G['noise'] = {k: np.concatenate(v) for k, v in noise.items()}

def _build(args):
    split, cond, k, dur = args
    rng = np.random.default_rng((7919 if split == 'val' else 1000) + k + {'w2': 0, 'board': 50000, 'talk': 60000, 'quiet': 70000}[cond])
    c = COND[cond]; dur = c['dur']; n = int(dur * SR); noise = _G['noise'][split]
    if len(noise) < 2 * n: noise = np.tile(noise, int(np.ceil(2 * n / len(noise))))      # val/test noise regions are short
    vol = rng.uniform(*c['vol']); st = int(rng.integers(0, len(noise) - n))
    wav = np.round(noise[st: st + n].astype(np.float64) * vol * (1 << 23)).astype(np.int64)
    files, labels = zip(*_G['files'][split]); events = []
    kw_idx = [i for i, y in enumerate(labels) if y < N_KEYWORD]; unk_idx = [i for i, y in enumerate(labels) if y >= N_KEYWORD]
    pos = int(rng.uniform(*c['gap']) * SR)
    while pos + SR <= n:
        if c['unk'] is None: i = int(rng.integers(len(files)))
        else: i = int(rng.choice(unk_idx)) if rng.random() < c['unk'] else int(rng.choice(kw_idx))
        wav[pos: pos + SR] += apply_gain_pcm24(wav_float_to_pcm24(load_item(files[i])), rng.uniform(*c['gain']))
        events.append(dict(onset=pos // HOP, offset=(pos + SR) // HOP, label=int(labels[i])))
        pos += SR + int(rng.uniform(*c['gap']) * SR)
    wav = np.clip(wav, -(1 << 23), (1 << 23) - 1)
    d = _G['fe'].run_debug(wav[None])
    feat = d['feat'][0]
    return dict(k=k, split=split, cond=cond, events=events, feat=feat, E1=energy_sum_feat(feat).astype(np.int32),
                E2=energy_log2_mel(d['mel'][0]).astype(np.int32), wav=None)

def _eval_params(args):
    prm, margin, energy, key = args
    streams = _G[key] if isinstance(key, str) else key
    dets_all = []; duty = 0; tot = 0; iou = []
    for s in streams:
        E = s[energy]; segs = vad_segments(E, prm)
        dets = segment_decisions(_G['P'], s['feat'], segs, margin)
        dets_all.append(dets)
        tot += len(E); duty += sum(e - a for a, e, kept in segs if kept)
        for ev in s['events']:
            best = 0.0
            for a, e, kept in segs:
                inter = max(0, min(e, ev['offset']) - max(a, ev['onset'])); uni = max(e, ev['offset']) - min(a, ev['onset'])
                best = max(best, inter / uni)
            iou.append(best)
    r = score_vad(streams, dets_all); r.update(prm); r.update(margin=margin, energy=energy, snn_duty=duty / tot, vad_event_iou=float(np.mean(iou)))
    return r

def score_vad(streams, dets_all, tail=50):
    hit = miss = wrong = fa = 0; lat = []; n_kw = 0; hours = 0.0
    for s, dets in zip(streams, dets_all):
        T = len(s['E1']); hours += T * HOP / SR / 3600
        used = np.zeros(len(dets), bool)
        for e in s['events']:
            if e['label'] >= N_KEYWORD:
                for j, d in enumerate(dets):
                    if not used[j] and e['onset'] <= d[0] <= e['offset'] + tail: used[j] = True; fa += 1
                continue
            n_kw += 1; got = None
            for j, d in enumerate(dets):
                if not used[j] and e['onset'] <= d[0] <= e['offset'] + tail: used[j] = True; got = d; break
            if got is None: miss += 1
            elif got[1] == e['label']: hit += 1; lat.append(got[0] - e['offset'])
            else: wrong += 1
        fa += int((~used).sum())
    return dict(n_keyword=n_kw, det_rate=hit / max(n_kw, 1), wrong_rate=wrong / max(n_kw, 1), miss_rate=miss / max(n_kw, 1),
                seg_acc=hit / max(hit + wrong, 1),
                false_alarms_per_hour=fa / max(hours, 1e-9), hours=hours, latency_median=float(np.median(lat)) if lat else None,
                latency_p90=float(np.percentile(lat, 90)) if lat else None)

def oracle(streams):
    P = _G['P']; n = ok = 0
    for s in streams:
        for e in s['events']:
            if e['label'] >= N_KEYWORD: continue
            n += 1; g = run_int(P, s['feat'][None, e['onset']:e['offset']], trace=False); ok += int(g['pred_sum'][0] == e['label'])
    return ok / max(n, 1)

def sliding(streams, W=50, M=6000, R=100):
    P = _G['P']; out = []
    for s in streams:
        g = run_int(P, s['feat'][None], trace=False)
        out.append(dict(vout=g['vout'][0], events=s['events']))
    return sliding_score(out, W, M, 1, R)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=os.path.join(ROOT, 'snn', 'runs', 'M3k3e60_c12_q8_kd'))
    ap.add_argument('--n_val', type=int, default=100); ap.add_argument('--n_test', type=int, default=200); ap.add_argument('--dur', type=float, default=30.0)
    ap.add_argument('--workers', type=int, default=16); ap.add_argument('--out', default=os.path.join(ROOT, 'snn', 'runs', 'vad_eval'))
    ap.add_argument('--regrid', action='store_true')
    ap.add_argument('--conds', default='w2,board', help='stream conditions to build/evaluate')
    ap.add_argument('--eval_only', action='store_true', help='no grid: evaluate the run with the params of --params_from on the given conds')
    ap.add_argument('--params_from', default=os.path.join(ROOT, 'snn', 'runs', 'vad_eval', 'report.json'))
    ap.add_argument('--tag', default='', help='output name suffix for eval-only reports')
    ap.add_argument('--margins', default='0', help='comma list of margins for eval-only')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    S = {}; conds = a.conds.split(',')
    with Pool(a.workers, initializer=_init, initargs=(a.run,)) as pool:       # parent stays torch/BLAS-free before forking
        for split, n in (('val', a.n_val), ('test', a.n_test)):
            for cond in conds:
                cache = os.path.join(a.out, f'streams_{split}_{cond}.npz')
                if os.path.exists(cache): S[(split, cond)] = list(np.load(cache, allow_pickle=True)['streams'])
                else:
                    S[(split, cond)] = pool.map(_build, [(split, cond, k, a.dur) for k in range(n)])
                    np.savez(cache, streams=np.array(S[(split, cond)], dtype=object))
                print(f"{split}/{cond}: {len(S[(split,cond)])} streams, keyword events {sum(1 for s in S[(split,cond)] for e in s['events'] if e['label']<N_KEYWORD)} ({time.time()-t0:.0f}s)", flush=True)
    for cond in conds: _G['val_' + cond] = S[('val', cond)]; _G['test_' + cond] = S[('test', cond)]
    if a.eval_only:
        rep0 = json.load(open(a.params_from)); bv = rep0['best_val']; bp = {k: bv[k] for k in ('T_on', 'T_off', 'N_on', 'H', 'P', 'L_min', 'L_max')}
        out = dict(run=a.run, params=bp, energy=bv['energy'], results={})
        with Pool(a.workers, initializer=_init, initargs=(a.run,)) as pool:
            for split in ('val', 'test'):
                for cond in conds:
                    key = f'{split}_{cond}'; out['results'][key] = {}
                    out['results'][key]['oracle_acc'] = pool.apply(_oracle_key, (key,))
                    for m in [int(v) for v in a.margins.split(',')]:
                        r = pool.apply(_eval_params, ((bp, m, bv['energy'], key),)); out['results'][key][f'margin_{m}'] = r
                        print(f"{key} margin {m}: det {r['det_rate']:.3f} wrong {r['wrong_rate']:.3f} miss {r['miss_rate']:.3f} FA/h {r['false_alarms_per_hour']:.1f} duty {r['snn_duty']:.3f} (oracle {out['results'][key]['oracle_acc']:.3f})", flush=True)
        name = os.path.basename(a.run.rstrip('/')) + (('_' + a.tag) if a.tag else '')
        json.dump(out, open(os.path.join(a.out, f'eval_{name}.json'), 'w'), indent=1, default=float); print('saved', name); return
    _G['val'] = S[('val', 'w2')] + S[('val', 'board')]
    grid = []
    for energy in ('E1', 'E2'):
        Ton_list = [300, 600, 1000, 1500, 2200] if energy == 'E1' else [340, 360, 380, 400, 420]
        for T_on in Ton_list:
            for ratio in (0.3, 0.6):
                for N_on in (2, 4):
                    for H in (15, 30):
                        for Pr in (10, 20):
                            for L_max in (98, 130):
                                grid.append((dict(T_on=T_on, T_off=int(T_on * ratio) if energy == 'E1' else T_on - int((1 - ratio) * 30), N_on=N_on, H=H, P=Pr, L_min=25, L_max=L_max), 0, energy, 'val'))
    print(f"grid: {len(grid)} parameter sets", flush=True)
    with Pool(a.workers, initializer=_init, initargs=(a.run,)) as pool:        # streams inherited by fork (no pickling)
        gpath = os.path.join(a.out, 'val_grid.json')
        res = json.load(open(gpath)) if (os.path.exists(gpath) and not a.regrid) else pool.map(_eval_params, grid)
        for r in res: r['objective'] = r['det_rate'] - r['wrong_rate'] - 0.002 * r['false_alarms_per_hour']
        res.sort(key=lambda r: -r['objective'])
        json.dump(res, open(os.path.join(a.out, 'val_grid.json'), 'w'), indent=1)
        print("top val parameter sets:")
        for r in res[:8]: print({k: r[k] for k in ('energy', 'T_on', 'T_off', 'N_on', 'H', 'P', 'L_max')}, f"det {r['det_rate']:.3f} wrong {r['wrong_rate']:.3f} FA/h {r['false_alarms_per_hour']:.1f} duty {r['snn_duty']:.3f} iou {r['vad_event_iou']:.2f}", flush=True)
        best = res[0]
        bp = {k: best[k] for k in ('T_on', 'T_off', 'N_on', 'H', 'P', 'L_min', 'L_max')}
        # margin candidates = percentiles of the top1-top2 margin of the VAD segments on val (keyword events)
        mg = pool.apply(_segment_margins, (bp, best['energy'], 'val'))
        M_list = [0] + [int(v) for v in np.percentile(mg, [10, 20, 30, 40, 50, 60])]
        print("val segment margin percentiles 10..60:", M_list[1:], flush=True)
        mres = pool.map(_eval_params, [(bp, m, best['energy'], 'val') for m in M_list])
        for r in mres: print(f"  margin {r['margin']:6d}: det {r['det_rate']:.3f} wrong {r['wrong_rate']:.3f} FA/h {r['false_alarms_per_hour']:.1f}", flush=True)
        lowfa = min([r for r in mres if r['false_alarms_per_hour'] <= 15.0] or [min(mres, key=lambda r: r['false_alarms_per_hour'])], key=lambda r: -r['det_rate'])
        report = dict(best_val=best, lowfa_val=lowfa, margin_sweep_val=mres, test={})
        for cond in COND:
            key = 'test_' + cond
            rows = dict(sliding_w2rule=pool.apply(_sliding_key, (key,)), oracle_acc=pool.apply(_oracle_key, (key,)),
                        vad_recommended=pool.apply(_eval_params, ((bp, 0, best['energy'], key),)),
                        vad_low_fa=pool.apply(_eval_params, ((bp, lowfa['margin'], best['energy'], key),)))
            report['test'][cond] = rows
            print(f"== TEST {cond}: oracle acc {rows['oracle_acc']:.3f} | sliding det {rows['sliding_w2rule']['det_rate']:.3f} wrong {rows['sliding_w2rule']['wrong_rate']:.3f} FA/h {rows['sliding_w2rule']['false_alarms_per_hour']:.1f}")
            for k in ('vad_recommended', 'vad_low_fa'):
                r = rows[k]; print(f"   {k}: det {r['det_rate']:.3f} wrong {r['wrong_rate']:.3f} miss {r['miss_rate']:.3f} FA/h {r['false_alarms_per_hour']:.1f} lat {r['latency_median']} duty {r['snn_duty']:.3f} iou {r['vad_event_iou']:.2f} margin {r['margin']}", flush=True)
        json.dump(report, open(os.path.join(a.out, 'report.json'), 'w'), indent=1, default=float)
        print('done', time.time() - t0)

def _sliding_key(key): return sliding(_G[key])
def _segment_margins(prm, energy, key):
    out = []
    for s in _G[key]:
        segs = vad_segments(s[energy], prm)
        for st, e, kept in segs:
            if not kept: continue
            if not any(ev['label'] < N_KEYWORD and st <= ev['offset'] + 50 and e - 1 >= ev['onset'] for ev in s['events']): continue
            g = run_int(_G['P'], s['feat'][None, st:e], trace=False); v = g['vout'][0].sum(0); o = np.sort(v)[::-1]; out.append(int(o[0] - o[1]))
    return out
def _oracle_key(key): return oracle(_G[key])

if __name__ == '__main__':
    main()
