"""fa_breakdown.py — 4-A step 0: decompose the false alarms of the VAD-segment decision on the cached test streams.
Each detection is classified: hit / wrong-class / FA-unknown (unknown utterance -> keyword) / FA-dup (second
detection inside an already-detected utterance) / FA-noise (no utterance overlaps the segment). Per predicted class too.
usage: python snn/eval/fa_breakdown.py [--cond board] [--margin 0]"""
import os, sys, json, argparse, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, ROOT)
from snn.eval import vad_eval as V
from snn.golden.snn_int_ref import run_int
KW = ['yes', 'no', 'stop', 'go', 'up', 'down', 'left', 'right', 'on', 'off', 'silence', 'unknown']

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cond', default='board'); ap.add_argument('--margin', type=int, default=0)
    ap.add_argument('--streams', default=''); ap.add_argument('--run', default=os.path.join(ROOT, 'snn', 'runs', 'M3k3e60_c12_q8_kd'))
    ap.add_argument('--out', default=''); a = ap.parse_args()
    rep = json.load(open(os.path.join(ROOT, 'snn', 'runs', 'vad_eval', 'report.json'))); bv = rep['best_val']
    prm = {k: bv[k] for k in ('T_on', 'T_off', 'N_on', 'H', 'P', 'L_min', 'L_max')}
    V._init(a.run); P = V._G['P']
    path = a.streams or os.path.join(ROOT, 'snn', 'runs', 'vad_eval', f'streams_test_{a.cond}.npz')
    streams = list(np.load(path, allow_pickle=True)['streams'])
    cnt = dict(hit=0, wrong=0, fa_unknown=0, fa_dup=0, fa_noise=0, fa_silence_event=0); per_cls = {c: dict(hit=0, wrong=0, fa_unknown=0, fa_dup=0, fa_noise=0) for c in KW[:10]}
    seg_stats = dict(n_seg=0, n_kept=0, seg_len=[]); hours = 0; n_kw = 0; n_unk = 0; unk_as_kw_by_true = {}
    for s in streams:
        E = s[bv['energy']]; segs = V.vad_segments(E, prm); hours += len(E) * 160 / 16000 / 3600
        seg_stats['n_seg'] += len(segs); seg_stats['n_kept'] += sum(k for _, _, k in segs)
        evs = s['events']; n_kw += sum(1 for e in evs if e['label'] < 10); n_unk += sum(1 for e in evs if e['label'] >= 10)
        used = {}
        for st, e, kept in segs:
            if not kept: continue
            seg_stats['seg_len'].append(e - st)
            g = run_int(P, s['feat'][None, st:e], trace=False); v = g['vout'][0].sum(0); o = np.sort(v)[::-1]; c = int(v.argmax()); m = int(o[0] - o[1])
            if c >= 10 or m < a.margin: continue                           # silence/unknown decision or below margin: no detection
            t = e - 1
            ov = [ (i, ev) for i, ev in enumerate(evs) if ev['onset'] <= t <= ev['offset'] + 50 or (st < ev['offset'] and e > ev['onset']) ]
            if not ov: kind = 'fa_noise'
            else:
                i, ev = ov[0]
                if ev['label'] >= 10: kind = 'fa_unknown'; unk_as_kw_by_true[c] = unk_as_kw_by_true.get(c, 0) + 1
                elif i in used: kind = 'fa_dup'
                else: used[i] = 1; kind = 'hit' if c == ev['label'] else 'wrong'
            cnt[kind] += 1; per_cls[KW[c]][kind] += 1
    fa = cnt['fa_unknown'] + cnt['fa_dup'] + cnt['fa_noise']
    res = dict(cond=a.cond, margin=a.margin, hours=hours, n_keyword_events=n_kw, n_unknown_events=n_unk, counts=cnt, fa_total=fa, fa_per_hour=fa / hours,
               fa_breakdown_per_hour={k: cnt[k] / hours for k in ('fa_unknown', 'fa_dup', 'fa_noise')}, per_class=per_cls,
               unknown_rate_as_keyword=cnt['fa_unknown'] / max(n_unk, 1), segments=dict(n=seg_stats['n_seg'], kept=seg_stats['n_kept'], len_median=float(np.median(seg_stats['seg_len'])) if seg_stats['seg_len'] else 0))
    print(json.dumps({k: v for k, v in res.items() if k != 'per_class'}, indent=1))
    print("per predicted class:"); [print(f"  {c:6s} {d}") for c, d in per_cls.items()]
    if a.out: json.dump(res, open(a.out, 'w'), indent=1)

if __name__ == '__main__': main()
