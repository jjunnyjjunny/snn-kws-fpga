"""
stream_eval.py — B-4 streaming evaluation of the integer SNN (no state reset between utterances).

1. Build long test streams (default 200 x 30 s): test-split keyword / unknown utterances placed with random
   gaps (0.5..3 s), per-utterance random gain (-20..+10 dB, 24-bit saturation), continuous background noise
   from the test region of _background_noise_ files (volume U(0, 0.1)). Ground truth = (onset, offset, class).
2. Integer front-end -> features [T, 40] ; integer SNN streaming inference (snn_int_ref.run_int on the
   whole sequence = state carried across utterances) -> vout [T, C] (also saved).
3. Decision rules on vout: window W in {50,75,98}, margin M, consecutive N, refractory R.
   Detection = frame where the rule fires with class < N_KEYWORD.
4. Metrics: keyword detection rate, misclassification rate, false alarms per hour, latency (frames after offset).

usage: python snn/eval/stream_eval.py --run snn/runs/M3k3e60_c12_q8_kd --n_streams 200 --workers 16
"""
import os, sys, json, argparse, time, itertools
import numpy as np, soundfile as sf
from multiprocessing import Pool
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from snn.frontend.frontend_ref import FrontendInt, wav_float_to_pcm24, apply_gain_pcm24
from snn.data.make_dataset import load_item, DATA, KEYWORDS
from snn.golden.snn_int_ref import run_int

SR = 16000; HOP = 160; N_KEYWORD = 10
_G = {}

def build_stream(rng, files, labels, noise, dur_s=30.0, gap=(0.5, 3.0), gain=(-20.0, 10.0)):
    n = int(dur_s * SR)
    vol = rng.uniform(0.0, 0.1)
    st = int(rng.integers(0, len(noise) - n))
    wav = np.round(noise[st: st + n].astype(np.float64) * vol * (1 << 23)).astype(np.int64)
    events = []
    pos = int(rng.uniform(*gap) * SR)
    while pos + SR <= n:
        i = int(rng.integers(len(files)))
        clip = apply_gain_pcm24(wav_float_to_pcm24(load_item(files[i])), rng.uniform(*gain))
        wav[pos: pos + SR] += clip
        events.append(dict(onset=pos // HOP, offset=(pos + SR) // HOP, label=int(labels[i]), file=files[i]))
        pos += SR + int(rng.uniform(*gap) * SR)
    wav = np.clip(wav, -(1 << 23), (1 << 23) - 1)
    return wav, events

def _init(run, seed):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    from snn.golden.export_snn_mem import load_run, extract
    model, cfg, res = load_run(run)
    _G['P'] = extract(model); _G['fe'] = FrontendInt()
    fl = json.load(open(os.path.join(ROOT, 'snn', 'data', 'filelist_c12.json')))['test']
    fl = [(r, y) for r, y in fl if y != 10]                       # keywords + unknown (no silence clips)
    _G['files'] = [r for r, _ in fl]; _G['labels'] = [y for _, y in fl]
    names = sorted(f for f in os.listdir(os.path.join(DATA, '_background_noise_')) if f.endswith('.wav'))
    noise = []
    for f in names:
        x, _ = sf.read(os.path.join(DATA, '_background_noise_', f), dtype='float32')
        noise.append(x[int(0.9 * len(x)):])                       # test region only
    _G['noise'] = np.concatenate(noise)

def _work(args):
    k, dur = args
    rng = np.random.default_rng(1000 + k)
    wav, events = build_stream(rng, _G['files'], _G['labels'], _G['noise'], dur)
    feat = _G['fe'].run(wav[None])                                   # [1, T, 40]
    g = run_int(_G['P'], feat, trace=False)
    return dict(k=k, events=events, vout=g['vout'][0].astype(np.int32), feat_zero=float((feat == 0).mean()), feat=feat[0])

def decide(vout, W, M, N, R, n_keyword=N_KEYWORD):
    """returns list of (frame, class) detections."""
    T, C = vout.shape
    cs = np.cumsum(np.concatenate([np.zeros((1, C), np.int64), vout.astype(np.int64)], 0), 0)
    idx = np.arange(1, T + 1); lo = np.maximum(idx - W, 0)
    win = cs[idx] - cs[lo]                                            # [T, C] sum of last W frames
    order = np.argsort(-win, axis=1)
    top = order[:, 0]; margin = win[np.arange(T), top] - win[np.arange(T), order[:, 1]]
    dets = []; consec = 0; prev = -1; refr = 0
    for t in range(T):
        c = int(top[t])
        consec = consec + 1 if c == prev else 1
        prev = c
        if refr > 0:
            refr -= 1; continue
        if c < n_keyword and margin[t] >= M and consec >= N:
            dets.append((t, c)); refr = R; consec = 0; prev = -1
    return dets

def score(streams, W, M, N, R, tail=50):
    hit = miss = wrong = fa = 0; lat = []; n_kw = 0; hours = 0.0
    for s in streams:
        dets = decide(s['vout'], W, M, N, R)
        T = s['vout'].shape[0]; hours += T * HOP / SR / 3600
        used = np.zeros(len(dets), bool)
        for e in s['events']:
            if e['label'] >= N_KEYWORD:                                # unknown utterance: any detection = false alarm
                for j, (t, c) in enumerate(dets):
                    if not used[j] and e['onset'] <= t <= e['offset'] + tail: used[j] = True; fa += 1
                continue
            n_kw += 1
            got = None
            for j, (t, c) in enumerate(dets):
                if not used[j] and e['onset'] <= t <= e['offset'] + tail:
                    used[j] = True; got = (t, c); break
            if got is None: miss += 1
            elif got[1] == e['label']: hit += 1; lat.append(got[0] - e['offset'])
            else: wrong += 1
        fa += int((~used).sum())                                       # detections outside any utterance window
    return dict(W=W, M=M, N=N, R=R, n_keyword_events=n_kw, det_rate=hit / max(n_kw, 1), wrong_rate=wrong / max(n_kw, 1),
                miss_rate=miss / max(n_kw, 1), false_alarms_per_hour=fa / max(hours, 1e-9), hours=hours,
                latency_median_frames=float(np.median(lat)) if lat else None, latency_p90_frames=float(np.percentile(lat, 90)) if lat else None)

def oracle_analysis(streams, P):
    """Per keyword event: argmax of sum(vout) over [onset, offset) with (a) carried-over streaming state,
    (b) state reset at onset (clip re-run in isolation on the same noisy/gained audio is not available here,
    so (b) is approximated by re-running the SNN on the event's feature window alone)."""
    carried = reset = n = 0
    for s in streams:
        feat = s.get('feat')
        for e in s['events']:
            if e['label'] >= N_KEYWORD: continue
            n += 1
            w = s['vout'][e['onset']:e['offset']].astype(np.int64).sum(0)
            carried += int(w.argmax() == e['label'])
            if feat is not None:
                g = run_int(P, feat[None, e['onset']:e['offset']], trace=False)
                reset += int(g['pred_sum'][0] == e['label'])
    return dict(n=n, acc_carried_state=carried / max(n, 1), acc_reset_at_onset=reset / max(n, 1) if streams[0].get('feat') is not None else None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=os.path.join(ROOT, 'snn', 'runs', 'M3k3e60_c12_q8_kd'))
    ap.add_argument('--n_streams', type=int, default=200); ap.add_argument('--dur', type=float, default=30.0)
    ap.add_argument('--workers', type=int, default=16); ap.add_argument('--out', default=os.path.join(ROOT, 'snn', 'runs', 'stream_eval'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    cache = os.path.join(a.out, 'streams.npz')
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True); streams = list(z['streams'])
    else:
        with Pool(a.workers, initializer=_init, initargs=(a.run, 0)) as pool:
            streams = pool.map(_work, [(k, a.dur) for k in range(a.n_streams)])
        np.savez(cache, streams=np.array(streams, dtype=object))
    print(f"streams: {len(streams)} x {a.dur}s, events {sum(len(s['events']) for s in streams)}, feat zero ratio {np.mean([s['feat_zero'] for s in streams]):.3f} ({time.time()-t0:.0f}s)")
    _init(a.run, 0); P = _G['P']
    orc = oracle_analysis(streams, P)
    print("oracle event windows:", orc)
    # margin candidates: percentiles of the top1-top2 margin (W=75) at keyword-event offsets
    mg = []
    for s in streams:
        T, C = s['vout'].shape
        cs = np.cumsum(np.concatenate([np.zeros((1, C), np.int64), s['vout'].astype(np.int64)], 0), 0)
        for e in s['events']:
            if e['label'] >= N_KEYWORD: continue
            t = min(e['offset'], T); w = cs[t] - cs[max(t - 75, 0)]; o = np.sort(w)[::-1]; mg.append(o[0] - o[1])
    mg = np.array(mg); print("event-offset margin percentiles (W=75) 10/25/50/75:", np.percentile(mg, [10, 25, 50, 75]).astype(int))
    M_list = [0] + [int(v) for v in np.percentile(mg, [5, 10, 20, 30, 40, 50])]
    rows = []
    for W in (50, 75, 98):
        for M in M_list:
            for N in (1, 3, 5, 10):
                for R in (0, 50, 100):
                    rows.append(score(streams, W, M, N, R))
    json.dump(rows, open(os.path.join(a.out, 'rules.json'), 'w'), indent=1)
    rows.sort(key=lambda r: (-(r['det_rate'] - r['wrong_rate']) + 0.002 * r['false_alarms_per_hour']))
    print("top rules (det − wrong − 0.002·FA/h):")
    for r in rows[:12]:
        print(f"W={r['W']:3d} M={r['M']:6d} N={r['N']:2d} R={r['R']:3d} det={r['det_rate']:.3f} wrong={r['wrong_rate']:.3f} miss={r['miss_rate']:.3f} FA/h={r['false_alarms_per_hour']:.1f} lat_med={r['latency_median_frames']}")
    base = [r for r in rows if r['W'] == 98 and r['M'] == 0 and r['N'] == 1 and r['R'] == 0][0]
    print("baseline W=98 M=0 N=1 R=0:", base)
    # also the "sweet spot" view: best detection rate subject to FA/h <= 5 and wrong <= 3 %
    ok = [r for r in rows if r['false_alarms_per_hour'] <= 5 and r['wrong_rate'] <= 0.03]
    ok.sort(key=lambda r: -r['det_rate'])
    print("best with FA/h<=5 and wrong<=3%:", ok[0] if ok else None)
    json.dump(dict(n_streams=len(streams), dur=a.dur, oracle=orc, margin_percentiles=[int(v) for v in np.percentile(mg, [10, 25, 50, 75])],
                   best=rows[:12], best_constrained=ok[:5], baseline=base), open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)

if __name__ == '__main__':
    main()
