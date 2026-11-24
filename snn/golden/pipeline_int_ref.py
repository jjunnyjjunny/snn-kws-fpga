"""
pipeline_int_ref.py — B-4 integer golden of the whole live pipeline, frame by frame, mirroring snn/rtl/kws_top.v:
  pcm24 samples -> integer front-end (frame t = x[160t:160t+480]) -> E[t] -> VAD -> (start: SNN reset + pre-roll
  replay | active: feed frame) -> segment sum of output membranes -> decision at segment end.
Per frame it records: active, seg_start, seg_end, fed, det_any, det_valid, det_class (+ segment sums at decisions).
usage (dump test streams for tb_pipeline):
  python snn/golden/pipeline_int_ref.py --dump 4 --dur 8 --T_on 1000 --T_off 600 --N_on 2 --H 30 --P 20 --L_min 25 --L_max 98 --energy E1
"""
import os, sys, json, argparse
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from snn.frontend.frontend_ref import FrontendInt, log2_int, make_log_lut
from snn.golden.snn_int_ref import SnnStepper
from snn.eval.vad_eval import VAD, N_KEYWORD

def run_pipeline(P, pcm24, prm, energy='E1', margin=0):
    fe = FrontendInt(); d = fe.run_debug(np.asarray(pcm24, dtype=np.int64)[None])
    feat = d['feat'][0]; T = feat.shape[0]
    E = feat.astype(np.int64).sum(-1) if energy == 'E1' else log2_int(d['mel'][0].sum(-1), make_log_lut())
    vad = VAD(**{k: prm[k] for k in ('T_on', 'T_off', 'N_on', 'H', 'P', 'L_min', 'L_max')})
    snn = SnnStepper(P); seg_sum = None; rec = []; dets = []
    for t in range(T):
        ev = vad.step(int(E[t])); r = dict(t=t, E=int(E[t]), active=int(vad.active), seg_start=0, seg_end=0, fed=0, det_any=0, det_valid=0, det_class=0, keep=0)
        if ev is not None and ev[0] == 'start':
            r['seg_start'] = 1; snn.reset(); seg_sum = np.zeros(len(snn.vo), np.int64)
            for tt in range(ev[1], t + 1): seg_sum += snn.step(feat[tt])
            r['fed'] = t + 1 - ev[1]
        elif ev is not None:                                 # 'end' / 'force' : the current frame is fed, then decide
            seg_sum += snn.step(feat[t]); r['fed'] = 1; r['seg_end'] = 1; keep = ev[1] >= prm['L_min']; r['keep'] = int(keep)
            if keep:
                o = np.sort(seg_sum)[::-1]; c = int(seg_sum.argmax()); m = int(o[0] - o[1])
                r['det_any'] = 1; r['det_class'] = c; r['det_valid'] = int(c < N_KEYWORD and m >= margin); r['seg_sum'] = [int(v) for v in seg_sum]
                dets.append(dict(t=t, cls=c, valid=r['det_valid'], margin=m, seg_sum=[int(v) for v in seg_sum]))
        elif vad.active:
            seg_sum += snn.step(feat[t]); r['fed'] = 1
        rec.append(r)
    return dict(T=T, rec=rec, dets=dets, feat=feat, E=E, snn_frames=sum(r['fed'] for r in rec))

def dump(out_dir, name, pcm24, res, prm, energy, events):
    d = os.path.join(out_dir, name); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'pcm24.hex'), 'w') as f:
        f.write(f"// {name} {len(pcm24)} x 24-bit samples\n// width=24 signed=1\n")
        for v in pcm24: f.write(f"{int(v) & 0xffffff:06x}\n")
    with open(os.path.join(d, 'expected.hex'), 'w') as f:
        f.write("// per frame: bit0 active, bit1 seg_start, bit2 seg_end, bit3 det_any, bit4 det_valid, bits[8:5] det_class, bits[15:9] fed count\n// width=16\n")
        for r in res['rec']:
            w = r['active'] | (r['seg_start'] << 1) | (r['seg_end'] << 2) | (r['det_any'] << 3) | (r['det_valid'] << 4) | (r['det_class'] << 5) | (min(r['fed'], 127) << 9)
            f.write(f"{w:04x}\n")
    with open(os.path.join(d, 'energy.hex'), 'w') as f:
        f.write("// per frame E (16 bit)\n// width=16\n"); [f.write(f"{int(r['E']):04x}\n") for r in res['rec']]
    with open(os.path.join(d, 'seg_sums.hex'), 'w') as f:
        f.write("// per decision: 12 x 32-bit segment sums (two's complement), in decision order\n// width=32\n")
        for dt in res['dets']:
            for v in dt['seg_sum']: f.write(f"{int(v) & 0xffffffff:08x}\n")
    json.dump(dict(name=name, n_samples=len(pcm24), T=res['T'], params=prm, energy=energy, events=events, dets=res['dets'],
                   snn_frames=res['snn_frames'], duty=res['snn_frames'] / res['T']), open(os.path.join(d, 'info.json'), 'w'), indent=1)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=os.path.join(ROOT, 'snn', 'runs', 'M3k3e60_c12_q8_kd'))
    ap.add_argument('--dump', type=int, default=4); ap.add_argument('--dur', type=float, default=8.0); ap.add_argument('--cond', default='board')
    for k, v in (('T_on', 1000), ('T_off', 600), ('N_on', 2), ('H', 30), ('P', 20), ('L_min', 25), ('L_max', 98), ('margin', 0)): ap.add_argument('--' + k, type=int, default=v)
    ap.add_argument('--energy', default='E1'); ap.add_argument('--out', default=os.path.join(ROOT, 'snn', 'golden', 'vectors', 'pipeline'))
    ap.add_argument('--margin_sel', type=int, default=0)
    a = ap.parse_args()
    import torch; torch.backends.cuda.matmul.allow_tf32 = False
    from snn.golden.export_snn_mem import load_run, extract
    from snn.eval import vad_eval
    model, cfg, res = load_run(a.run); P = extract(model)
    vad_eval._init(a.run)
    prm = dict(T_on=a.T_on, T_off=a.T_off, N_on=a.N_on, H=a.H, P=a.P, L_min=a.L_min, L_max=a.L_max)
    for i in range(a.dump):
        # short board-condition stream from the test split (seed offset 5000+i), keep the raw waveform
        rng = np.random.default_rng(5000 + i + {'board': 0, 'talk': 100, 'quiet': 200, 'w2': 300}.get(a.cond, 0)); c = vad_eval.COND[a.cond]
        n = int(a.dur * vad_eval.SR); noise = vad_eval._G['noise']['test']
        if len(noise) < 2 * n: noise = np.tile(noise, int(np.ceil(2 * n / len(noise))))
        vol = rng.uniform(*c['vol']); st = int(rng.integers(0, len(noise) - n))
        wav = np.round(noise[st: st + n].astype(np.float64) * vol * (1 << 23)).astype(np.int64)
        files, labels = zip(*vad_eval._G['files']['test']); events = []
        kw_idx = [k for k, y in enumerate(labels) if y < 10]; unk_idx = [k for k, y in enumerate(labels) if y >= 10]
        gap = c['gap'] if a.cond != 'board' else (0.3, 1.5); pos = int(rng.uniform(*gap) * vad_eval.SR)
        while pos + vad_eval.SR <= n:
            if c['unk'] is None: j = int(rng.integers(len(files)))
            else: j = int(rng.choice(unk_idx)) if rng.random() < c['unk'] else int(rng.choice(kw_idx))
            wav[pos: pos + vad_eval.SR] += vad_eval.apply_gain_pcm24(vad_eval.wav_float_to_pcm24(vad_eval.load_item(files[j])), rng.uniform(*c['gain']))
            events.append(dict(onset=pos // 160, offset=(pos + vad_eval.SR) // 160, label=int(labels[j]), file=files[j]))
            pos += vad_eval.SR + int(rng.uniform(*gap) * vad_eval.SR)
        wav = np.clip(wav, -(1 << 23), (1 << 23) - 1)
        r = run_pipeline(P, wav, prm, a.energy, a.margin)
        dump(a.out, f'stream_{i}', wav, r, prm, a.energy, events)
        if a.margin_sel: 
            info = json.load(open(os.path.join(a.out, f'stream_{i}', 'info.json'))); info['margin_sel'] = a.margin_sel; json.dump(info, open(os.path.join(a.out, f'stream_{i}', 'info.json'), 'w'), indent=1)
        print(f"stream_{i}: T={r['T']} events={[(e['label'], e['onset']) for e in events]} dets={[(d['t'], d['cls'], d['valid']) for d in r['dets']]} snn_frames={r['snn_frames']} duty={r['snn_frames']/r['T']:.2f}")
