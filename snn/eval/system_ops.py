"""
system_ops.py — operations / estimated energy / latency in the SAME VAD-segment system for
  DS-CNN 24ch U2-style + VAD : one inference on the last 98 frames at segment end
      (a) dense, (b) input zero-skip (conv1 skips zero input features), (c) input + ReLU-output zero-skip
      (conv2/conv3 skip zero activations; measured sparsity on the actual segment windows)
  SNN U2 + VAD : every fed frame (pre-roll + live), measured input MACs (dense / zero-skip) and ACs (spikes x fan-out)
Per segment and per hour on the test streams quiet / board / talk (VAD segments from the same parameters).
Energy: Horowitz 2014, 45 nm, datapath only — MAC(8b mult + 32b add) 0.3 pJ, AC(32b add) 0.1 pJ. ESTIMATES.
Latency: serial engine, 1 MAC/cycle @ 48 MHz. Writes snn/runs/system_ops.json and system_ops.md.
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
from multiprocessing import Pool
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, ROOT)
from snn.eval import vad_eval as V
from snn.golden.snn_int_ref import run_int
E_MAC, E_AC = 0.3e-12, 0.1e-12; T = 98; FCLK = 48e6
_G = {}

def dscnn_macs_window(net, x):
    """x [98,40] uint8 -> dict(dense, in_skip, in_act_skip) MAC counts for one inference (counts multiplies with a zero operand as skipped)."""
    xt = ((torch.from_numpy(x).float() - net.mean) / net.std)[None, None]            # normalised input (nonzero even where feature == 0!)
    nz_in = (torch.from_numpy(x) != 0).float()[None, None]
    ch = net.c1.out_channels
    # conv1: patches of the raw feature map -> count nonzero *features* per patch
    p1 = F.unfold(nz_in, kernel_size=(10, 4), stride=(5, 4), padding=(4, 1))        # [1, 40, L]
    c1_dense = p1.shape[1] * p1.shape[2] * ch; c1_skip = int(p1.sum().item()) * ch
    with torch.no_grad():
        a1 = F.relu6(net.b1(net.c1(xt))); a2 = F.relu6(net.b2(net.c2(a1))); a3 = F.relu6(net.b3(net.c3(a2)))
    p2 = F.unfold((a1 != 0).float(), 3, padding=1); c2_dense = p2.shape[1] * p2.shape[2] * ch; c2_skip = int(p2.sum().item()) * ch
    p3 = F.unfold((a2 != 0).float(), 3, padding=1); c3_dense = p3.shape[1] * p3.shape[2] * ch; c3_skip = int(p3.sum().item()) * ch
    fc = ch * net.fc.out_features
    return dict(dense=c1_dense + c2_dense + c3_dense + fc, in_skip=c1_skip + c2_dense + c3_dense + fc, in_act_skip=c1_skip + c2_skip + c3_skip + fc,
                in_sparsity=1 - c1_skip / c1_dense, a1_sparsity=1 - c2_skip / c2_dense, a2_sparsity=1 - c3_skip / c3_dense)

def snn_ops_segment(P, x):
    """x [L,40] -> dict(mac_dense, mac_skip, ac) over the fed frames (3 taps per frame look at frames t, t-1, t-2)."""
    L = len(x); hid = [l for l in P['layers'] if l['type'] != 'out']; out = [l for l in P['layers'] if l['type'] == 'out'][0]
    g = run_int(P, x[None], trace=True)
    mac_dense = L * 3 * 40 * 64
    nz = (x != 0).sum(1).astype(np.int64); mac_skip = 0
    for t in range(L):
        for j in range(3):
            if t - j >= 0: mac_skip += int(nz[t - j]) * 64
    fan = [hid[1]['W'].shape[1] * 64 // hid[1]['W'].shape[1] * 0 + 64 * 3, 64 * 3, 64 * 3, out['W'].shape[0]]   # each spike feeds K=3 taps of the next layer (64 neurons) or the output (12)
    ac = 0
    for i, l in enumerate(hid):
        s = g['traces'][l['name']]['s'][0].sum(); ac += int(s) * fan[i]
    return dict(mac_dense=mac_dense, mac_skip=mac_skip, ac=ac, frames=L)

def _init(run_snn, run_ds):
    torch.backends.cuda.matmul.allow_tf32 = False; torch.set_num_threads(1)
    from snn.golden.export_snn_mem import load_run, extract
    from snn.baseline.train_dscnn import DSCNN
    m, cfg, res = load_run(run_snn); _G['P'] = extract(m)
    c = json.load(open(os.path.join(run_ds, 'config.json'))); d = DSCNN(12, c.get('ch', 24), c['mean'], c['std']); d.load_state_dict(torch.load(os.path.join(run_ds, 'best.pt'), map_location='cpu')); d.eval(); _G['net'] = d

def _stream(args):
    s, prm, energy = args
    segs = [(a, e) for a, e, kept in V.vad_segments(s[energy], prm) if kept]
    out = dict(hours=len(s['E1']) * 160 / 16000 / 3600, n_seg=len(segs), ds=dict(dense=0, in_skip=0, in_act_skip=0), snn=dict(mac_dense=0, mac_skip=0, ac=0, frames=0), sp=[])
    for a, e in segs:
        w = s['feat'][max(e - T, 0):e]; xx = np.zeros((T, 40), np.uint8); xx[T - len(w):] = w
        d = dscnn_macs_window(_G['net'], xx); out['sp'].append((d['in_sparsity'], d['a1_sparsity'], d['a2_sparsity']))
        for k in ('dense', 'in_skip', 'in_act_skip'): out['ds'][k] += d[k]
        so = snn_ops_segment(_G['P'], s['feat'][a:e])
        for k in ('mac_dense', 'mac_skip', 'ac', 'frames'): out['snn'][k] += so[k]
    return out

def main():
    rep = json.load(open(os.path.join(ROOT, 'snn', 'runs', 'vad_eval', 'report.json'))); bv = rep['best_val']
    prm = {k: bv[k] for k in ('T_on', 'T_off', 'N_on', 'H', 'P', 'L_min', 'L_max')}; energy = bv['energy']
    run_snn = os.path.join(ROOT, 'snn', 'runs', 'M3k3e60_U2_q8_kd_s0'); run_ds = os.path.join(ROOT, 'snn', 'runs', 'dscnn_c12_U2_s0')
    res = {}
    with Pool(16, initializer=_init, initargs=(run_snn, run_ds)) as pool:
        for cond in ('quiet', 'board', 'talk'):
            streams = list(np.load(os.path.join(ROOT, 'snn', 'runs', 'vad_eval', f'streams_test_{cond}.npz'), allow_pickle=True)['streams'])
            outs = pool.map(_stream, [(s, prm, energy) for s in streams])
            H = sum(o['hours'] for o in outs); nseg = sum(o['n_seg'] for o in outs); sp = np.array([p for o in outs for p in o['sp']])
            ds = {k: sum(o['ds'][k] for o in outs) for k in ('dense', 'in_skip', 'in_act_skip')}; sn = {k: sum(o['snn'][k] for o in outs) for k in ('mac_dense', 'mac_skip', 'ac', 'frames')}
            res[cond] = dict(hours=H, segments=nseg, seg_per_h=nseg / H, frames_per_seg=sn['frames'] / max(nseg, 1),
                sparsity=dict(input=float(sp[:, 0].mean()), relu1=float(sp[:, 1].mean()), relu2=float(sp[:, 2].mean())) if len(sp) else None,
                per_segment=dict(ds_dense=ds['dense'] / max(nseg, 1), ds_in_skip=ds['in_skip'] / max(nseg, 1), ds_in_act_skip=ds['in_act_skip'] / max(nseg, 1),
                                 snn_mac_dense=sn['mac_dense'] / max(nseg, 1), snn_mac_skip=sn['mac_skip'] / max(nseg, 1), snn_ac=sn['ac'] / max(nseg, 1)),
                per_hour=dict(ds_dense=ds['dense'] / H, ds_in_skip=ds['in_skip'] / H, ds_in_act_skip=ds['in_act_skip'] / H,
                              snn_mac_dense=sn['mac_dense'] / H, snn_mac_skip=sn['mac_skip'] / H, snn_ac=sn['ac'] / H))
            print(cond, res[cond]['segments'], 'segments', f"{res[cond]['seg_per_h']:.0f}/h", res[cond]['per_segment'], flush=True)
    # energy (pJ) per segment / per hour
    for cond, r in res.items():
        for scope in ('per_segment', 'per_hour'):
            p = r[scope]; r[scope + '_energy_pJ'] = dict(ds_dense=p['ds_dense'] * E_MAC * 1e12, ds_in_skip=p['ds_in_skip'] * E_MAC * 1e12, ds_in_act_skip=p['ds_in_act_skip'] * E_MAC * 1e12,
                snn_dense=(p['snn_mac_dense'] * E_MAC + p['snn_ac'] * E_AC) * 1e12, snn_skip=(p['snn_mac_skip'] * E_MAC + p['snn_ac'] * E_AC) * 1e12)
    # latency / buffer / parallelism (serial 1 MAC @ 48 MHz)
    b = res['board']['per_segment']
    lat = dict(ds_dense_ms=b['ds_dense'] / FCLK * 1e3, ds_in_skip_ms=b['ds_in_skip'] / FCLK * 1e3, ds_in_act_skip_ms=b['ds_in_act_skip'] / FCLK * 1e3,
               snn_last_frame_ms=16748 / FCLK * 1e3, snn_last_frame_avg_ms=5551 / FCLK * 1e3,
               ds_buffer_bits=T * 40 * 8, ds_buffer_bram18k=int(np.ceil(T * 40 * 8 / 18432)), snn_preroll_bits=21 * 40 * 8, snn_state_bits=4 * 64 * 16 + 12 * 24 + 64 * (5 + 9 + 17 + 1) + 3 * 40 * 8,
               ds_parallel_for_1ms_dense=b['ds_dense'] / (FCLK * 1e-3), ds_parallel_for_1ms_in_act_skip=b['ds_in_act_skip'] / (FCLK * 1e-3),
               ds_parallel_for_snn_latency_dense=b['ds_dense'] / 16748, ds_parallel_for_snn_latency_in_act_skip=b['ds_in_act_skip'] / 16748)
    res['latency'] = lat
    json.dump(res, open(os.path.join(ROOT, 'snn', 'runs', 'system_ops.json'), 'w'), indent=1)
    L = ["| 스트림 | 구간/h | SNN 공급 프레임/구간 | DS-CNN dense | DS-CNN 입력 0건너뛰기 | DS-CNN 입력+ReLU 0건너뛰기 | SNN MAC dense + AC | SNN MAC 0건너뛰기 + AC |", "|---|---|---|---|---|---|---|---|"]
    for cond, r in res.items():
        if cond == 'latency': continue
        p = r['per_segment']; L.append(f"| {cond} | {r['seg_per_h']:.0f} | {r['frames_per_seg']:.0f} | {p['ds_dense']:,.0f} | {p['ds_in_skip']:,.0f} | {p['ds_in_act_skip']:,.0f} | {p['snn_mac_dense']:,.0f} + {p['snn_ac']:,.0f} | {p['snn_mac_skip']:,.0f} + {p['snn_ac']:,.0f} |")
    L.append("\n| 스트림 | DS-CNN dense ops/h | DS-CNN 입력+ReLU 0건너뛰기 ops/h | SNN dense ops/h | SNN 0건너뛰기 ops/h | 에너지(추정, µJ/h): DS dense / DS skip / SNN dense / SNN skip |\n|---|---|---|---|---|---|")
    for cond, r in res.items():
        if cond == 'latency': continue
        p = r['per_hour']; e = r['per_hour_energy_pJ']
        L.append(f"| {cond} | {p['ds_dense']:,.0f} | {p['ds_in_act_skip']:,.0f} | {p['snn_mac_dense'] + p['snn_ac']:,.0f} | {p['snn_mac_skip'] + p['snn_ac']:,.0f} | {e['ds_dense']/1e6:.2f} / {e['ds_in_act_skip']/1e6:.2f} / {e['snn_dense']/1e6:.2f} / {e['snn_skip']/1e6:.2f} |")
    open(os.path.join(ROOT, 'snn', 'runs', 'system_ops.md'), 'w').write("\n".join(L)); print("\n".join(L)); print(json.dumps(lat, indent=1))

if __name__ == '__main__': main()
