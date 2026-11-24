"""
ops_energy.py — B-5: operation counts and (estimated) energy per inference / per second of streaming for
DS-CNN (24/48 ch, hls4ml MFCC version) vs SNN candidates. Writes snn/runs/figures/ops_energy.csv and
ops_vs_acc.png (x = ops per second of streaming, log; y = 12-class test accuracy).

Energy constants (45 nm, Horowitz, "Computing's energy problem (and what we can do about it)", ISSCC 2014,
Fig. 1 table): int add 8 b = 0.03 pJ, int add 32 b = 0.1 pJ, int mult 8 b = 0.2 pJ.
  MAC (8b x 8b mult + 32b accumulate) := 0.2 + 0.1 = 0.3 pJ ; AC (8b weight into 24b acc) := 0.1 pJ (32b add, conservative)
These are ESTIMATES (추정) of datapath energy only — no memory access, no control, no clock tree.
"""
import os, sys, json, csv
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
E_MAC, E_AC = 0.3e-12, 0.1e-12
T = 98; FPS = 100                      # frames per second (hop 10 ms)

def dscnn_macs(h, w, ch):
    """Conv2d(1->ch, 10x4, stride 5x4) -> 2 x Conv(ch->ch 3x3) -> GAP -> FC(ch->12)."""
    h1, w1 = (h + 2*4 - 10)//5 + 1, (w + 2*1 - 4)//4 + 1
    c1 = h1*w1*ch*10*4
    c23 = 2 * h1*w1*ch*(3*3*ch)
    return c1 + c23 + ch*12, (h1, w1)

def dscnn_old_macs():
    """week-0 hls4ml model: input 49x10, Conv(10x4, stride 5x2) -> 10x5x24, 2 x Conv3x3(24), FC 10."""
    h1, w1, ch = 10, 5, 24
    return h1*w1*ch*10*4 + 2*h1*w1*ch*9*ch + ch*10

def R(n): return json.load(open(os.path.join(ROOT, 'snn', 'runs', n, 'results.json')))

def main():
    zs = json.load(open(os.path.join(ROOT, 'snn', 'runs', 'zero_skip_stats.json')))[0]   # clean test set
    rows = []
    m24, _ = dscnn_macs(98, 40, 24); m48, _ = dscnn_macs(98, 40, 48); mold = dscnn_old_macs()
    for name, macs, acc, note in [('DS-CNN 24ch (log-mel 98x40)', m24, R('dscnn_c12_e100')['test_acc'], '12cls'),
                                  ('DS-CNN 48ch (log-mel 98x40)', m48, R('dscnn_c12_ch48')['test_acc'], '12cls'),
                                  ('DS-CNN 24ch hls4ml (MFCC 49x10, week 0)', mold, 0.9175, '10cls, on board')]:
        rows.append(dict(model=name, family='DS-CNN', acc=acc, mac_per_inf=macs, ac_per_inf=0, mac_per_s_stream=macs*FPS, ac_per_s_stream=0,
                         ops_per_s_stream=macs*FPS, energy_per_s_pJ=macs*FPS*E_MAC*1e12, note=note + ', streaming = full window recomputed every 10 ms'))
    snn = [('M1 LIF 128-128 Q8', 'M1_c12_q8'), ('M3 TCN K=2 Q8+KD', 'M3_c12_q8_kd'),
           ('M3k3 Q4 (30ep)', 'M3k3_c12_q4'), ('M3k3 Q6 (60ep)', 'M3k3e60_c12_q6'), ('M3k3 Q8 (60ep)', 'M3k3e60_c12_q8'), ('M3k3 Q8+KD (60ep, recommended)', 'M3k3e60_c12_q8_kd')]
    for name, run in snn:
        r = R(run); h = r['hw']
        mac_dense = h['mac_per_sample']; ac = h['ac_per_sample']
        mac_zs = mac_dense * (1 - zs['mac_saving']) if run.startswith('M3k3') else mac_dense * (1 - zs['feat_zero_ratio'])
        rows.append(dict(model=name + ' (dense input)', family='SNN', acc=r['test_acc']['mean'], mac_per_inf=mac_dense, ac_per_inf=ac,
                         mac_per_s_stream=mac_dense/T*FPS, ac_per_s_stream=ac/T*FPS, ops_per_s_stream=(mac_dense+ac)/T*FPS,
                         energy_per_s_pJ=(mac_dense/T*FPS*E_MAC + ac/T*FPS*E_AC)*1e12, note='no zero-skip; AC = measured spikes x fan-out'))
        rows.append(dict(model=name + ' (zero-skip)', family='SNN-zs', acc=r['test_acc']['mean'], mac_per_inf=mac_zs, ac_per_inf=ac,
                         mac_per_s_stream=mac_zs/T*FPS, ac_per_s_stream=ac/T*FPS, ops_per_s_stream=(mac_zs+ac)/T*FPS,
                         energy_per_s_pJ=(mac_zs/T*FPS*E_MAC + ac/T*FPS*E_AC)*1e12, note=f"input-layer zero-skip ({100*zs['mac_saving']:.0f} % of MACs skipped, measured)"))
    out = os.path.join(ROOT, 'snn', 'runs', 'figures')
    with open(os.path.join(out, 'ops_energy.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"{'model':45s} {'acc':>6s} {'MAC/inf':>10s} {'AC/inf':>9s} {'ops/s stream':>13s} {'pJ/s (est)':>11s}")
    for r in rows:
        print(f"{r['model']:45s} {100*r['acc']:6.2f} {r['mac_per_inf']:10.0f} {r['ac_per_inf']:9.0f} {r['ops_per_s_stream']:13.0f} {r['energy_per_s_pJ']:11.0f}")
    # ---- figure (dataviz palette: slot1 blue = DS-CNN, slot2 orange = SNN dense, slot3 aqua = SNN zero-skip)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    col = {'DS-CNN': '#2a78d6', 'SNN': '#eb6834', 'SNN-zs': '#1baf7a'}; lab = {'DS-CNN': 'DS-CNN (window recomputed every 10 ms)', 'SNN': 'SNN, dense input layer', 'SNN-zs': 'SNN, input zero-skip'}
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    fig.patch.set_facecolor('#fcfcfb'); ax.set_facecolor('#fcfcfb')
    for fam in ('DS-CNN', 'SNN', 'SNN-zs'):
        rr = [r for r in rows if r['family'] == fam]
        ax.scatter([r['ops_per_s_stream'] for r in rr], [100*r['acc'] for r in rr], s=64, color=col[fam], label=lab[fam], edgecolor='#fcfcfb', linewidth=1.5, zorder=3)
        for r in rr:
            if fam == 'SNN' or 'M3k3 Q8 (60ep)' in r['model']: continue      # dense twins / overlapping point: not labelled (legend + table carry them)
            short = r['model'].replace(' (dense input)', '').replace(' (zero-skip)', '').replace(' (log-mel 98x40)', '').replace(' hls4ml (MFCC 49x10, week 0)', ' hls4ml')
            ax.annotate(short, (r['ops_per_s_stream'], 100*r['acc']), textcoords='offset points', xytext=(7, -3 if 'Q6' in r['model'] else 5), fontsize=7.5, color='#52514e')
    ax.set_xscale('log'); ax.set_xlabel('operations per second, streaming (MAC + AC, log scale)', color='#0b0b0b')
    ax.set_ylabel('test accuracy (%)', color='#0b0b0b'); ax.set_title('Accuracy vs streaming compute (12-class GSC; hls4ml point = 10-class)', fontsize=10, color='#0b0b0b')
    ax.grid(True, which='major', color='#e6e5e0', linewidth=0.8, zorder=0); ax.grid(False, which='minor')
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color('#c3c2b7')
    ax.tick_params(colors='#52514e'); ax.legend(frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'ops_vs_acc.png'))
    print('saved', os.path.join(out, 'ops_energy.csv'), os.path.join(out, 'ops_vs_acc.png'))

if __name__ == '__main__':
    main()
