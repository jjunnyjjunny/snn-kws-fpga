"""collect_results.py — aggregate snn/runs/*/results.json into markdown tables (stdout + snn/runs/summary.md)."""
import os, sys, json, glob
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RUNS = os.path.join(ROOT, 'snn', 'runs')

def load():
    rows = {}
    for p in sorted(glob.glob(os.path.join(RUNS, '*', 'results.json'))):
        name = os.path.basename(os.path.dirname(p))
        r = json.load(open(p))
        r['_cfg'] = json.load(open(os.path.join(os.path.dirname(p), 'config.json')))
        g = os.path.join(os.path.dirname(p), 'golden_check.json')
        r['_golden'] = json.load(open(g)) if os.path.exists(g) else None
        rows[name] = r
    return rows

def pct(x): return f"{100*x:.2f}"

def main():
    rows = load()
    out = []
    out.append("## DS-CNN baselines (PyTorch, integer log-mel features)\n")
    out.append("| run | data | ch | params | val | test | test -20dB | -10dB | +6dB | +10dB | unknown acc |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for n, r in rows.items():
        if not n.startswith('dscnn'): continue
        c = r['_cfg']
        out.append(f"| {n} | {os.path.basename(c['data'])} | {c.get('ch',24)} | {r['n_params']} | {pct(r['best_val'])} | **{pct(r['test_acc'])}** | "
                   f"{pct(r.get('test_acc_g-20',0))} | {pct(r.get('test_acc_g-10',0))} | {pct(r.get('test_acc_g+6',0))} | {pct(r.get('test_acc_g+10',0))} | "
                   f"{pct(r['per_class_acc'].get('unknown', float('nan'))) if 'unknown' in r['per_class_acc'] else '-'} |")
    out.append("\n## SNN runs\n")
    out.append("| run | model | data | mode/bits | hidden | k | KD | val | test(mean) | test(last) | test(max) | -20dB | -10dB | +10dB | rates/layer | params | w-bits | BRAM18K | MAC/frame | AC/frame | cyc/frame(serial) | vmax | v bits | golden match |\n|" + "---|" * 22)
    for n, r in rows.items():
        if n.startswith('dscnn'): continue
        c = r['_cfg']; h = r['hw']; T = 98
        g = r['_golden']
        gm = '-' if g is None else f"{100*g['pred_agreement']:.1f}%"
        out.append(f"| {n} | {r['model']} | {os.path.basename(c['data']).replace('feat_','').replace('.npz','')} | {r['mode']}{'/'+str(r['bits']) if r['mode']=='int' else ''} | {r['hidden']} | {r['k']} | {'Y' if r.get('kd') else '-'} | "
                   f"{pct(r['best_val']) if r['best_val']>=0 else '-'} | **{pct(r['test_acc']['mean'])}** | {pct(r['test_acc']['last'])} | {pct(r['test_acc']['max'])} | "
                   f"{pct(r.get('test_acc_g-20',0))} | {pct(r.get('test_acc_g-10',0))} | {pct(r.get('test_acc_g+10',0))} | "
                   f"{'/'.join(f'{x:.3f}' for x in r['spike_rate_per_layer'])} | {h['n_params']} | {h['weight_bits']} | {h['bram18k']} | "
                   f"{h['mac_per_sample']/T:.0f} | {h['ac_per_sample']/T:.0f} | {h['cycles_per_frame_serial']:.0f} | "
                   f"{'/'.join(f'{x:.0f}' for x in r['vmax_per_layer'])} | {'/'.join(str(x) for x in r['v_bits_needed'])} | {gm} |")
    s = "\n".join(out)
    open(os.path.join(RUNS, 'summary.md'), 'w').write(s)
    print(s)

if __name__ == '__main__':
    main()
