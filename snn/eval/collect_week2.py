"""collect_week2.py — B-1/B-2/B-3 result tables (markdown) from snn/runs/*/results.json -> snn/runs/week2_tables.md.
Missing runs are reported as 'pending' so the script can be re-run while training continues."""
import os, json, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
def R(n):
    p = os.path.join(ROOT, 'snn', 'runs', n, 'results.json')
    return json.load(open(p)) if os.path.exists(p) else None
def acc(r): return None if r is None else (r['test_acc']['mean'] if isinstance(r['test_acc'], dict) else r['test_acc'])
def val(r): return None if r is None else r['best_val']
def f(x): return 'pending' if x is None else f"{100*x:.2f}"
def ms(vals):
    v = [x for x in vals if x is not None]
    if not v: return 'pending'
    s = f"{100*np.mean(v):.2f} ± {100*np.std(v):.2f} (n={len(v)})"
    return s + ('' if len(v) == len(vals) else f', {len(vals)-len(v)} pending')
out = []
out.append("### B-1. 10클래스 (test 4,074개; 기준 DS-CNN 24ch 91.43 % − 2 %p = 89.43 %)\n")
out.append("| 런 | 설명 | val | test |\n|---|---|---|---|")
for n, d in [('dscnn_c10_e100', 'DS-CNN 24ch (seed 0)'), ('dscnn_c10_ch48', 'DS-CNN 48ch teacher'),
             ('M3k3_c10_float_e60_s0', 'M3k3 float 60ep s0'), ('M3k3e60_c10_q8_s0', 'M3k3 Q8 (60ep) s0'), ('M3k3e60_c10_q8_kd_s0', 'M3k3 Q8+KD (60ep) s0'),
             ('M3k3e60_c10_q8_kd_s1', 'M3k3 Q8+KD (60ep) s1'), ('M3k3e60_c10_q8_kd_s2', 'M3k3 Q8+KD (60ep) s2')]:
    r = R(n); out.append(f"| {n} | {d} | {f(val(r))} | {f(acc(r))} |")
out.append("\n### B-2. 시드 반복 (seed 0/1/2, 평균 ± 표준편차, test)\n")
out.append("| 설정 | seed 0 | seed 1 | seed 2 | 평균 ± std |\n|---|---|---|---|---|")
for d, names in [('DS-CNN 24ch, 12cls', ['dscnn_c12_e100', 'dscnn_c12_s1', 'dscnn_c12_s2']),
                 ('DS-CNN 24ch, 10cls', ['dscnn_c10_e100', 'dscnn_c10_s1', 'dscnn_c10_s2']),
                 ('M3k3 float 60ep, 12cls', ['M3k3_c12_float_e60', 'M3k3_c12_float_e60_s1', 'M3k3_c12_float_e60_s2']),
                 ('M3k3 Q8+KD 60ep, 12cls', ['M3k3e60_c12_q8_kd', 'M3k3e60_c12_q8_kd_s1', 'M3k3e60_c12_q8_kd_s2']),
                 ('M3k3 float 60ep, 10cls', ['M3k3_c10_float_e60_s0', 'M3k3_c10_float_e60_s1', 'M3k3_c10_float_e60_s2']),
                 ('M3k3 Q8+KD 60ep, 10cls', ['M3k3e60_c10_q8_kd_s0', 'M3k3e60_c10_q8_kd_s1', 'M3k3e60_c10_q8_kd_s2'])]:
    a = [acc(R(n)) for n in names]; out.append(f"| {d} | {f(a[0])} | {f(a[1])} | {f(a[2])} | {ms(a)} |")
out.append("\n### B-3. 장기 학습 (12cls, val 기준 best checkpoint)\n")
out.append("| float 에폭 | float val | float test | Q8+KD val | Q8+KD test |\n|---|---|---|---|---|")
for e, fn, qn in [(30, 'M3k3_c12_float', 'M3k3_c12_q8_kd'), (60, 'M3k3_c12_float_e60', 'M3k3e60_c12_q8_kd'), (100, 'M3k3_c12_float_e100', 'M3k3e100_c12_q8_kd'), (150, 'M3k3_c12_float_e150', 'M3k3e150_c12_q8_kd')]:
    fr, qr = R(fn), R(qn); out.append(f"| {e} | {f(val(fr))} | {f(acc(fr))} | {f(val(qr))} | {f(acc(qr))} |")
s = "\n".join(out); open(os.path.join(ROOT, 'snn', 'runs', 'week2_tables.md'), 'w').write(s); print(s)
