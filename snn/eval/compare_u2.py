"""compare_u2.py — table: DS-CNN 24ch and SNN M3k3 Q8+KD trained the base way vs the U2 way (unknown 3x + class
weights); 12-class clip accuracy, unknown/silence/keyword-mean accuracy, and the SNN-DSCNN gap within each training
style. Writes snn/runs/compare_u2.md (markdown) and compare_u2.json."""
import os, json, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); S = os.path.join(ROOT, 'snn')
def R(n):
    p = os.path.join(S, 'runs', n, 'results.json'); return json.load(open(p)) if os.path.exists(p) else None
def acc(r): return r['test_acc']['mean'] if isinstance(r['test_acc'], dict) else r['test_acc']
groups = {('DS-CNN 24ch', 'base'): ['dscnn_c12_e100', 'dscnn_c12_s1', 'dscnn_c12_s2'], ('DS-CNN 24ch', 'U2'): ['dscnn_c12_U2_s0', 'dscnn_c12_U2_s1', 'dscnn_c12_U2_s2'],
          ('SNN M3k3 Q8+KD', 'base'): ['M3k3e60_c12_q8_kd', 'M3k3e60_c12_q8_kd_s1', 'M3k3e60_c12_q8_kd_s2'], ('SNN M3k3 Q8+KD', 'U2'): ['M3k3e60_U2_q8_kd_s0', 'M3k3e60_U2_q8_kd_s1']}
rows = {}
for (model, style), runs in groups.items():
    rs = [r for r in map(R, runs) if r]
    if not rs: rows[(model, style)] = None; continue
    a = np.array([acc(r) for r in rs]); unk = np.array([r['per_class_acc']['unknown'] for r in rs]); sil = np.array([r['per_class_acc']['silence'] for r in rs])
    kw = np.array([np.mean([v for k, v in r['per_class_acc'].items() if k not in ('unknown', 'silence')]) for r in rs])
    rows[(model, style)] = dict(n=len(rs), acc=a.mean(), acc_std=a.std(), unk=unk.mean(), sil=sil.mean(), kw=kw.mean(), runs=[r for r in runs if R(r)])
L = ["| 모델 | 학습 방식 | 시드 수 | 12cls 클립 정확도 | 키워드 평균 | unknown | silence |", "|---|---|---|---|---|---|---|"]
for (model, style), r in rows.items():
    L.append(f"| {model} | {style} | {r['n']} | {100*r['acc']:.2f} ± {100*r['acc_std']:.2f} | {100*r['kw']:.1f} | {100*r['unk']:.1f} | {100*r['sil']:.1f} |" if r else f"| {model} | {style} | pending | | | | |")
gaps = {}
for style in ('base', 'U2'):
    d, s = rows.get(('DS-CNN 24ch', style)), rows.get(('SNN M3k3 Q8+KD', style))
    if d and s: gaps[style] = dict(gap=100 * (s['acc'] - d['acc']), dscnn=100 * d['acc'], snn=100 * s['acc'], dscnn_drop=None, snn_drop=None)
if rows.get(('DS-CNN 24ch', 'U2')) and rows.get(('DS-CNN 24ch', 'base')): gaps['U2']['dscnn_drop'] = 100 * (rows[('DS-CNN 24ch', 'U2')]['acc'] - rows[('DS-CNN 24ch', 'base')]['acc'])
if rows.get(('SNN M3k3 Q8+KD', 'U2')) and rows.get(('SNN M3k3 Q8+KD', 'base')): gaps['U2']['snn_drop'] = 100 * (rows[('SNN M3k3 Q8+KD', 'U2')]['acc'] - rows[('SNN M3k3 Q8+KD', 'base')]['acc'])
L.append("")
for style, g in gaps.items():
    L.append(f"- **{style} 방식 SNN−DS-CNN 격차: {g['gap']:+.2f} %p** (DS-CNN {g['dscnn']:.2f} / SNN {g['snn']:.2f})" + (f"; U2 방식으로 바꿀 때 하락: DS-CNN {g['dscnn_drop']:+.2f} %p, SNN {g['snn_drop']:+.2f} %p" if style == 'U2' and g.get('dscnn_drop') is not None else ""))
md = "\n".join(L); open(os.path.join(S, 'runs', 'compare_u2.md'), 'w').write(md); json.dump(dict(rows={f'{m}|{s}': r for (m, s), r in rows.items()}, gaps=gaps), open(os.path.join(S, 'runs', 'compare_u2.json'), 'w'), indent=1)
print(md)
