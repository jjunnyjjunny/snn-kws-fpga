"""
playback_compare.py — per-slot analysis of board playback runs (BOARD_STEP2 §1B) against the digital golden:
  * per-run scoring (same alignment as playback_score.py) with the outcome of every slot,
  * repeatability between two runs at the same operating point (slot-by-slot agreement, list of differing slots),
  * false alarms in unknown/silence slots: are they leaked detections of the previous keyword slot (late segment end)?
  * relation between the clip level (wav peak dBFS, + chain gain) and wrong/missed slots,
  * board confusion matrix (runs merged) next to the golden -> fig13.
usage: python snn/eval/playback_compare.py --runs R0=snn/board/log/R0_det.csv:0 R0b=snn/board/log/R0b_det.csv:0 R2=snn/board/log/R2_det.csv:6212 \
           --golden snn/board/play/playback_golden_det.csv --gain_db 1 --pair R0,R0b --out snn/board/log/playback_compare --fig snn/docs/figures/final/fig13_board_confusion.png
"""
import os, sys, csv, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..')); sys.path.insert(0, ROOT)
from snn.eval.playback_score import read_det, find_offset, KEYWORDS, CLASSES

def score_slots(dets, slots, lead, cadence, t0=None):
    o, hits, second = (t0 * 0.01, None, None) if t0 is not None else find_offset(dets, slots, lead, cadence)
    res = {}
    for s in slots:
        k = int(s['slot']); res[k] = dict(word=s['word'], kw=s['word'] in KEYWORDS, dets=[], outcome='', cls=None, lat=None)
    for d in dets:
        k = int(np.floor((d['t'] - o - lead) / cadence))
        if k in res: res[k]['dets'].append(d)
    for k, r in res.items():
        ds = r['dets']
        if ds:
            r['cls'] = CLASSES[ds[0]['cls']]; r['lat'] = ds[0]['t'] - o - lead - k * cadence   # s after the slot start (clip starts at +0.5 s)
        if r['kw']: r['outcome'] = 'miss' if not ds else ('correct' if r['cls'] == r['word'] else 'wrong')
        else: r['outcome'] = 'ok' if not ds else 'fa'
    return o, hits, second, res

def summary(res):
    kw = [r for r in res.values() if r['kw']]; neg = [r for r in res.values() if not r['kw']]
    c = sum(r['outcome'] == 'correct' for r in kw); w = sum(r['outcome'] == 'wrong' for r in kw); m = sum(r['outcome'] == 'miss' for r in kw); fa = sum(r['outcome'] == 'fa' for r in neg)
    return dict(n_kw=len(kw), correct=c, wrong=w, miss=m, n_neg=len(neg), fa=fa, acc_slots=(c + len(neg) - fa) / len(res))

def clip_peaks(slots):
    from snn.data.make_dataset import load_item
    pk = {}
    for s in slots:
        y = load_item(s['file']); pk[int(s["slot"])] = float(20 * np.log10(max(np.abs(y).max(), 1e-9)))
    return pk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help='NAME=det.csv[:margin]'); ap.add_argument('--golden', default=os.path.join(ROOT, 'snn', 'board', 'play', 'playback_golden_det.csv'))
    ap.add_argument('--golden_gain', default='', help='optional second golden det csv (chain-gain golden) shown as another column')
    ap.add_argument('--answers', default=os.path.join(ROOT, 'snn', 'board', 'play', 'playback_answers.csv'))
    ap.add_argument('--gain_db', type=float, default=1.0, help='measured chain gain (wav -> ADC), for the level groups')
    ap.add_argument('--pair', default='', help='two run names for the repeatability check, e.g. R0,R0b'); ap.add_argument('--merge', default='', help='run names merged for the confusion matrix (default: the pair)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'snn', 'board', 'log', 'playback_compare')); ap.add_argument('--fig', default='')
    a = ap.parse_args()
    slots = list(csv.DictReader(open(a.answers))); cadence = float(slots[1]['slot_start_s']) - float(slots[0]['slot_start_s']); lead = float(slots[0]['slot_start_s'])
    runs = {}
    for spec in a.runs:
        name, rest = spec.split('='); path, _, mg = rest.partition(':'); mg = int(mg) if mg else 0
        dets = read_det(path, mg); o, hits, second, res = score_slots(dets, slots, lead, cadence)
        runs[name] = dict(path=path, margin=mg, offset=o, hits=hits, second=second, res=res, n_det=len(dets), **summary(res))
    gold = {}
    for name, path, mg in [('골든 m0', a.golden, 0), ('골든 m6212', a.golden, 6212)] + ([(f'골든 {a.gain_db:+.0f} dB m0', a.golden_gain, 0), (f'골든 {a.gain_db:+.0f} dB m6212', a.golden_gain, 6212)] if a.golden_gain else []):
        dets = read_det(path, mg); o, hits, second, res = score_slots(dets, slots, lead, cadence); gold[name] = dict(margin=mg, offset=o, res=res, n_det=len(dets), **summary(res))
    pk = clip_peaks(slots); L = []
    # ---- 1. summary table
    L += ["## 재생 시험 보드 vs 골든 (슬롯 210: 키워드 160, unknown 40, silence 10)", "",
          "| 런 | 마진 | 검출 수 | 정렬(1위/2위) | 키워드 정검출 | 오분류 | 무검출 | 비키워드 오검출(FA) | 슬롯 정확도 |", "|---|---|---|---|---|---|---|---|---|"]
    for name, r in list(gold.items()) + list(runs.items()):
        al = f"{r['hits']}/{r['second']}" if r.get('hits') is not None else '–'
        L.append(f"| {name} | {r['margin']} | {r['n_det']} | {al} | {r['correct']}/{r['n_kw']} ({100 * r['correct'] / r['n_kw']:.1f} %) | {r['wrong']} ({100 * r['wrong'] / r['n_kw']:.1f} %) | {r['miss']} ({100 * r['miss'] / r['n_kw']:.1f} %) | {r['fa']}/{r['n_neg']} | {100 * r['acc_slots']:.1f} % |")
    # ---- 2. repeatability
    pair = a.pair.split(',') if a.pair else []
    if len(pair) == 2 and all(p in runs for p in pair):
        A, B = runs[pair[0]]['res'], runs[pair[1]]['res']
        same = [k for k in A if A[k]['outcome'] == B[k]['outcome'] and A[k]['cls'] == B[k]['cls']]
        same_oc = [k for k in A if A[k]['outcome'] == B[k]['outcome']]
        diff = [k for k in A if k not in same]
        L += ["", f"## 재현성 {pair[0]} vs {pair[1]}", "",
              f"- 슬롯별 완전 일치(결과·클래스) {len(same)}/{len(A)} = {100 * len(same) / len(A):.1f} %, 결과 유형만 일치 {len(same_oc)}/{len(A)} = {100 * len(same_oc) / len(A):.1f} %",
              f"- 키워드 슬롯 중 두 런 모두 정검출 {sum(A[k]['outcome'] == 'correct' and B[k]['outcome'] == 'correct' for k in A)}, 둘 다 실패 {sum(A[k]['kw'] and A[k]['outcome'] != 'correct' and B[k]['outcome'] != 'correct' for k in A)}, 한쪽만 정검출 {sum(A[k]['kw'] and (A[k]['outcome'] == 'correct') != (B[k]['outcome'] == 'correct') for k in A)}",
              "", f"| 슬롯 | 발화 | 클립 피크 dBFS(wav) | {pair[0]} | {pair[1]} | 골든 m0 |", "|---|---|---|---|---|---|"]
        G = gold['골든 m0']['res']
        fmt = lambda r: (r['outcome'] + (f"({r['cls']})" if r['cls'] and r['outcome'] in ('wrong', 'fa') else ''))
        for k in diff: L.append(f"| {k} | {A[k]['word']} | {pk[k]:.1f} | {fmt(A[k])} | {fmt(B[k])} | {fmt(G[k])} |")
    # ---- 3. false alarms: leaked detections?
    L += ["", "## 비키워드 슬롯 오검출(FA)의 성격", "", "| 런 | 슬롯 | 발화 | 판정 | 슬롯 시작 후 s | 직전 슬롯(발화 → 결과) | 해석 |", "|---|---|---|---|---|---|---|"]
    for name, r in runs.items():
        R = r['res']
        for k in sorted(R):
            if R[k]['outcome'] != 'fa': continue
            p = R.get(k - 1); prev = f"{p['word']} → {p['outcome']}" if p else '–'
            leak = p is not None and p['kw'] and p['outcome'] == 'miss' and R[k]['cls'] == p['word']
            L.append(f"| {name} | {k} | {R[k]['word']} | {R[k]['cls']} | {R[k]['lat']:.2f} | {prev} | {'직전 키워드의 늦은 판정(누설)' if leak else '진짜 오검출'} |")
    lat = {name: [r['res'][k]['lat'] for k in r['res'] if r['res'][k]['outcome'] == 'correct'] for name, r in runs.items()}
    L += ["", "정검출 슬롯의 판정 시각(슬롯 시작 기준, 클립은 +0.5 s에 시작): " + '; '.join(f"{n} 중앙값 {np.median(v):.2f} s, 최대 {max(v):.2f} s" for n, v in lat.items() if v) + f" (슬롯 길이 {cadence} s)"]
    # ---- 4. level vs outcome
    bins = [(-100, -20, '< −20'), (-20, -10, '−20 ~ −10'), (-10, -3, '−10 ~ −3'), (-3, 10, '> −3')]
    L += ["", f"## 클립 레벨(wav 피크 dBFS; 보드 입력은 체인 이득 {a.gain_db:+.0f} dB)과 결과", "",
          "| wav 피크 dBFS | 키워드 슬롯 수 | " + ' | '.join(f"{n} 정검출 / 오분류 / 무검출" for n in list(gold)[:1] + list(runs)) + " |", "|---|---|" + "---|" * (1 + len(runs))]
    for lo, hi, lab in bins:
        ks = [int(s['slot']) for s in slots if s['word'] in KEYWORDS and lo <= pk[int(s['slot'])] < hi]
        if not ks: continue
        cells = []
        for name, r in [('골든 m0', gold['골든 m0'])] + list(runs.items()):
            R = r['res']; c = sum(R[k]['outcome'] == 'correct' for k in ks); w = sum(R[k]['outcome'] == 'wrong' for k in ks); m = sum(R[k]['outcome'] == 'miss' for k in ks)
            cells.append(f"{c} ({100 * c / len(ks):.0f} %) / {w} / {m}")
        L.append(f"| {lab} | {len(ks)} | " + ' | '.join(cells) + " |")
    kwk = [int(s['slot']) for s in slots if s['word'] in KEYWORDS]
    for name, r in runs.items():
        R = r['res']; bad = [pk[k] for k in kwk if R[k]['outcome'] != 'correct']; good = [pk[k] for k in kwk if R[k]['outcome'] == 'correct']
        L.append(f"- {name}: 정검출 슬롯의 wav 피크 중앙값 {np.median(good):.1f} dBFS, 실패(오분류+무검출) 슬롯 {np.median(bad):.1f} dBFS; 피크 > −1 dBFS(보드에서 클리핑 가능) 슬롯 {sum(pk[k] > -1 for k in kwk)}개 중 실패 {sum(pk[k] > -1 and R[k]['outcome'] != 'correct' for k in kwk)}")
    # golden-correct but board-failed slots: level of those
    G = gold['골든 m0']['res']
    for name, r in runs.items():
        R = r['res']; lost = [k for k in kwk if G[k]['outcome'] == 'correct' and R[k]['outcome'] != 'correct']; gained = [k for k in kwk if G[k]['outcome'] != 'correct' and R[k]['outcome'] == 'correct']
        L.append(f"- {name}: 골든은 맞혔으나 보드가 놓친 슬롯 {len(lost)}개(피크 중앙값 {np.median([pk[k] for k in lost]):.1f} dBFS, 무검출 {sum(R[k]['outcome'] == 'miss' for k in lost)} / 오분류 {sum(R[k]['outcome'] == 'wrong' for k in lost)}), 보드만 맞힌 슬롯 {len(gained)}개")
    # ---- 5. confusion matrix figure
    merge = (a.merge.split(',') if a.merge else pair) or list(runs)
    cols = KEYWORDS + ['무검출']
    def cm_of(res_list):
        cm = np.zeros((len(CLASSES), len(cols)), int)
        for R in res_list:
            for k, r in R.items():
                i = CLASSES.index(r['word']); j = cols.index(r['cls']) if r['cls'] in KEYWORDS else len(KEYWORDS); cm[i, j] += 1
        return cm
    cmb = cm_of([runs[n]['res'] for n in merge]); cmg = cm_of([gold['골든 m0']['res']])
    L += ["", f"## 혼동행렬 (보드 {'+'.join(merge)}, 행 = 발화, 열 = 판정)", "", "| 발화 \\ 판정 | " + ' | '.join(cols) + " |", "|---|" + "---|" * len(cols)]
    for i, w in enumerate(CLASSES): L.append(f"| {w} | " + ' | '.join(str(v) for v in cmb[i]) + " |")
    if a.fig:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family': ['Noto Sans CJK KR', 'DejaVu Sans'], 'savefig.dpi': 300, 'axes.unicode_minus': False})
        INK = '#1b1b1b'; fig, axs = plt.subplots(1, 2, figsize=(11, 4.9))
        nb = len(merge)
        for ax, m, title in zip(axs, (cmb, cmg), (f"보드 재생 시험 {'+'.join(merge)} (마진 0, {nb}회 합산)", '디지털 골든 (같은 wav, 마진 0)')):
            mn = m / m.sum(1, keepdims=True); ax.imshow(mn, cmap='Blues', vmin=0, vmax=1)
            ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(CLASSES))); ax.set_xticklabels(cols, rotation=60, fontsize=7); ax.set_yticklabels(CLASSES, fontsize=7)
            for i in range(len(CLASSES)):
                for j in range(len(cols)):
                    if m[i, j] > 0: ax.text(j, i, int(m[i, j]), ha='center', va='center', fontsize=6, color='white' if mn[i, j] > 0.5 else INK)
            kw_rows = [CLASSES.index(w) for w in KEYWORDS]; acc = sum(m[i, i] for i in kw_rows) / m[kw_rows].sum()
            ax.set_title(f"{title} — 키워드 정검출 {100 * acc:.1f} %", fontsize=9.5); ax.set_xlabel('판정'); ax.set_ylabel('발화'); ax.grid(False)
            ax.axvline(len(KEYWORDS) - 0.5, color='#888', lw=0.6); ax.axhline(len(KEYWORDS) - 0.5, color='#888', lw=0.6)
        fig.tight_layout(); fig.savefig(a.fig); print('fig ->', a.fig)
    txt = '\n'.join(L); print(txt); open(a.out + '.md', 'w').write(txt + '\n')
    json.dump(dict(runs={n: {k: v for k, v in r.items() if k != 'res'} | dict(slots={k: dict(word=x['word'], outcome=x['outcome'], cls=x['cls'], lat=x['lat']) for k, x in r['res'].items()}) for n, r in runs.items()},
                   golden={n: {k: v for k, v in r.items() if k != 'res'} for n, r in gold.items()}, peaks=pk), open(a.out + '.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
