"""
playback_score.py — score a board playback test (BOARD_STEP2 §1B) or its digital golden run.
  answers : snn/board/play/playback_answers.csv (slot, slot_start_s, clip_start_s, word, class_id, file)
  det     : ILA export(s) captured with capture condition dbg_det_valid == 1 (one row per detection: dbg_det_class,
            dbg_cnt_frames = 10 ms time stamp, dbg_live_count, dbg_cnt_snn_frames), or playback_golden_det.csv (same
            columns + 'margin', which --margin filters to emulate the sw[3:2] operating point).
Alignment: the board's frame counter and the wav start are not synchronised, so the offset o (s) between detection time
and wav time is found automatically: the o (0.05 s grid) that maximises the number of detections falling in a keyword slot
of the same class; --t0 FRAMES (cnt_frames at the wav start) overrides it. A detection at time t belongs to slot
floor((t - o - lead) / cadence). Scoring per slot: keyword slot -> correct / wrong class / miss (no detection) / multi;
unknown or silence slot -> correct if no detection, else false alarm. Writes <out>.md and .json.
usage: python snn/eval/playback_score.py --det snn/board/log/play_A0_det.csv [--answers ...] [--margin 0] [--t0 N] [--out ...]
"""
import os, sys, csv, json, argparse
import numpy as np
KEYWORDS = ['yes', 'no', 'stop', 'go', 'up', 'down', 'left', 'right', 'on', 'off']
CLASSES = KEYWORDS + ['silence', 'unknown']
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

def read_det(path, margin=0):
    rows = list(csv.reader(open(path))); hdr = [h.strip() for h in rows[0]]
    def col(c):
        ci = [i for i, h in enumerate(hdr) if c in h]; return ci[0] if ci else None
    radix = rows[1] if len(rows) > 1 and rows[1] and rows[1][0].lower().startswith('radix') else None
    def val(r, ci):
        v = r[ci].strip(); rx = radix[ci].strip().upper() if radix else 'UNSIGNED'
        return int(v, 16) if 'HEX' in rx else int(v)
    ci_c, ci_t, ci_n, ci_s, ci_m = col('dbg_det_class'), col('dbg_cnt_frames'), col('dbg_live_count'), col('dbg_cnt_snn_frames'), col('margin')
    out = []
    for r in rows[1:]:
        if not r or r[0].lower().startswith('radix') or len(r) <= ci_c or not r[ci_c].strip(): continue
        m = val(r, ci_m) if ci_m is not None else None
        if m is not None and m < margin: continue
        out.append(dict(cls=val(r, ci_c), t=val(r, ci_t) * 0.01, live=val(r, ci_n) if ci_n is not None else None, snn=val(r, ci_s) if ci_s is not None else None, margin=m))
    ded = []
    for d in out:
        if ded and d['live'] is not None and d['live'] == ded[-1]['live']: continue
        ded.append(d)
    return ded

def find_offset(dets, slots, lead, cadence, step=0.05):
    """offset o such that det time - o = wav time; maximise same-class hits in keyword slots. Any o inside a plateau of
    about one slot width gives the same hits, so the plateau centre is returned; 'second' is the best score at least half a
    slot away from it (a genuinely different alignment)."""
    if not dets: return 0.0, 0, 0
    kw = [(float(s['slot_start_s']), int(s['class_id'])) for s in slots if s['word'] in KEYWORDS]
    starts = np.array([t for t, _ in kw]); cls = np.array([c for _, c in kw])
    tmin, tmax = min(d['t'] for d in dets), max(d['t'] for d in dets)
    cands = np.arange(tmin - starts.max() - cadence, tmax - starts.min() + cadence, step); hits = np.zeros(len(cands), int)
    for k, o in enumerate(cands):
        for d in dets:
            i = np.searchsorted(starts, d['t'] - o, side='right') - 1
            if i >= 0 and (d['t'] - o) - starts[i] < cadence and cls[i] == d['cls']: hits[k] += 1
    best = int(hits.max()); top = np.where(hits == best)[0]
    # contiguous block of the maximum that contains the median top index
    mid = top[len(top) // 2]; lo = hi = mid
    while lo - 1 >= 0 and hits[lo - 1] == best: lo -= 1
    while hi + 1 < len(hits) and hits[hi + 1] == best: hi += 1
    o = float(cands[(lo + hi) // 2]); far = np.abs(cands - o) >= cadence / 2
    second = int(hits[far].max()) if far.any() else 0
    return o, best, second

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', nargs='+', required=True); ap.add_argument('--answers', default=os.path.join(ROOT, 'snn', 'board', 'play', 'playback_answers.csv'))
    ap.add_argument('--margin', type=int, default=0, help="golden CSV only: keep detections with margin >= this (board: set by sw[3:2])")
    ap.add_argument('--t0', type=float, default=None, help='cnt_frames value at the wav start (skips the automatic alignment)')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    slots = list(csv.DictReader(open(a.answers))); cadence = float(slots[1]['slot_start_s']) - float(slots[0]['slot_start_s']); lead = float(slots[0]['slot_start_s'])
    dets = sum([read_det(p, a.margin) for p in a.det], []); dets.sort(key=lambda d: d['t'])
    if a.t0 is not None: o, hits, second = a.t0 * 0.01, None, None
    else: o, hits, second = find_offset(dets, slots, lead, cadence)
    assigned = {int(s['slot']): [] for s in slots}; extra = []
    for d in dets:
        s = int(np.floor((d['t'] - o - lead) / cadence)); (assigned[s] if s in assigned else extra).append(d)
    cols = KEYWORDS + ['miss', 'multi']; truths = CLASSES
    cm = {w: {c: 0 for c in cols} for w in truths}; per = {w: [0, 0] for w in truths}; fa_slots = 0; n_neg = 0
    for s in slots:
        w = s['word']; ds = assigned[int(s['slot'])]; per[w][1] += 1
        if w in KEYWORDS:
            if not ds: cm[w]['miss'] += 1
            else:
                if len(ds) > 1: cm[w]['multi'] += len(ds) - 1
                c = CLASSES[ds[0]['cls']]; cm[w][c if c in KEYWORDS else 'miss'] += 1
                if c == w: per[w][0] += 1
        else:
            n_neg += 1
            if ds: fa_slots += 1; cm[w][CLASSES[ds[0]['cls']] if ds[0]['cls'] < 10 else 'miss'] += 1; cm[w]['multi'] += len(ds) - 1
            else: per[w][0] += 1
    n_kw = sum(per[w][1] for w in KEYWORDS); hit = sum(per[w][0] for w in KEYWORDS); miss = sum(cm[w]['miss'] for w in KEYWORDS)
    wrong = n_kw - hit - miss; acc_slots = (hit + sum(per[w][0] for w in ('silence', 'unknown'))) / len(slots)
    hrs = (len(slots) * cadence) / 3600
    L = [f"## 재생 시험 채점: {', '.join(os.path.basename(p) for p in a.det)} (마진 {a.margin}, 슬롯 {len(slots)}, 검출 {len(dets)}, 오프셋 {o:.2f} s"
         + (f", 정렬 점수 {hits}/{len(dets)} (2위 {second})" if hits is not None else ', t0 지정') + f", 슬롯 밖 검출 {len(extra)})\n",
         f"- 키워드 {n_kw}회: 정검출 {hit} (**{hit / max(n_kw, 1):.1%}**), 오분류 {wrong} ({wrong / max(n_kw, 1):.1%}), 무반응 {miss} ({miss / max(n_kw, 1):.1%})",
         f"- unknown/silence {n_neg}회: 오검출 슬롯 {fa_slots} ({fa_slots / max(n_neg, 1):.1%}) → 재생 구간 기준 {fa_slots / hrs:.1f} FA/h",
         f"- 슬롯 정확도(키워드 정검출 + 비키워드 무검출) **{acc_slots:.1%}**", "",
         "| 발화 \\ 판정 | " + ' | '.join(cols) + " | 정검출/발화 |", "|---|" + "---|" * (len(cols) + 1)]
    for w in truths: L.append(f"| {w} | " + ' | '.join(str(cm[w][c]) for c in cols) + f" | {per[w][0]} / {per[w][1]} |")
    if dets and dets[-1]['snn'] is not None:
        L.append(f"\n가동률(마지막 행): snn_frames {dets[-1]['snn']} / frames {int(dets[-1]['t'] * 100)} = {dets[-1]['snn'] / max(dets[-1]['t'] * 100, 1):.4f}")
    if hits is not None and second is not None and hits and second >= 0.8 * hits:
        L.append(f"\n**경고**: 정렬이 모호함(1위 {hits}, 2위 {second}). 재생 시작 시각의 cnt_frames를 `--t0`로 주거나 ILA를 재생 직전에 Run Trigger 하세요.")
    txt = '\n'.join(L); print(txt)
    out = a.out or (os.path.splitext(a.det[0])[0] + f"_m{a.margin}_result")
    open(out + '.md', 'w').write(txt + '\n')
    json.dump(dict(offset_s=o, align_hits=hits, align_second=second, cm=cm, per=per, n_kw=n_kw, hit=hit, wrong=wrong, miss=miss, fa_slots=fa_slots, n_neg=n_neg, acc_slots=acc_slots, extra=len(extra)),
              open(out + '.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
