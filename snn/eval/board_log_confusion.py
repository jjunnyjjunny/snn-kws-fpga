"""
board_log_confusion.py — board live-test scoring (BOARD_STEP2): utterance log (slot cadence) x ILA detection capture.
  utterance log CSV (snn/board/log/board_log_template.csv filled in): slot,word,distance_cm,note   (one utterance per slot,
      spoken every CADENCE seconds after the start cue; word = keyword, or an unknown word, or 'none' for a silent slot)
  detection CSV(s): Vivado ILA export with BASIC capture, capture condition dbg_det_valid == 1 (one row per detection),
      columns dbg_det_class, dbg_cnt_frames (10 ms frame counter = time stamp), dbg_live_count, dbg_cnt_snn_frames.
Each detection is assigned to the slot floor((t - t0) / cadence) where t = cnt_frames x 10 ms and t0 = the frame count at
the start cue (--t0; default: first detection time minus half a cadence, i.e. the first utterance was detected).
Outputs: confusion matrix (true word -> detected class / miss / extra), keyword recall per distance, false detections on
unknown/none slots, per-session markdown. Separate quiet/talk sessions: give only detection CSVs with --fa_minutes.
usage:
  python snn/eval/board_log_confusion.py --log snn/board/log/session1.csv --det snn/board/log/session1_det.csv [--cadence 5] [--t0 FRAMES]
  python snn/eval/board_log_confusion.py --det snn/board/log/quiet10min_det.csv --fa_minutes 10
"""
import os, sys, csv, argparse, json
import numpy as np
KEYWORDS = ['yes', 'no', 'stop', 'go', 'up', 'down', 'left', 'right', 'on', 'off']
CLASSES = KEYWORDS + ['silence', 'unknown']

def read_ila(path):
    rows = list(csv.reader(open(path))); hdr = [h.strip() for h in rows[0]]
    def col(c):
        ci = [i for i, h in enumerate(hdr) if c in h]; return ci[0] if ci else None
    radix = rows[1] if rows[1] and rows[1][0].lower().startswith('radix') else None
    def val(r, ci):
        v = r[ci].strip(); rx = radix[ci].strip().upper() if radix else 'HEX'
        return int(v, 16) if rx == 'HEX' else int(v)
    ci_c, ci_t, ci_n, ci_s = col('dbg_det_class'), col('dbg_cnt_frames'), col('dbg_live_count'), col('dbg_cnt_snn_frames')
    out = []
    for r in rows[1:]:
        if not r or r[0].lower().startswith('radix') or len(r) <= ci_c or not r[ci_c].strip(): continue
        out.append(dict(cls=val(r, ci_c), t=val(r, ci_t) * 0.01, live=val(r, ci_n) if ci_n is not None else None, snn=val(r, ci_s) if ci_s is not None else None))
    # one row per detection pulse: drop consecutive duplicates (same live_count) in case the qualifier captured several cycles
    ded = []
    for d in out:
        if ded and d['live'] is not None and d['live'] == ded[-1]['live']: continue
        ded.append(d)
    return ded

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default=''); ap.add_argument('--det', nargs='+', required=True)
    ap.add_argument('--cadence', type=float, default=5.0); ap.add_argument('--t0', type=float, default=None, help='frame count at the start cue')
    ap.add_argument('--fa_minutes', type=float, default=0.0, help='false-alarm session length (no utterance log)')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    dets = sum([read_ila(p) for p in a.det], [])
    dets.sort(key=lambda d: d['t'])
    if a.fa_minutes > 0:
        n = len(dets); print(f"false-alarm session: {n} detections in {a.fa_minutes:.1f} min -> {n / a.fa_minutes * 60:.1f} /h ; classes "
                              + ', '.join(f"{CLASSES[d['cls']]}@{d['t']:.1f}s" for d in dets))
        if dets and dets[-1]['snn'] is not None:
            print(f"duty at last row: snn_frames {dets[-1]['snn']} / frames {int(dets[-1]['t'] * 100)} = {dets[-1]['snn'] / max(dets[-1]['t'] * 100, 1):.4f}")
        return
    log = [r for r in csv.DictReader(open(a.log)) if r.get('word', '').strip()]
    t0 = a.t0 * 0.01 if a.t0 is not None else (dets[0]['t'] - a.cadence / 2 if dets else 0.0)
    slots = {int(r['slot']): r for r in log}
    assigned = {k: [] for k in slots}; extra = []
    for d in dets:
        s = int(np.floor((d['t'] - t0) / a.cadence))
        (assigned[s] if s in assigned else extra).append(d)
    truths = sorted({r['word'] for r in slots.values()}, key=lambda w: (w not in KEYWORDS, w))
    cols = KEYWORDS + ['miss', 'multi']
    cm = {w: {c: 0 for c in cols} for w in truths}
    per_dist = {}
    for s, r in slots.items():
        w = r['word']; ds = assigned[s]; dist = r.get('distance_cm', '')
        key = (w, dist); per_dist.setdefault(key, [0, 0]); per_dist[key][1] += 1
        if not ds: cm[w]['miss'] += 1
        else:
            if len(ds) > 1: cm[w]['multi'] += len(ds) - 1
            c = CLASSES[ds[0]['cls']]; cm[w][c if c in cols else 'miss'] += 1
            if c == w: per_dist[key][0] += 1
    L = [f"## 보드 라이브 결과: {os.path.basename(a.log)} (슬롯 {len(slots)}, 검출 {len(dets)}, cadence {a.cadence} s, t0 {t0:.1f} s, 슬롯 밖 검출 {len(extra)})\n",
         "| 발화 \\ 판정 | " + ' | '.join(cols) + " |", "|---|" + "---|" * len(cols)]
    for w in truths: L.append(f"| {w} | " + ' | '.join(str(cm[w][c]) for c in cols) + " |")
    kw = [w for w in truths if w in KEYWORDS]
    tot = sum(sum(cm[w][c] for c in KEYWORDS) + cm[w]['miss'] for w in kw); hit = sum(cm[w][w] for w in kw)
    wrong = sum(cm[w][c] for w in kw for c in KEYWORDS if c != w); miss = sum(cm[w]['miss'] for w in kw)
    L.append(f"\n키워드 {tot}회: 정검출 {hit} ({hit / max(tot, 1):.1%}), 오분류 {wrong}, 무반응 {miss}")
    unk = [w for w in truths if w not in KEYWORDS]
    if unk:
        fa = sum(cm[w][c] for w in unk for c in KEYWORDS); n_unk = sum(sum(cm[w][c] for c in cols) for w in unk)
        L.append(f"unknown/none 슬롯 {n_unk}회: 키워드로 오검출 {fa} ({fa / max(n_unk, 1):.1%})")
    L.append("\n| 단어 | 거리 cm | 정검출 / 발화 |"); L.append("|---|---|---|")
    for (w, d), (h, n) in sorted(per_dist.items(), key=lambda kv: (kv[0][0] not in KEYWORDS, kv[0][0], kv[0][1])): L.append(f"| {w} | {d} | {h} / {n} |")
    if dets and dets[-1]['snn'] is not None:
        L.append(f"\n가동률(마지막 행): snn_frames {dets[-1]['snn']} / frames {int(dets[-1]['t'] * 100)} = {dets[-1]['snn'] / max(dets[-1]['t'] * 100, 1):.4f}")
    txt = '\n'.join(L); print(txt)
    if a.out: open(a.out, 'w').write(txt + '\n')
    json.dump(dict(cm=cm, per_dist={f"{w}@{d}": v for (w, d), v in per_dist.items()}, extra=len(extra), t0=t0), open((a.out or a.log) + '.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
