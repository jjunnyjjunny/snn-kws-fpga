"""
power_log_analyze.py — board input-power differential from a Keysight N6705C Data Logger CSV (BOARD_STEP2 §4).

Input
  --log   CSV exported from the N6705C data logger (or 14585A / BenchVue): a time column (seconds, or a clock
          HH:MM:SS[.fff] / ISO date-time) plus voltage [V] and current [A] columns (a power [W] column is used if
          present). Header names are matched case-insensitively (time/elapsed/sec, volt/V, curr/amp/I, power/W);
          delimiter is sniffed (',', ';', tab). Override with --cols time,volt,curr[,power] (0-based indices).
          Rows that do not parse as numbers (title, units, comments) are skipped. --period S replaces a missing time
          column by index*period.
  --marks text file with one state start per line, written by hand while measuring:
              # label  start-time            (same time format as the log: seconds or clock)
              P0 12.0
              P1 75.5
              P0 14:23:10.5
          Labels repeat for repeats (P0 three times -> 3 repeats). A line '<label> <start> <end>' gives an explicit
          end; otherwise the interval is --dur seconds (default 60). Comments start with '#'.
  --skip  seconds discarded at the start of every interval (settling, default 5); --trim seconds discarded at its end
          (default 2).  --t_offset adds a constant to all mark times (if the stopwatch was started late).

Output (<out>.md / .json, optional --fig PNG at 300 dpi)
  per state: repeats, mean V / mA / mW, between-repeat std (mW), mean within-interval std (mW), samples, and the
  difference to the reference state (--ref, default the first label) with its uncertainty
  sqrt(std_ref^2/n_ref + std_s^2/n_s) (between-repeat). Extra pairs (--pairs P1-P4,P2-P5) are reported the same way.
  A difference smaller than 2x its uncertainty, or smaller than the logger resolution (--resolution mW, default 0),
  is marked '해상도 이하' (below resolution), as BOARD_STEP2 §4 requires.
usage:
  python snn/eval/power_log_analyze.py --log snn/board/log/power_dlog.csv --marks snn/board/log/power_marks.txt \
         [--dur 60 --skip 5 --trim 2 --ref P0 --pairs P1-P4,P2-P5,P3-P6] [--out snn/board/log/power_result] [--fig]
"""
import os, sys, re, csv, json, argparse, datetime
import numpy as np

def parse_time(s):
    """seconds (float) or clock string -> seconds; clock strings are returned as absolute seconds of day / epoch."""
    s = s.strip().strip('"')
    try: return float(s)
    except ValueError: pass
    for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S.%f', '%Y/%m/%d %H:%M:%S', '%m/%d/%Y %H:%M:%S.%f', '%m/%d/%Y %H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S'):
        try: return datetime.datetime.strptime(s, fmt).timestamp()
        except ValueError: pass
    m = re.fullmatch(r'(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)', s)
    if m: return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.fullmatch(r'(\d{1,2}):(\d{2}(?:\.\d+)?)', s)
    if m: return int(m.group(1)) * 60 + float(m.group(2))
    return None

def to_float(s):
    s = s.strip().strip('"').replace(',', '.') if s.count(',') == 1 and '.' not in s else s.strip().strip('"')
    try: return float(s)
    except ValueError: return None

def read_log(path, cols=None, period=None):
    raw = open(path, encoding='utf-8-sig', errors='replace').read().splitlines()
    lines = [l for l in raw if l.strip()]
    sample = '\n'.join(lines[:50])
    delim = ','
    try: delim = csv.Sniffer().sniff(sample, delimiters=',;\t').delimiter
    except csv.Error:
        for d in ('\t', ';'):
            if sample.count(d) > sample.count(','): delim = d
    rows = [[c.strip() for c in l.split(delim)] for l in lines]
    # header = last row before the first mostly-numeric row that contains a time/volt/curr keyword
    def numeric_row(r): return sum(1 for c in r if to_float(c) is not None or parse_time(c) is not None) >= max(2, len(r) // 2)
    first = next((i for i, r in enumerate(rows) if numeric_row(r) and len(r) >= 2), None)
    if first is None: raise SystemExit(f"no numeric rows found in {path}")
    hdr = None
    for i in range(first - 1, -1, -1):
        if any(re.search(r'time|elapsed|sec|volt|curr|amp|power|^[viwp]\b', c, re.I) for c in rows[i]): hdr = rows[i]; break
    if cols is None:
        ci_t = ci_v = ci_i = ci_p = None
        if hdr:
            for k, h in enumerate(hdr):
                hl = h.lower()
                if ci_t is None and re.search(r'time|elapsed|sec|date', hl): ci_t = k
                elif ci_v is None and re.search(r'volt|\bv\b|\(v\)|\[v\]', hl): ci_v = k
                elif ci_i is None and re.search(r'curr|\bamp|\bi\b|\(a\)|\[a\]', hl): ci_i = k
                elif ci_p is None and re.search(r'power|watt|\bp\b|\(w\)|\[w\]', hl): ci_p = k
        if ci_v is None or ci_i is None:            # fall back to positional: [time,] V, I
            n = len(rows[first]); ci_t = 0 if (ci_t is None and n >= 3) else ci_t; ci_v = 1 if n >= 3 else 0; ci_i = 2 if n >= 3 else 1
    else:
        ci_t, ci_v, ci_i = cols[0], cols[1], cols[2]; ci_p = cols[3] if len(cols) > 3 else None
    t, v, i, p = [], [], [], []
    for r in rows[first:]:
        if len(r) <= max(ci_v, ci_i): continue
        fv, fi = to_float(r[ci_v]), to_float(r[ci_i])
        if fv is None or fi is None: continue
        ft = parse_time(r[ci_t]) if ci_t is not None and ci_t < len(r) else None
        if ci_t is not None and ft is None: continue
        t.append(ft); v.append(fv); i.append(fi); p.append(to_float(r[ci_p]) if ci_p is not None and ci_p < len(r) else None)
    v, i = np.array(v), np.array(i)
    if ci_t is None or any(x is None for x in t):
        if period is None: raise SystemExit("log has no usable time column: pass --period <s> (logger sample period) or --cols")
        t = np.arange(len(v)) * period; t_is_clock = False
    else:
        t = np.array(t, float); t_is_clock = any(parse_time(r[ci_t]) is not None and to_float(r[ci_t]) is None for r in rows[first:first + 3])
    pw = np.array([x if x is not None else np.nan for x in p]) if any(x is not None for x in p) else v * i
    if np.isnan(pw).any(): pw = np.where(np.isnan(pw), v * i, pw)
    info = dict(delimiter=repr(delim), header=hdr, cols=dict(time=ci_t, volt=ci_v, curr=ci_i, power=ci_p), n=len(v), t_is_clock=t_is_clock,
                t_start=float(t[0]), t_end=float(t[-1]), sample_period_s=float(np.median(np.diff(t))) if len(t) > 1 else None)
    return t, v, i, pw, info

def read_marks(path, dur):
    marks = []
    for l in open(path, encoding='utf-8-sig'):
        l = l.split('#')[0].strip()
        if not l: continue
        parts = re.split(r'[,\s]+', l)
        if len(parts) < 2: raise SystemExit(f"bad mark line: {l!r} (expected '<label> <start> [end]')")
        lab = parts[0]; st = parse_time(' '.join(parts[1:3])) if len(parts) >= 3 and parse_time(' '.join(parts[1:3])) is not None and parse_time(parts[1]) is None else parse_time(parts[1])
        rest = parts[2:] if parse_time(parts[1]) is not None else parts[3:]
        en = parse_time(rest[0]) if rest else None
        if st is None: raise SystemExit(f"cannot parse start time in: {l!r}")
        marks.append(dict(label=lab, start=st, end=en if en is not None else st + dur, explicit_end=en is not None))
    return marks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True); ap.add_argument('--marks', required=True)
    ap.add_argument('--dur', type=float, default=60.0); ap.add_argument('--skip', type=float, default=5.0); ap.add_argument('--trim', type=float, default=2.0)
    ap.add_argument('--t_offset', type=float, default=0.0, help='added to all mark times (s)')
    ap.add_argument('--cols', default=None, help='0-based column indices time,volt,curr[,power]'); ap.add_argument('--period', type=float, default=None)
    ap.add_argument('--ref', default=None, help='reference state label (default: first label in the marks file)')
    ap.add_argument('--pairs', default='', help="extra differences, e.g. 'P1-P4,P2-P5' (first minus second)")
    ap.add_argument('--resolution', type=float, default=0.0, help='logger power resolution in mW (differences below it are flagged)')
    ap.add_argument('--out', default=''); ap.add_argument('--fig', action='store_true')
    a = ap.parse_args()
    cols = [int(c) for c in a.cols.split(',')] if a.cols else None
    t, v, i, pw, info = read_log(a.log, cols, a.period)
    marks = read_marks(a.marks, a.dur)
    for m in marks: m['start'] += a.t_offset; m['end'] += a.t_offset
    labels = []
    for m in marks:
        if m['label'] not in labels: labels.append(m['label'])
    ref = a.ref or labels[0]
    if ref not in labels: raise SystemExit(f"reference {ref} not in marks {labels}")
    # per-interval statistics
    per = {lab: [] for lab in labels}
    for k, m in enumerate(marks):
        lo, hi = m['start'] + a.skip, m['end'] - a.trim
        sel = (t >= lo) & (t < hi)
        if sel.sum() < 2:
            print(f"warning: {m['label']} #{k} [{lo:.1f}, {hi:.1f}) has {int(sel.sum())} samples (log {t[0]:.1f}..{t[-1]:.1f}); skipped"); continue
        per[m['label']].append(dict(idx=k, t0=float(lo), t1=float(hi), n=int(sel.sum()), v=float(v[sel].mean()), i_mA=float(1e3 * i[sel].mean()),
                                    p_mW=float(1e3 * pw[sel].mean()), p_std_mW=float(1e3 * pw[sel].std(ddof=0)), p_min_mW=float(1e3 * pw[sel].min()), p_max_mW=float(1e3 * pw[sel].max())))
    st = {}
    for lab in labels:
        rs = per[lab]
        if not rs: continue
        pm = np.array([r['p_mW'] for r in rs])
        st[lab] = dict(reps=len(rs), n=int(sum(r['n'] for r in rs)), v=float(np.mean([r['v'] for r in rs])), i_mA=float(np.mean([r['i_mA'] for r in rs])),
                       p_mW=float(pm.mean()), p_rep_std_mW=float(pm.std(ddof=1)) if len(rs) > 1 else float('nan'), p_within_std_mW=float(np.mean([r['p_std_mW'] for r in rs])),
                       p_reps_mW=[float(x) for x in pm], intervals=rs)
    def diff(x, y):
        sx, sy = st[x], st[y]
        d = sx['p_mW'] - sy['p_mW']
        # uncertainty of the difference of means: between-repeat std if available, else within-interval std / sqrt(n)
        ux = (sx['p_rep_std_mW'] / np.sqrt(sx['reps'])) if sx['reps'] > 1 and not np.isnan(sx['p_rep_std_mW']) else sx['p_within_std_mW'] / np.sqrt(sx['n'])
        uy = (sy['p_rep_std_mW'] / np.sqrt(sy['reps'])) if sy['reps'] > 1 and not np.isnan(sy['p_rep_std_mW']) else sy['p_within_std_mW'] / np.sqrt(sy['n'])
        u = float(np.sqrt(ux ** 2 + uy ** 2)); below = abs(d) < max(2 * u, a.resolution)
        return dict(a=x, b=y, diff_mW=float(d), unc_mW=u, below_resolution=bool(below))
    diffs = [diff(lab, ref) for lab in st if lab != ref]
    pairs = []
    for pr in [p for p in a.pairs.split(',') if p.strip()]:
        x, y = [s.strip() for s in pr.split('-')]
        if x in st and y in st: pairs.append(diff(x, y))
        else: print(f"warning: pair {pr} skipped (unknown label)")
    # drift of the reference across its repeats
    drift = None
    if ref in st and st[ref]['reps'] > 1: drift = dict(first_mW=st[ref]['p_reps_mW'][0], last_mW=st[ref]['p_reps_mW'][-1], delta_mW=st[ref]['p_reps_mW'][-1] - st[ref]['p_reps_mW'][0])
    # ---- report
    L = [f"## 보드 입력 전력 (N6705C 로그 `{os.path.basename(a.log)}`, 구간 {a.dur:.0f} s, 앞 {a.skip:.0f} s·뒤 {a.trim:.0f} s 제외, 샘플 주기 {info['sample_period_s']:.3g} s, 기준 {ref})\n",
         "| 상태 | 반복 | 전압 V | 전류 mA | 전력 mW (평균) | 반복 간 표준편차 mW | 구간 내 표준편차 mW | 반복별 mW | 기준 대비 차분 mW |", "|---|---|---|---|---|---|---|---|---|"]
    dmap = {d['a']: d for d in diffs}
    for lab in st:
        s = st[lab]; d = dmap.get(lab)
        dtxt = '기준' if lab == ref else (f"{d['diff_mW']:+.1f} ± {d['unc_mW']:.1f}" + (' (해상도 이하)' if d['below_resolution'] else ''))
        L.append(f"| {lab} | {s['reps']} | {s['v']:.3f} | {s['i_mA']:.2f} | **{s['p_mW']:.1f}** | {s['p_rep_std_mW']:.1f} | {s['p_within_std_mW']:.1f} | {', '.join(f'{x:.1f}' for x in s['p_reps_mW'])} | {dtxt} |")
    if pairs:
        L += ["", "| 차분 | mW | 불확도 mW (반복 간) | 판정 |", "|---|---|---|---|"]
        for d in pairs: L.append(f"| {d['a']} − {d['b']} | {d['diff_mW']:+.1f} | ± {d['unc_mW']:.1f} | {'해상도 이하' if d['below_resolution'] else '유의'} |")
    if drift: L.append(f"\n기준 상태 드리프트: 첫 반복 {drift['first_mW']:.1f} → 마지막 반복 {drift['last_mW']:.1f} mW ({drift['delta_mW']:+.1f} mW).")
    L.append(f"\n판정 규칙: |차분| < max(2 × 불확도, 로거 분해능 {a.resolution:.1f} mW)이면 '해상도 이하'로 적고 Vivado 추정표와 섞지 않는다. 불확도 = sqrt(σ_a²/n_a + σ_b²/n_b), σ = 반복 간 표준편차(반복 1회면 구간 내 표준편차/√샘플 수).")
    txt = '\n'.join(L); print(txt)
    out = a.out or os.path.splitext(a.log)[0] + '_result'
    open(out + '.md', 'w').write(txt + '\n')
    json.dump(dict(log=a.log, marks=a.marks, info=info, settings=dict(dur=a.dur, skip=a.skip, trim=a.trim, t_offset=a.t_offset, ref=ref, resolution_mW=a.resolution),
                   states=st, diffs_vs_ref=diffs, pairs=pairs, drift=drift), open(out + '.json', 'w'), indent=1)
    if a.fig:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        plt.rcParams['font.family'] = ['Noto Sans CJK KR', 'DejaVu Sans']
        fig, ax = plt.subplots(figsize=(10, 3.6)); tt = t - t[0]
        ax.plot(tt, 1e3 * pw, color='#2a78d6', lw=0.6, label='입력 전력')
        pal = ['#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6', '#008300', '#e34948']
        for m in marks:
            k = labels.index(m['label']); ax.axvspan(m['start'] + a.skip - t[0], m['end'] - a.trim - t[0], color=pal[k % len(pal)], alpha=0.15)
            ax.text((m['start'] + m['end']) / 2 - t[0], ax.get_ylim()[1] if False else np.nanpercentile(1e3 * pw, 99.5), m['label'], ha='center', va='bottom', fontsize=8, color=pal[k % len(pal)])
        ax.set_xlabel('시간 (s)'); ax.set_ylabel('mW'); ax.set_title('보드 입력 전력과 상태 구간 (음영 = 집계 구간)', fontsize=10); ax.grid(alpha=0.3); ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout(); fig.savefig(out + '.png', dpi=300); print('figure:', out + '.png')

if __name__ == '__main__':
    main()
