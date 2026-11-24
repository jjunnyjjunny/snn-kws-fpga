"""
board_capture_check.py — analyse 16 kHz audio captures exported from the ILA (mode 3: dbg_cap_sample[17:0] =
audio_l_in[23:6], one CSV row per new_sample, Vivado `write_hw_ila_data -csv_file` format with a 'Radix' row).
usage:
  python snn/eval/board_capture_check.py snn/data/board_capture/*.csv [--run snn/runs/M3k3e60_U2_q8_kd_s0]
       [--out snn/runs/board_capture] [--fig snn/docs/figures] [--ref_n 200] [--pad_lead 0.2 --pad_trail 1.0]
For every capture: wav export, level (peak/RMS dBFS, 18- and 24-bit full scale), noise floor / SNR, band energy
ratios, /s/ (4-8 kHz) vs vowel (0-1 kHz) frame levels, features vs the training distribution, the integer pipeline
golden (front-end + VAD + SNN + decision, same parameters as kws_top) and the clip-mode prediction, E1 trajectory.
Groups (file name before the last '_NN') are averaged and compared with Speech Commands test clips of the same word.
Writes <out>/summary.json, <out>/summary.md, <out>/channel_response.npz and three figures.
"""
import os, sys, json, argparse, csv, re, glob
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from snn.frontend.frontend_ref import FrontendInt, wav_float_to_pcm24
from snn.golden.snn_int_ref import run_int
from snn.golden.pipeline_int_ref import run_pipeline
from snn.data.make_dataset import DATA, KEYWORDS, load_item

SR = 16000; FS18 = 1 << 17; FS24 = 1 << 23
PRM = dict(T_on=600, T_off=360, N_on=4, H=30, P=20, L_min=25, L_max=130)     # kws_top defaults (B-2 val-selected)
BANDS = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']       # dataviz default categorical order

def db(x, ref): return 20 * np.log10(max(float(x), 1e-9) / ref)

def read_capture(path, col='dbg_cap_sample', valid_col='dbg_cap_valid'):
    rows = list(csv.reader(open(path)))
    hdr = [h.strip() for h in rows[0]]
    def find(c):
        ci = [i for i, h in enumerate(hdr) if c in h]
        return ci[0] if ci else None
    ci, vi = find(col), find(valid_col)
    if ci is None: raise SystemExit(f"column containing '{col}' not found in {hdr}")
    radix = rows[1][ci].strip().upper() if rows[1] and rows[1][0].lower().startswith('radix') else 'SIGNED'
    vals = []
    for r in rows[1:]:
        if len(r) <= ci or not r[ci].strip() or r[0].lower().startswith('radix'): continue
        if vi is not None and r[vi].strip() not in ('1', '01'): continue
        v = r[ci].strip()
        if radix == 'HEX' or (radix != 'SIGNED' and radix != 'UNSIGNED' and any(c in v.lower() for c in 'abcdef')):
            v = int(v, 16)
            if v >= (1 << 17): v -= (1 << 18)
        else:
            v = int(v)
            if radix == 'UNSIGNED' and v >= (1 << 17): v -= (1 << 18)
        vals.append(v)
    return np.array(vals, dtype=np.int64), radix

def frames_of(x, n=320, hop=160):
    T = max((len(x) - n) // hop + 1, 1)
    return np.stack([x[t * hop: t * hop + n] for t in range(T)])

def band_powers(x, n=320, hop=160):
    """per-frame band power (full scale = 1.0) in BANDS, Hann window, rfft 512."""
    F = frames_of(x, n, hop).astype(np.float64) / FS24
    w = np.hanning(n); S = np.abs(np.fft.rfft(F * w, 512)) ** 2 / (np.sum(w ** 2) * SR)
    f = np.fft.rfftfreq(512, 1 / SR)
    return np.stack([S[:, (f >= lo) & (f < hi)].sum(1) * (SR / 512) for lo, hi in BANDS], 1)   # power per band

def welch(x, nperseg=512):
    from scipy.signal import welch as _w
    f, p = _w(x.astype(np.float64) / FS24, fs=SR, nperseg=nperseg, noverlap=nperseg // 2, window='hann')
    return f, p

def level_stats(pcm):
    peak = np.abs(pcm).max(); rms = np.sqrt(np.mean(pcm.astype(np.float64) ** 2))
    x18 = pcm >> 6
    fr = frames_of(pcm, 320, 160).astype(np.float64)
    frms = np.sqrt((fr ** 2).mean(1)); o = np.sort(frms)
    nq = max(len(o) // 10, 1); floor = np.sqrt(np.mean(o[:nq] ** 2)); loud = np.sqrt(np.mean(o[-max(len(o) // 5, 1):] ** 2))
    return dict(peak18=int(np.abs(x18).max()), rms18=float(np.sqrt(np.mean((x18.astype(np.float64)) ** 2))),
                peak_dbfs18=db(np.abs(x18).max(), FS18), rms_dbfs18=db(np.sqrt(np.mean(x18.astype(np.float64) ** 2)), FS18),
                peak_dbfs24=db(peak, FS24), rms_dbfs24=db(rms, FS24), dc=float(pcm.mean()),
                floor_dbfs=db(floor, FS24), loud_dbfs=db(loud, FS24), snr_db=db(loud, FS24) - db(floor, FS24))

def s_vowel(pcm):
    """frame-level: max 4-8 kHz band level (dBFS), max 0-1 kHz band level, and the ratio (dB)."""
    bp = band_powers(pcm); hi = bp[:, 3]; lo = bp[:, 0]
    return dict(s_max_dbfs=10 * np.log10(max(hi.max(), 1e-18)), vowel_max_dbfs=10 * np.log10(max(lo.max(), 1e-18)),
                s_minus_vowel_db=10 * np.log10(max(hi.max(), 1e-18)) - 10 * np.log10(max(lo.max(), 1e-18)),
                frac=[float(v) for v in bp.sum(0) / max(bp.sum(), 1e-18)])

def ref_clips(word, n, seed=0):
    fl = json.load(open(os.path.join(ROOT, 'snn', 'data', 'filelist_c12.json')))['test']
    y = KEYWORDS.index(word); rels = [r for r, l in fl if l == y and not r.startswith('_background')]
    rng = np.random.default_rng(seed); rels = [rels[i] for i in rng.permutation(len(rels))[:n]]
    return [wav_float_to_pcm24(load_item(r)) for r in rels]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', nargs='+'); ap.add_argument('--col', default='dbg_cap_sample')
    ap.add_argument('--run', default=os.path.join(ROOT, 'snn', 'runs', 'M3k3e60_U2_q8_kd_s0'))
    ap.add_argument('--data', default=os.path.join(ROOT, 'snn', 'data', 'feat_c12_m40_w480_U2.npz'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'snn', 'runs', 'board_capture'))
    ap.add_argument('--fig', default=os.path.join(ROOT, 'snn', 'docs', 'figures'))
    ap.add_argument('--ref_n', type=int, default=200)
    ap.add_argument('--pad_lead', type=float, default=0.2); ap.add_argument('--pad_trail', type=float, default=1.0)
    ap.add_argument('--margins', default='0,2838,6212,9633')
    a = ap.parse_args()
    files = sorted(sum([glob.glob(p) for p in a.csv], []))
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.fig, exist_ok=True)
    wav_dir = os.path.join(os.path.dirname(files[0]), 'wav'); os.makedirs(wav_dir, exist_ok=True)
    import soundfile as sf, torch; torch.backends.cuda.matmul.allow_tf32 = False
    from snn.golden.export_snn_mem import load_run, extract
    model, cfg, res = load_run(a.run); P = extract(model)
    z = np.load(a.data); Xt = z['X_test']; classes = json.loads(str(z['meta']))['classes']
    fe = FrontendInt(); margins = [int(m) for m in a.margins.split(',')]
    lead, trail = int(a.pad_lead * SR), int(a.pad_trail * SR)
    out = dict(run=a.run, params=PRM, pad=dict(lead_s=a.pad_lead, trail_s=a.pad_trail), files={}, groups={}, ref={})
    psd = {}
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]; grp = re.sub(r'_\d+$', '', name)
        x18, radix = read_capture(path, a.col); pcm = x18 << 6
        sf.write(os.path.join(wav_dir, name + '.wav'), pcm.astype(np.float64) / FS24, SR, subtype='PCM_24')
        r = dict(group=grp, n=int(len(pcm)), dur_s=len(pcm) / SR, radix=radix)
        r.update(level_stats(pcm)); r.update(s_vowel(pcm))
        f, p = welch(pcm); psd.setdefault(grp, []).append(p)
        # features / energy on the padded signal (same padding as the golden)
        xp = np.concatenate([np.zeros(lead, np.int64), pcm, np.zeros(trail, np.int64)])
        d = fe.run_debug(xp[None]); feat = d['feat'][0]; E = feat.astype(np.int64).sum(-1)
        r['feat_zero_ratio'] = float((feat == 0).mean()); r['feat_mean'] = float(feat.mean()); r['feat_max'] = int(feat.max())
        r['E1'] = [int(v) for v in E]; r['E1_max'] = int(E.max()); r['E1_frames_above_Ton'] = int((E > PRM['T_on']).sum())
        run = 0; best = 0
        for v in E: run = run + 1 if v > PRM['T_on'] else 0; best = max(best, run)
        r['E1_longest_run_above_Ton'] = best; r['E1_max_over_Ton'] = float(E.max() / PRM['T_on'])
        # integer pipeline golden (VAD + pre-roll + SNN + decision), per margin
        g = run_pipeline(P, xp, PRM, 'E1', 0)
        r['golden'] = dict(vad_triggered=int(any(rr['seg_start'] for rr in g['rec'])), snn_frames=int(g['snn_frames']),
                           dets=[dict(t=dt['t'], cls=dt['cls'], name=classes[dt['cls']], margin=dt['margin'],
                                      valid={m: int(dt['cls'] < 10 and dt['margin'] >= m) for m in margins}) for dt in g['dets']])
        # clip mode: 98-frame window with the highest energy, like the dataset clips
        T = feat.shape[0]; best_s, best_e = 0, -1
        for s in range(0, max(T - 98, 0) + 1, 7):
            e = E[s:s + 98].sum()
            if e > best_e: best_e, best_s = e, s
        gi = run_int(P, feat[None, best_s:best_s + 98], trace=False); v = gi['vout'][0].sum(0); o = np.argsort(-v)
        r['clip'] = dict(start_frame=best_s, pred=classes[o[0]], second=classes[o[1]], margin=int(v[o[0]] - v[o[1]]), sums=[int(t) for t in v])
        out['files'][name] = r
        print(f"{name:14s} {r['dur_s']:.2f}s peak18 {r['peak18']:6d} peak {r['peak_dbfs24']:6.1f} dBFS rms {r['rms_dbfs24']:6.1f} floor {r['floor_dbfs']:6.1f} snr {r['snr_db']:5.1f} dB | "
              f"s-vowel {r['s_minus_vowel_db']:6.1f} dB | E1 max {r['E1_max']:5d} run>{PRM['T_on']} {best:3d} | golden "
              + (', '.join(f"{dt['name']}(m{dt['margin']})" for dt in r['golden']['dets']) if r['golden']['dets'] else ('VAD only' if r['golden']['vad_triggered'] else 'no VAD'))
              + f" | clip {r['clip']['pred']} ({r['clip']['margin']})")
    # ---- groups and references
    words = sorted({g.split('_')[0] for g in psd if g.split('_')[0] in KEYWORDS})
    ref = {}
    for w in words:
        clips = ref_clips(w, a.ref_n)
        rs = [level_stats(c) for c in clips]; sv = [s_vowel(c) for c in clips]; ps = [welch(c)[1] for c in clips]
        d_ = fe.run_debug(np.stack(clips)); E = d_['feat'].astype(np.int64).sum(-1)
        ref[w] = dict(n=len(clips), peak_dbfs24=float(np.mean([r['peak_dbfs24'] for r in rs])), rms_dbfs24=float(np.mean([r['rms_dbfs24'] for r in rs])),
                      floor_dbfs=float(np.mean([r['floor_dbfs'] for r in rs])), snr_db=float(np.mean([r['snr_db'] for r in rs])),
                      s_minus_vowel_db=float(np.median([s['s_minus_vowel_db'] for s in sv])), s_minus_vowel_p25=float(np.percentile([s['s_minus_vowel_db'] for s in sv], 25)),
                      s_minus_vowel_p75=float(np.percentile([s['s_minus_vowel_db'] for s in sv], 75)),
                      frac=[float(v) for v in np.mean([s['frac'] for s in sv], 0)], E1_max_median=float(np.median(E.max(1))), E1_max_p10=float(np.percentile(E.max(1), 10)),
                      psd=np.mean(ps, 0))
    out['ref'] = {w: {k: v for k, v in r.items() if k != 'psd'} for w, r in ref.items()}
    out['test_set'] = dict(zero_ratio=float((Xt == 0).mean()), mean=float(Xt.mean()))
    for grp, lst in psd.items():
        fs_ = [out['files'][n] for n in out['files'] if out['files'][n]['group'] == grp]
        out['groups'][grp] = dict(n=len(fs_), **{k: float(np.mean([f_[k] for f_ in fs_])) for k in ('peak_dbfs24', 'rms_dbfs24', 'floor_dbfs', 'snr_db', 's_minus_vowel_db', 'E1_max')},
                                  frac=[float(v) for v in np.mean([f_['frac'] for f_ in fs_], 0)])
    # channel response estimate: board PSD (all 'stop' captures, speech-only is approximated by the whole clip) / reference PSD
    chan = {}
    for grp, lst in psd.items():
        w = grp.split('_')[0]
        if w in ref:
            h = 10 * np.log10(np.mean(lst, 0) + 1e-20) - 10 * np.log10(ref[w]['psd'] + 1e-20)
            chan[grp] = h
    f = welch(np.zeros(16000, np.int64) + 1)[0]
    if chan:
        np.savez(os.path.join(a.out, 'channel_response.npz'), f=f, **{k: v for k, v in chan.items()})
        out['channel_db_at'] = {grp: {str(int(fq)): float(np.interp(fq, f, h)) for fq in (250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000)} for grp, h in chan.items()}
    json.dump(out, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    # ---- figures
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    groups = sorted(psd)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, grp in enumerate([g for g in groups if g.startswith('stop')] + [g for g in groups if g == 'quiet']):
        ax[0].plot(f / 1000, 10 * np.log10(np.mean(psd[grp], 0) + 1e-20), color=PAL[i % len(PAL)], lw=2, label=f"board {grp} (n={len(psd[grp])})")
    if 'stop' in ref: ax[0].plot(f / 1000, 10 * np.log10(ref['stop']['psd'] + 1e-20), color='#444', lw=2, ls='--', label=f"Speech Commands test 'stop' (n={ref['stop']['n']})")
    ax[0].set_xlabel('kHz'); ax[0].set_ylabel('PSD (dBFS/Hz)'); ax[0].set_title('Average PSD: board captures vs dataset'); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    for i, grp in enumerate([g for g in groups if g.startswith('stop')]):
        ax[1].plot(f / 1000, chan[grp], color=PAL[i % len(PAL)], lw=2, label=f"{grp} / dataset")
    ax[1].axhline(0, color='#888', lw=1); ax[1].set_xlabel('kHz'); ax[1].set_ylabel('dB'); ax[1].set_title("Board / dataset PSD ratio ('stop')"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.fig, 'board_psd.png'), dpi=150); plt.close(fig)
    fig, axs = plt.subplots(2, 4, figsize=(13, 5.5), sharey=True); axs = axs.ravel()
    for i, grp in enumerate(groups[:8]):
        for j, n in enumerate([n for n in out['files'] if out['files'][n]['group'] == grp]):
            E = out['files'][n]['E1']; axs[i].plot(np.arange(len(E)) * 0.01, E, color=PAL[j % len(PAL)], lw=1.5, label=n)
        axs[i].axhline(PRM['T_on'], color='#444', ls='--', lw=1); axs[i].axhline(PRM['T_off'], color='#999', ls=':', lw=1)
        axs[i].set_title(grp); axs[i].legend(fontsize=6); axs[i].set_xlabel('s'); axs[i].grid(alpha=.3)
    axs[0].set_ylabel('E1 (sum of 40 features)'); fig.suptitle(f"VAD energy E1 (T_on {PRM['T_on']} dashed, T_off {PRM['T_off']} dotted); {a.pad_lead} s zero lead", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.fig, 'board_e1.png'), dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    gs = [g for g in groups if g.startswith('stop')]; xs = np.arange(len(gs) + 1)
    vals = [out['groups'][g]['s_minus_vowel_db'] for g in gs] + [ref['stop']['s_minus_vowel_db'] if 'stop' in ref else 0]
    ax.bar(xs, vals, color=[PAL[0]] * len(gs) + ['#888'], width=0.6)
    for x_, v in zip(xs, vals): ax.text(x_, v + (0.5 if v >= 0 else -2.5), f"{v:.1f}", ha='center', fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(gs + ['dataset stop\n(median)']); ax.set_ylabel('max 4-8 kHz frame - max 0-1 kHz frame (dB)')
    ax.set_title("/s/ band vs vowel band level, 'stop'"); ax.grid(alpha=.3, axis='y')
    fig.tight_layout(); fig.savefig(os.path.join(a.fig, 'board_sband.png'), dpi=150); plt.close(fig)
    # ---- markdown tables
    L = []
    L.append(f"## 캡처별 레벨·골든 (run {os.path.basename(a.run)}, 패딩 앞 {a.pad_lead} s / 뒤 {a.pad_trail} s 디지털 0, VAD {PRM})\n")
    L.append("| 캡처 | 길이 s | peak(18b 카운트) | peak dBFS | RMS dBFS | noise floor dBFS | SNR dB | /s/−모음 dB | 특징 0 비율 | E1 max | E1>T_on 연속 | 골든 판정(마진0) | 마진 | 클립 예측 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n, r in out['files'].items():
        dets = r['golden']['dets']
        gd = ', '.join(dt['name'] for dt in dets) if dets else ('VAD만(판정 없음)' if r['golden']['vad_triggered'] else 'VAD 미발동')
        mg = ', '.join(str(dt['margin']) for dt in dets) if dets else '-'
        L.append(f"| {n} | {r['dur_s']:.2f} | {r['peak18']} | {r['peak_dbfs24']:.1f} | {r['rms_dbfs24']:.1f} | {r['floor_dbfs']:.1f} | {r['snr_db']:.1f} | {r['s_minus_vowel_db']:.1f} | {r['feat_zero_ratio']:.2f} | {r['E1_max']} | {r['E1_longest_run_above_Ton']} | {gd} | {mg} | {r['clip']['pred']} ({r['clip']['margin']}) |")
    L.append("\n(18-bit와 24-bit 풀스케일 기준 dBFS는 같다: cap_sample = pcm[23:6]이므로 비율이 동일. peak 카운트는 18-bit 단위.)\n")
    L.append("## 그룹 평균 vs Speech Commands test 같은 단어\n")
    L.append("| 그룹 | n | peak dBFS | RMS dBFS | floor dBFS | SNR dB | /s/−모음 dB | 0–1k % | 1–2k % | 2–4k % | 4–8k % | E1 max 평균 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for g, r in out['groups'].items():
        fr = r['frac']; L.append(f"| board {g} | {r['n']} | {r['peak_dbfs24']:.1f} | {r['rms_dbfs24']:.1f} | {r['floor_dbfs']:.1f} | {r['snr_db']:.1f} | {r['s_minus_vowel_db']:.1f} | {100*fr[0]:.1f} | {100*fr[1]:.1f} | {100*fr[2]:.1f} | {100*fr[3]:.1f} | {r['E1_max']:.0f} |")
    for w, r in out['ref'].items():
        fr = r['frac']; L.append(f"| dataset test '{w}' | {r['n']} | {r['peak_dbfs24']:.1f} | {r['rms_dbfs24']:.1f} | {r['floor_dbfs']:.1f} | {r['snr_db']:.1f} | {r['s_minus_vowel_db']:.1f} (p25 {r['s_minus_vowel_p25']:.1f}, p75 {r['s_minus_vowel_p75']:.1f}) | {100*fr[0]:.1f} | {100*fr[1]:.1f} | {100*fr[2]:.1f} | {100*fr[3]:.1f} | {r['E1_max_median']:.0f} (p10 {r['E1_max_p10']:.0f}) |")
    if chan:
        L.append("\n## 보드/데이터셋 PSD 비(채널 응답 추정, dB; 레벨 차이 포함)\n")
        L.append("| 그룹 | " + ' | '.join(f"{k} Hz" for k in out['channel_db_at'][next(iter(chan))]) + " |"); L.append("|---|" + "---|" * 9)
        for g, h in out['channel_db_at'].items(): L.append(f"| {g} | " + ' | '.join(f"{v:.1f}" for v in h.values()) + " |")
    L.append(f"\n test set 특징: 0 비율 {out['test_set']['zero_ratio']:.3f}, 평균 {out['test_set']['mean']:.2f}\n")
    open(os.path.join(a.out, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))

if __name__ == '__main__':
    main()
