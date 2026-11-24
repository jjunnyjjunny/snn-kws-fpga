"""make_figures.py — D-2 figures for the final report (PNG + source data) in snn/docs/figures/:
 1 system_block.svg (hand-written below)   2 ops_vs_period.png (copied from runs/figures)   3 timeline.png
 4 confusion.png  5 activity_share.png (from runs/activity_*.txt, if present)  6 seed_stats.png"""
import os, sys, json, re, shutil, csv, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); S = os.path.join(ROOT, 'snn'); F = os.path.join(S, 'docs', 'figures'); os.makedirs(F, exist_ok=True)
sys.path.insert(0, ROOT)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
BG, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e0'; C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
def style(ax):
    ax.set_facecolor(BG); ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color('#c3c2b7')
    ax.tick_params(colors=INK2)
def R(n):
    p = os.path.join(S, 'runs', n, 'results.json'); return json.load(open(p)) if os.path.exists(p) else None
# ---- 2. copy
for f in ('ops_vs_period.png', 'ops_vs_period.csv', 'ops_vs_acc.png', 'ops_energy.csv'):
    if os.path.exists(os.path.join(S, 'runs', 'figures', f)): shutil.copy(os.path.join(S, 'runs', 'figures', f), os.path.join(F, f))
# ---- 6. seed statistics bar chart
groups = [('DS-CNN 24ch 12cls', ['dscnn_c12_e100', 'dscnn_c12_s1', 'dscnn_c12_s2']), ('SNN M3k3 Q8+KD 12cls', ['M3k3e60_c12_q8_kd', 'M3k3e60_c12_q8_kd_s1', 'M3k3e60_c12_q8_kd_s2']),
          ('DS-CNN 24ch 10cls', ['dscnn_c10_e100', 'dscnn_c10_s1', 'dscnn_c10_s2']), ('SNN M3k3 Q8+KD 10cls', ['M3k3e60_c10_q8_kd_s0', 'M3k3e60_c10_q8_kd_s1', 'M3k3e60_c10_q8_kd_s2'])]
rows = []
for name, runs in groups:
    accs = [(r['test_acc']['mean'] if isinstance(r['test_acc'], dict) else r['test_acc']) for r in map(R, runs) if r]
    rows.append(dict(group=name, mean=100 * np.mean(accs), std=100 * np.std(accs), n=len(accs), accs=[100 * a for a in accs]))
with open(os.path.join(F, 'seed_stats.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['group', 'mean', 'std', 'n', 'seed_accs']); [w.writerow([r['group'], f"{r['mean']:.2f}", f"{r['std']:.2f}", r['n'], ' '.join(f'{a:.2f}' for a in r['accs'])]) for r in rows]
fig, ax = plt.subplots(figsize=(7, 4), dpi=150); fig.patch.set_facecolor(BG); style(ax)
x = np.arange(len(rows)); cols = [C[0], C[1], C[0], C[1]]
ax.bar(x, [r['mean'] for r in rows], yerr=[r['std'] for r in rows], color=cols, width=0.6, capsize=4, zorder=3, edgecolor=BG)
for i, r in enumerate(rows): ax.text(i, r['mean'] + r['std'] + 0.15, f"{r['mean']:.2f} ± {r['std']:.2f}", ha='center', fontsize=8, color=INK2)
ax.set_xticks(x); ax.set_xticklabels([r['group'] for r in rows], fontsize=8); ax.set_ylim(85, 93.5); ax.set_ylabel('test accuracy (%)', color=INK)
ax.set_title('Accuracy over 3 seeds (mean ± std)', fontsize=10, color=INK); fig.tight_layout(); fig.savefig(os.path.join(F, 'seed_stats.png'))
# ---- 4. confusion matrices
cf = json.load(open(os.path.join(S, 'runs', 'confusion.json'))); cl = cf['classes']
fig, axs = plt.subplots(1, 2, figsize=(11, 5), dpi=150); fig.patch.set_facecolor(BG)
for ax, key, title in zip(axs, ('snn', 'dscnn'), ('SNN M3k3 Q8+KD (integer golden)', 'DS-CNN 24ch')):
    m = np.array(cf[key]['cm'], dtype=float); mn = m / m.sum(1, keepdims=True)
    im = ax.imshow(mn, cmap='Blues', vmin=0, vmax=1); ax.set_xticks(range(len(cl))); ax.set_yticks(range(len(cl))); ax.set_xticklabels(cl, rotation=60, fontsize=7); ax.set_yticklabels(cl, fontsize=7)
    for i in range(len(cl)):
        for j in range(len(cl)):
            if m[i, j] > 0 and (i == j or m[i, j] >= 10): ax.text(j, i, int(m[i, j]), ha='center', va='center', fontsize=6, color='white' if mn[i, j] > 0.5 else INK)
    ax.set_title(f"{title} — acc {100*cf[key]['acc']:.1f} %", fontsize=9, color=INK); ax.set_xlabel('predicted', color=INK2); ax.set_ylabel('true', color=INK2)
fig.tight_layout(); fig.savefig(os.path.join(F, 'confusion.png'))
with open(os.path.join(F, 'confusion.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['model', 'true'] + cl); [w.writerow([k, cl[i]] + cf[k]['cm'][i]) for k in ('snn', 'dscnn') for i in range(len(cl))]
# ---- 3. timeline on the activity_board stream (golden pipeline + gating flag)
from snn.golden.pipeline_int_ref import run_pipeline
from snn.frontend.frontend_ref import FrontendInt, scale_input, frame
import torch; torch.backends.cuda.matmul.allow_tf32 = False
from snn.golden.export_snn_mem import load_run, extract
model, cfg, res = load_run(os.path.join(S, 'runs', 'M3k3e60_c12_q8_kd')); P = extract(model)
vd = os.path.join(S, 'golden', 'vectors', 'activity_board', 'stream_0'); info = json.load(open(os.path.join(vd, 'info.json')))
pcm = np.array([int(l, 16) - (1 << 24 if int(l, 16) >= (1 << 23) else 0) for l in open(os.path.join(vd, 'pcm24.hex')) if not l.startswith('//')], dtype=np.int64)
r = run_pipeline(P, pcm, info['params'], 'E1'); T = r['T']; t0 = json.load(open(os.path.join(S, 'runs', 't0_bound.json')))['T0_proven']
E0 = np.abs(frame(scale_input(pcm))).sum(-1); gated = E0 < t0
tt = np.arange(T) * 0.01; secs = min(12.0, T * 0.01)
fig, axs = plt.subplots(5, 1, figsize=(10, 7.5), dpi=150, sharex=True); fig.patch.set_facecolor(BG)
axs[0].plot(np.arange(len(pcm)) / 16000, pcm / 2 ** 23, color=C[0], lw=0.4); axs[0].set_ylabel('waveform', fontsize=8)
for e in info['events']: axs[0].axvspan(e['onset'] * 0.01, e['offset'] * 0.01, color=C[3] if e['label'] < 10 else C[7], alpha=0.15)
axs[1].plot(tt, r['E'], color=C[1], lw=1); axs[1].axhline(info['params']['T_on'], color=INK2, lw=0.6, ls='--'); axs[1].axhline(info['params']['T_off'], color=INK2, lw=0.6, ls=':'); axs[1].set_ylabel('E1 = Σf', fontsize=8)
act = np.array([x['active'] for x in r['rec']]); fed = np.array([x['fed'] > 0 for x in r['rec']])
axs[2].fill_between(tt, 0, act, step='post', color=C[2], alpha=0.6); axs[2].set_ylabel('VAD', fontsize=8); axs[2].set_yticks([0, 1])
axs[3].fill_between(tt, 0, fed, step='post', color=C[6], alpha=0.6); axs[3].fill_between(tt, 0, gated * 0.5, step='post', color=C[4], alpha=0.6); axs[3].set_ylabel('SNN on (violet)\nFE gated (pink)', fontsize=8); axs[3].set_yticks([0, 1])
for d in r['dets']:
    if d['t'] * 0.01 > secs: continue
    axs[4].axvline(d['t'] * 0.01, color=C[2] if d['valid'] else C[7], lw=1.5); axs[4].text(d['t'] * 0.01 + 0.08, 0.5, cl[d['cls']], fontsize=8, color=INK, rotation=90, va='center')
axs[4].set_yticks([]); axs[4].set_ylabel('decision', fontsize=8); axs[4].set_xlabel('time (s)', color=INK); axs[4].set_xlim(0, secs)
for ax in axs: style(ax)
fig.suptitle('Stream example (board condition): waveform, energy, VAD, SNN activity / front-end gating, decisions', fontsize=10, color=INK); fig.tight_layout(); fig.savefig(os.path.join(F, 'timeline.png'))
json.dump(dict(T=T, E=[int(v) for v in r['E']], active=act.tolist(), fed=fed.astype(int).tolist(), gated=gated.astype(int).tolist(), dets=r['dets'], events=info['events']), open(os.path.join(F, 'timeline_data.json'), 'w'))
# ---- 5. activity share (if measured)
act_rows = []
for fn in ('activity_board_talk.txt', 'activity_quiet.txt'):
    p = os.path.join(S, 'runs', fn)
    if not os.path.exists(p): continue
    cur = None
    for l in open(p):
        if l.startswith('=='): cur = l.strip('= \n')
        elif l.startswith('ACTIVITY'): d = dict(re.findall(r'(\w+)=(\S+)', l)); d['cfg'] = cur; act_rows.append(d)
        elif l.startswith('ACT ') and act_rows: act_rows[-1].update(dict(re.findall(r'(\w+)=(\S+)', l)))
if act_rows:
    with open(os.path.join(F, 'activity.csv'), 'w', newline='') as fh:
        keys = sorted(set(k for r in act_rows for k in r)); w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(act_rows)
    sel = [r for r in act_rows if r['cfg'].endswith('T0=6694 VAD=1')]
    if sel:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150); fig.patch.set_facecolor(BG); style(ax)
        names = [r['cfg'].split()[0] for r in sel]; parts = ['fe_cyc', 'snn_cyc', 'pb_extra', 'dec_cyc']; labels = ['front-end', 'SNN core', 'pre-roll feed (excl. SNN)', 'decision']
        frames = [int(r['frames']) for r in sel]
        for r in sel: r['pb_extra'] = max(int(r.get('pb_cyc', 0)) - int(r.get('snn_cyc', 0)), 0)
        bottom = np.zeros(len(sel))
        for k, (pk, lab) in enumerate(zip(parts, labels)):
            v = np.array([int(r.get(pk, 0)) / fr for r, fr in zip(sel, frames)])   # active cycles per frame (10 ms = 480k cycles)
            ax.bar(names, v, bottom=bottom, color=C[k], label=lab, zorder=3, edgecolor=BG); bottom += v
        for i, r in enumerate(sel): ax.text(i, bottom[i] + 100, f"{100*bottom[i]/480000:.2f} % of 480k", ha='center', fontsize=8, color=INK2)
        ax.set_ylabel('active cycles per 10 ms frame (budget 480,000)', color=INK); ax.set_title('Block activity per frame, measured in simulation (T0 gating on, VAD on)', fontsize=10, color=INK); ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(F, 'activity_share.png'))
        # before/after panel: front-end cycles (gating off vs on) and SNN cycles (VAD off vs on) per scenario
        scen = sorted(set(r['cfg'].split()[0] for r in act_rows), key=lambda x: ['board', 'talk', 'quiet'].index(x) if x in ('board', 'talk', 'quiet') else 9)
        def get(sc, t0, vad, key):
            for r in act_rows:
                if r['cfg'] == f'{sc} T0={t0} VAD={vad}': return int(r.get(key, 0)) / int(r['frames'])
            return float('nan')
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=150); fig.patch.set_facecolor(BG)
        x = np.arange(len(scen)); w = 0.38
        axs[0].bar(x - w/2, [get(sc, 0, 1, 'fe_cyc') for sc in scen], w, color=C[0], alpha=0.45, label='gating off', zorder=3, edgecolor=BG)
        axs[0].bar(x + w/2, [get(sc, 6694, 1, 'fe_cyc') for sc in scen], w, color=C[0], label='gating on (T0 = 6694, lossless)', zorder=3, edgecolor=BG)
        axs[0].set_xticks(x); axs[0].set_xticklabels(scen); axs[0].set_ylabel('front-end active cycles / frame', color=INK); axs[0].set_title('Front-end: 2-stage gating', fontsize=10, color=INK); axs[0].legend(frameon=False, fontsize=8)
        axs[1].bar(x - w/2, [get(sc, 0, 0, 'snn_cyc') for sc in scen], w, color=C[1], alpha=0.45, label='VAD threshold 0 (≈ always on)', zorder=3, edgecolor=BG)
        axs[1].bar(x + w/2, [get(sc, 6694, 1, 'snn_cyc') for sc in scen], w, color=C[1], label='VAD segments only', zorder=3, edgecolor=BG)
        axs[1].set_xticks(x); axs[1].set_xticklabels(scen); axs[1].set_ylabel('SNN active cycles / frame', color=INK); axs[1].set_title('SNN: VAD gating', fontsize=10, color=INK); axs[1].legend(frameon=False, fontsize=8)
        for ax in axs: style(ax)
        fig.tight_layout(); fig.savefig(os.path.join(F, 'activity_gating.png'))
print('figures written:', sorted(os.listdir(F)))
