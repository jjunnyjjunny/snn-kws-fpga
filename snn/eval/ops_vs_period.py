"""ops_vs_period.py — A: operations per second vs decision period (10 ms / 100 ms / 1 s) for DS-CNN (window
recomputed every decision) and the SNN (frame-incremental: independent of the decision period).
Writes snn/runs/figures/ops_vs_period.{png,csv}."""
import os, csv, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
F = os.path.join(ROOT, 'snn', 'runs', 'figures')
rows = {r['model']: r for r in csv.DictReader(open(os.path.join(F, 'ops_energy.csv')))}
periods = [0.01, 0.1, 1.0]
series = [('DS-CNN 24ch (log-mel)', lambda p: float(rows['DS-CNN 24ch (log-mel 98x40)']['mac_per_inf']) / p, '#2a78d6'),
          ('DS-CNN 48ch (log-mel)', lambda p: float(rows['DS-CNN 48ch (log-mel 98x40)']['mac_per_inf']) / p, '#6250d6'),
          ('hls4ml MFCC DS-CNN 24ch (board, 10cls)', lambda p: float(rows['DS-CNN 24ch hls4ml (MFCC 49x10, week 0)']['mac_per_inf']) / p, '#e34948'),
          ('SNN M3k3 Q8+KD, dense input', lambda p: float(rows['M3k3 Q8+KD (60ep, recommended) (dense input)']['ops_per_s_stream']), '#eb6834'),
          ('SNN M3k3 Q8+KD, input zero-skip', lambda p: float(rows['M3k3 Q8+KD (60ep, recommended) (zero-skip)']['ops_per_s_stream']), '#1baf7a')]
out = []
for name, f, _ in series:
    for p in periods: out.append(dict(model=name, decision_period_s=p, ops_per_s=f(p)))
with open(os.path.join(F, 'ops_vs_period.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['model', 'decision_period_s', 'ops_per_s']); w.writeheader(); w.writerows(out)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150); fig.patch.set_facecolor('#fcfcfb'); ax.set_facecolor('#fcfcfb')
for name, f, col in series:
    ys = [f(p) for p in periods]
    ax.plot(periods, ys, marker='o', ms=6, lw=2, color=col, label=name, markeredgecolor='#fcfcfb', markeredgewidth=1.2)
    ax.annotate(f"{ys[1]/1e6:.2g} M/s" if ys[1] >= 1e6 else f"{ys[1]/1e3:.0f} k/s", (periods[1], ys[1]), textcoords='offset points', xytext=(8, -4), fontsize=7.5, color='#52514e')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks(periods); ax.set_xticklabels(['10 ms', '100 ms', '1 s'])
ax.set_xlabel('decision period', color='#0b0b0b'); ax.set_ylabel('operations per second (MAC + AC)', color='#0b0b0b')
ax.set_title('Streaming compute vs decision period (SNN cost is per frame, independent of the period)', fontsize=9.5, color='#0b0b0b')
ax.grid(True, which='major', color='#e6e5e0', lw=0.8); ax.grid(False, which='minor')
for s in ('top', 'right'): ax.spines[s].set_visible(False)
for s in ('left', 'bottom'): ax.spines[s].set_color('#c3c2b7')
ax.tick_params(colors='#52514e'); ax.legend(frameon=False, fontsize=7.5, loc='upper right')
fig.tight_layout(); fig.savefig(os.path.join(F, 'ops_vs_period.png'))
d24 = float(rows['DS-CNN 24ch (log-mel 98x40)']['mac_per_inf']); zs = float(rows['M3k3 Q8+KD (60ep, recommended) (zero-skip)']['ops_per_s_stream'])
print({p: round(d24 / p / zs, 1) for p in periods}, 'x ratio DS-CNN24 / SNN zero-skip at each period')
