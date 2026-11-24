"""
make_fig12_gating.py — fig12: board ILA capture of the BUFGCE clock enables around a VAD segment start (gating ON vs OFF).
Source: snn/board/log/gate_on.csv / gate_off.csv (ILA depth 16384, trigger dbg_vad_active rising, position 2000, yes clip).
All dbg_* probes are registered once in user_logic (same delay), so the relative timing is cycle-exact.
usage: python snn/docs/make_fig12_gating.py [--out snn/docs/figures/final/fig12_ila_gating_waveform.png]
"""
import os, csv, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); LOG = os.path.join(ROOT, 'snn', 'board', 'log')
plt.rcParams.update({'font.family': ['Noto Sans CJK KR', 'DejaVu Sans'], 'savefig.dpi': 300, 'axes.unicode_minus': False, 'font.size': 8})
INK, INK2, BG = '#1b1b1b', '#5b5b5b', '#ffffff'; C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']
SIG = [('dbg_fe_frame_done', 'fe_frame_done', INK2), ('dbg_vad_active', 'vad_active', C[3]), ('dbg_snn_active', 'snn_active', C[4]), ('dbg_fe_ce', 'FE 클록 CE', C[0]), ('dbg_snn_ce', 'SNN 클록 CE', C[1])]

def load(name):
    rows = list(csv.reader(open(os.path.join(LOG, name)))); h = [x.strip() for x in rows[0]]
    col = {k: [i for i, x in enumerate(h) if k in x][0] for k, _, _ in SIG}; col['trig'] = h.index('TRIGGER')
    data = rows[2:]; n = np.array([int(r[0]) for r in data])
    return n, {k: np.array([int(r[c], 16) if k != 'trig' else int(r[c]) for r in data]) for k, c in col.items()}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(ROOT, 'snn', 'docs', 'figures', 'final', 'fig12_ila_gating_waveform.png')); ap.add_argument('--lo', type=int, default=1990); ap.add_argument('--hi', type=int, default=2010)
    a = ap.parse_args()
    fig, AX = plt.subplots(2, 2, figsize=(11, 6.4), gridspec_kw=dict(width_ratios=[3.1, 1], wspace=0.08, hspace=0.3)); axs = AX[:, 0]
    for ax, ins, (fname, title) in zip(axs, AX[:, 1], (('gate_on.csv', '게이팅 ON (sw[5] = OFF)'), ('gate_off.csv', '게이팅 OFF (sw[5] = ON, CE 강제 1)'))):
        n, s = load(fname); sel = (n >= a.lo) & (n <= a.hi); x = n[sel]
        for i, (k, lab, col) in enumerate(SIG):
            y0 = (len(SIG) - 1 - i) * 1.5; v = s[k][sel]
            ax.step(x, y0 + v, where='post', color=col, lw=1.6); ax.fill_between(x, y0, y0 + v, step='post', color=col, alpha=0.12)
            ax.text(a.lo - 0.4, y0 + 0.5, lab, ha='right', va='center', fontsize=8, color=INK)
            tr = [int(x[j]) for j in range(1, len(v)) if v[j] != v[j - 1]]
            for t in tr: ax.annotate(f"{t}", (t, y0 + 1.08), fontsize=6.5, color=col, ha='center')
        t_trig = int(n[s['trig'] == 1][0]); ax.axvline(t_trig, color=INK2, lw=0.7, ls='--'); ax.text(t_trig + 0.15, len(SIG) * 1.5 + 0.2, f'트리거 (vad_active 상승) @{t_trig}', fontsize=7, color=INK2)
        ax.set_ylim(-0.3, len(SIG) * 1.5 + 0.45); ax.set_yticks([]); ax.set_title(title, fontsize=9.5, loc='left'); ax.grid(axis='x', color='#e6e6e6', lw=0.5)
        for sp in ('top', 'right', 'left'): ax.spines[sp].set_visible(False)
        # inset: full window, CE signals only
        ins.step(n, s['dbg_fe_ce'] + 1.2, where='post', color=C[0], lw=0.8); ins.step(n, s['dbg_snn_ce'], where='post', color=C[1], lw=0.8)
        ins.axvline(t_trig, color=INK2, lw=0.5, ls='--'); ins.set_xlim(n[0], n[-1]); ins.set_ylim(-0.2, 2.5); ins.set_yticks([0.5, 1.7]); ins.set_yticklabels(['SNN CE', 'FE CE'], fontsize=6)
        ins.tick_params(axis='x', labelsize=6); ins.set_xlabel(f'전체 창 0~{n[-1]} 사이클 ({n[-1] / 48e3:.2f} ms @ 48 MHz)', fontsize=6); ins.set_facecolor('#fafafa'); ins.set_title(f"전체 창 (CE만) — 듀티 FE {100 * s['dbg_fe_ce'].mean():.0f} %, SNN {100 * s['dbg_snn_ce'].mean():.0f} %", fontsize=7, loc='left')
    axs[0].tick_params(labelbottom=False); axs[-1].set_xlabel(f'ILA 샘플(클록 사이클, 48 MHz) — 확대 구간 {a.lo}~{a.hi}'); [x.set_xticks(range(a.lo, a.hi + 1, 2)) for x in axs]
    fig.suptitle('그림 12. 보드 ILA 캡처: VAD 구간 시작에서의 FE/SNN 클록 CE 핸드오프 (yes 클립 재생, 트리거 vad_active 상승, 깊이 16384, 위치 2000)', fontsize=9.5)
    fig.text(0.01, 0.005, "fe_frame_done(1999) → VAD 출력·vad_active(2000) → 시퀀서 S_WAKE·snn_req(2001) → SNN CE(2002): 레지스터 3단 = 3사이클(= 0.06 µs, 프레임 10 ms 대비 무시 가능). FE CE는 프레임 처리 종료(req 하강 1999) 뒤 OFF_DELAY 2 → 2002에 내려간다.\n"
             "두 CE는 2002에 같은 사이클에 교대(겹침·공백 0사이클)하지만 서로 독립된 제어기의 결과다(설계된 핸드셰이크가 아님). 게이팅 OFF에서는 force_on으로 CE가 상시 1이고 나머지 신호의 타이밍은 동일하다.", fontsize=6.6, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96)); fig.savefig(a.out); print('->', a.out)

if __name__ == '__main__':
    main()
