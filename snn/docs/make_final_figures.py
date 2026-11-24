"""
make_final_figures.py — final-report figures, 300 dpi PNG, in snn/docs/figures/final/ (+ index.md with caption drafts).
Sources: fe_config.py / run config (diagrams), figures/seed_stats.csv, det_curve.csv, confusion.csv, activity.csv,
runs/vivado_impl.md + board/pkg/POWER_COMPARE.md numbers (typed below), golden pipeline on activity_board/stream_0
(VAD example, ROM U2 s0), runs/board_capture*/level_sweep.json + level_confusion.json, data/board_capture/*.csv +
runs/board_capture/summary.json (ILA mode-3 captures).
usage: python snn/docs/make_final_figures.py [--out snn/docs/figures/final] [--skip_vad]
"""
import os, sys, csv, json, argparse
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, ROOT)
S = os.path.join(ROOT, 'snn'); FIG = os.path.join(S, 'docs', 'figures')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family': ['Noto Sans CJK KR', 'DejaVu Sans'], 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300})
BG, INK, INK2, GRID = '#ffffff', '#0b0b0b', '#52514e', '#e6e5e0'
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6', '#008300', '#e34948']
CLASSES = ['yes', 'no', 'stop', 'go', 'up', 'down', 'left', 'right', 'on', 'off', 'silence', 'unknown']
CAP = []   # (file, caption)

def style(ax):
    ax.set_facecolor(BG); ax.grid(True, color=GRID, lw=0.7, zorder=0); ax.tick_params(colors=INK2)
    for s in ('left', 'bottom'): ax.spines[s].set_color('#c3c2b7')

def box(ax, x, y, w, h, lines, fc='#ffffff', ec=C[0], fs=8.5, lw=1.4, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec=ec, lw=lw, zorder=2))
    n = len(lines); dy = h / (n + 1)
    for k, t in enumerate(lines):
        ax.text(x + w / 2, y + h - dy * (k + 1), t, ha='center', va='center', fontsize=fs, color=INK, weight='bold' if (k == 0 and bold_first) else 'normal', zorder=3)

def arrow(ax, x0, y0, x1, y1, color=INK2, lw=1.2, style='-|>'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=12, color=color, lw=lw, zorder=1))

def save(fig, name, caption, out):
    fig.savefig(os.path.join(out, name), dpi=300, facecolor=BG); plt.close(fig); CAP.append((name, caption)); print('saved', name)

# ------------------------------------------------------------------ 1. system block diagram
def fig_system(out):
    fig, ax = plt.subplots(figsize=(11, 4.6)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis('off')
    y, h = 2.0, 1.0
    blocks = [(0.2, 1.1, ['ADAU1761 코덱', 'I2S 16 kHz', '24비트 PCM']), (1.7, 1.75, ['정수 프론트엔드', 'Hann·FFT512(18b)', 'mel40·log2 → 8b 특징']),
              (3.85, 1.4, ['VAD', 'E1 = Σf 히스테리시스', '프리롤 20프레임']), (5.65, 1.9, ['SNN 코어 M3k3', '스파이킹 TCN 64ch×4', 'LIF k=3, 8비트, 이벤트 구동']),
              (7.95, 1.4, ['구간 판정', 'argmax(구간 합)', '마진 sw[3:2]']), (9.75, 1.1, ['출력', '비프 / LED', 'ILA det_valid'])]
    for i, (x, w, lines) in enumerate(blocks):
        box(ax, x, y, w, h, lines, ec=C[0] if i not in (0, 5) else INK2)
        if i < len(blocks) - 1: arrow(ax, x + w, y + h / 2, blocks[i + 1][0], y + h / 2)
    # gating annotations
    box(ax, 1.55, 3.45, 2.05, 0.95, ['T0 프론트엔드 게이팅', 'Σ|x18| < 6,694 → 특징 0 증명', '디지털 무음에서만 발동'], fc='#fff7f3', ec=C[1], fs=7.8)
    box(ax, 5.35, 3.45, 2.5, 0.95, ['클록 게이팅 BUFGCE ×2', 'SNN 코어: VAD 구간에서만 클록', 'FE 처리 블록: 프레임 처리 중에만'], fc='#fff7f3', ec=C[1], fs=7.8)
    arrow(ax, 2.575, 3.45, 2.575, y + h, color=C[1]); arrow(ax, 6.6, 3.45, 6.6, y + h, color=C[1])
    box(ax, 3.75, 0.35, 3.7, 0.95, ['가중치·θ·b ROM (.mem, BRAM18K 23)', '파이썬 골든에서 생성, RTL과 비트 일치', 'ROM 교체만으로 모델 교체(U2 ↔ m3k3u2c)'], fc='#f3f8ff', ec=C[0], fs=7.8)
    arrow(ax, 6.6, 1.3, 6.6, y, color=C[0])
    ax.text(0.2, 0.15, 'Zynq-7020 PL, clk_48 단일 클록 · 모든 블록에 비트 일치 파이썬 골든(frontend_ref / snn_int_ref / pipeline_int_ref) · 실제 무음 절감 = VAD + BUFGCE (T0는 보드 잡음 바닥·DC 오프셋에서는 미발동)',
            fontsize=7.5, color=INK2)
    ax.set_title('그림 1. 시스템 블록도 (마이크 → 정수 프론트엔드 → VAD → SNN → 구간 판정)', fontsize=10, loc='left')
    save(fig, 'fig01_system_block.png', '시스템 블록도. ADAU1761 코덱의 16 kHz 24비트 PCM이 정수 프론트엔드(Hann·FFT512·mel40·log2, 8비트 특징)를 거쳐 VAD(E1 히스테리시스, 프리롤 20프레임)가 연 구간에서만 SNN 코어(M3k3, 8비트, 이벤트 구동)와 구간 판정(argmax, 마진 스위치)이 동작한다. 두 BUFGCE가 SNN 코어와 프론트엔드 처리 블록의 클록을 활동 중에만 켠다. T0 프론트엔드 게이팅은 디지털 무음에서만 발동하며 보드 잡음 바닥에서는 걸리지 않는다. 가중치 ROM(.mem)은 파이썬 골든에서 생성되어 RTL과 비트 일치한다.', out)

# ------------------------------------------------------------------ 2. integer front-end flow
def fig_frontend(out):
    from snn.frontend import fe_config as FC
    fig, ax = plt.subplots(figsize=(11, 3.9)); ax.set_xlim(0, 11); ax.set_ylim(0, 3.9); ax.axis('off')
    y, h = 1.55, 1.25
    st = [(['PCM 24b', f'{FC.SR // 1000} kHz'], 0.95), ([f'in_scale', f'≫{FC.IN_SHIFT} 반올림·포화', f'→ {FC.FFT_DATA_BITS}b'], 1.25),
          (['프레이밍', f'창 {FC.WIN} (30 ms)', f'홉 {FC.HOP} (10 ms)'], 1.25), (['Hann 창', f'Q1.{FC.WIN_Q} ROM ×{FC.WIN}', f'≫15 → {FC.FFT_DATA_BITS}b'], 1.25),
          ([f'FFT{FC.N_FFT} radix-2', f'{FC.FFT_STAGES}단, 단마다 ½', f'트위들 Q1.{FC.TW_Q}, 곱셈기 4'], 1.4), (['파워', f're²+im², {FC.N_BINS}빈', f'{FC.POW_BITS}b 무손실'], 1.2),
          ([f'mel {FC.N_MELS}', f'삼각 가중 {FC.MEL_W_BITS}b 희소', f'누산 {FC.MEL_ACC_BITS}b'], 1.25), (['log2', f'선행 1 + {FC.LOG_MANT_BITS}b LUT', f'Q6.{FC.LOG_FRAC_BITS}'], 1.2),
          (['특징 8b', f'clip(log2q − {FC.LOG_OFFSET})', f'{FC.N_MELS}×1 / 프레임'], 1.2)]
    x = 0.15; xs = []
    gap = (11 - 0.3 - sum(w for _, w in st)) / (len(st) - 1)
    for i, (lines, w) in enumerate(st):
        box(ax, x, y, w, h, lines, ec=C[0] if 0 < i < len(st) - 1 else INK2, fs=7.8); xs.append((x, w))
        if i < len(st) - 1: arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2)
        x += w + gap
    # energies / gating
    box(ax, 1.4, 0.15, 3.4, 0.95, ['E0 = Σ|x18| (창 480)', 'E0 < T0 = 6,694 → 특징 전부 0 (증명, 손실 없음)', '보드 무음: DC 오프셋·잡음 바닥 때문에 E0 ≫ T0 → 미발동'], fc='#fff7f3', ec=C[1], fs=7.5)
    box(ax, 6.4, 0.15, 3.9, 0.95, ['E1 = Σ 특징 40개 (프레임 에너지) → VAD', 'T_on 600 / T_off 360 / N_on 4 / 홀드 30 / 프리롤 20', '구간 길이 25~130 프레임 → SNN·판정 동작'], fc='#f3f8ff', ec=C[0], fs=7.5)
    arrow(ax, xs[2][0] + xs[2][1] / 2, y, 3.1, 1.1, color=C[1]); arrow(ax, xs[8][0] + xs[8][1] / 2, y, 8.35, 1.1, color=C[0])
    ax.text(0.15, 3.55, f'프레임당 {5942:,} 사이클(곱셈기 4) / 10,568(직렬 1) @ 48 MHz, 예산 480,000 · 블록별 RTL이 12벡터 × 98프레임에서 골든과 비트 일치 · ROM: Hann, 트위들 256, mel 희소 표, log LUT 32', fontsize=7.5, color=INK2)
    ax.set_title('그림 2. 정수 프론트엔드 흐름과 비트 폭 (fe_config.py)', fontsize=10, loc='left')
    save(fig, 'fig02_frontend_flow.png', '정수 프론트엔드 처리 흐름과 비트 폭. 24비트 PCM을 6비트 내려 18비트로 만든 뒤 30 ms 창·10 ms 홉으로 프레이밍하고, Q1.15 Hann 창, 9단 radix-2 FFT512(단마다 1/2 스케일, 곱셈기 4개), 무손실 36비트 파워, 8비트 희소 삼각 mel 40밴드, 선행 1 검출 + 5비트 LUT의 log2(Q6.4)를 거쳐 오프셋 332를 뺀 8비트 특징을 만든다. E0 = Σ|x18|는 T0 게이팅(디지털 무음에서만 발동), E1 = Σ특징은 VAD 입력이다.', out)

# ------------------------------------------------------------------ 3. M3k3 structure + receptive field
def fig_m3k3(out):
    cfg = json.load(open(os.path.join(S, 'runs', 'M3k3e60_U2_q8_kd_s0', 'config.json')))
    K = cfg['K']; dil = [int(d) for d in cfg['dilations'].split(',')]; H = cfg['hidden']; kk = cfg['k']; n_in, n_out = 40, 12
    params = n_in * K * H + sum(H * K * H for _ in dil[1:]) + H * n_out
    rf = 1 + (K - 1) * sum(dil)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=[1.05, 1.3]))
    ax = axs[0]; ax.set_xlim(0, 5); ax.set_ylim(-0.3, 6.4); ax.axis('off')
    y = 5.6; box(ax, 0.9, y, 3.2, 0.6, [f'입력: 특징 {n_in} (8b, 프레임 = 타임스텝 10 ms)'], ec=INK2, fs=7.8, bold_first=False); y -= 0.25
    for i, d in enumerate(dil):
        y -= 1.0
        box(ax, 0.7, y, 3.6, 0.72, [f'TCN{i + 1}: 인과 확장 합성곱 K={K}, dilation {d}', f'QLinear {(n_in if i == 0 else H) * K}→{H} (8b) → LIF {H} (k={kk}, 감산 리셋)'], fs=7.6)
        arrow(ax, 2.5, y + 0.72 + 0.27, 2.5, y + 0.72)
        if i < len(dil) - 1: ax.text(2.62, y - 0.14, '1b 스파이크', fontsize=6.8, color=INK2, ha='left', va='center')
    y -= 1.0; box(ax, 0.9, y, 3.2, 0.72, [f'FC {H}→{n_out} (8b) + 출력 적분기', '구간 합 argmax → 클래스 / 마진'], fs=7.6); arrow(ax, 2.5, y + 0.99, 2.5, y + 0.72)
    ax.text(0.05, 0.05, f'LIF: v ← v − (v≫{kk}) + I, 16b 막전위, 뉴런별 정수 θ·b · 파라미터 {params:,} (≈{params / 1e3:.0f}k)\n수용장 1 + (K−1)·Σd = {rf} 프레임 = {rf * 10} ms · 입력층 1 MAC(DSP 1) + 은닉층 가산기 직렬, 스파이크 있는 행만 ROM 읽기', fontsize=6.9, color=INK2, va='top')
    ax.set_title('(a) M3k3 구조', fontsize=9.5, loc='left')
    ax = axs[1]; T = rf; L = len(dil)
    for li in range(L + 1):
        for t in range(T):
            ax.plot(t, li, 'o', ms=3.2, color='#d9d8d2', zorder=2)
    # receptive field tree from the last output frame
    active = {L: {T - 1}}
    for li in range(L, 0, -1):
        d = dil[li - 1]; nxt = set()
        for t in active[li]:
            for j in range(K):
                if t - j * d >= 0: nxt.add(t - j * d); ax.plot([t - j * d, t], [li - 1, li], color=C[li - 1], lw=0.9, alpha=0.8, zorder=3)
        active[li - 1] = nxt
    for li in range(L + 1):
        for t in active[li]: ax.plot(t, li, 'o', ms=4.2, color=C[li - 1] if li else INK2, zorder=4)
    ax.set_yticks(range(L + 1)); ax.set_yticklabels(['입력 특징'] + [f'TCN{i + 1} (d={d})' for i, d in enumerate(dil)]); ax.set_xlabel('프레임 (10 ms)')
    ax.set_xlim(-0.6, T - 0.4); ax.set_ylim(-0.5, L + 0.5); ax.grid(False)
    for i, d in enumerate(dil): ax.text(T - 0.3, i + 1 + 0.22, f'K={K} 탭, 간격 {d}', ha='right', fontsize=7.2, color=C[i])
    ax.text(0, L + 0.32, f'마지막 출력 1프레임의 수용장 = {rf} 프레임 ({rf * 10} ms)', fontsize=7.8, color=INK)
    ax.set_title('(b) dilation 수용장 (마지막 프레임 기준)', fontsize=9.5, loc='left')
    fig.tight_layout()
    save(fig, 'fig03_m3k3_structure.png', f'M3k3 SNN 구조와 수용장. (a) 40개 8비트 특징을 입력으로 인과 확장 합성곱(K={K}, dilation {"/".join(map(str, dil))}) + LIF(64뉴런, k={kk}, 감산 리셋) 4층이 1비트 스파이크로 이어지고, FC 64→12와 출력 적분기의 구간 합 argmax로 판정한다(파라미터 약 {params / 1e3:.0f}k). (b) 마지막 출력 프레임이 보는 입력 범위는 1 + (K−1)·Σd = {rf} 프레임 = {rf * 10} ms이다.', out)

# ------------------------------------------------------------------ 4. accuracy (seed stats + system det curve)
def fig_accuracy(out):
    rows = list(csv.DictReader(open(os.path.join(FIG, 'seed_stats.csv'))))
    dc = list(csv.DictReader(open(os.path.join(FIG, 'det_curve.csv'))))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axs[0]; style(ax); x = np.arange(len(rows)); cols = [C[0], C[1], C[0], C[1]]
    ax.bar(x, [float(r['mean']) for r in rows], yerr=[float(r['std']) for r in rows], color=cols, width=0.6, capsize=4, zorder=3, edgecolor=BG)
    for i, r in enumerate(rows): ax.text(i, float(r['mean']) + float(r['std']) + 0.15, f"{float(r['mean']):.2f} ± {float(r['std']):.2f}", ha='center', fontsize=7.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([r['group'].replace('DS-CNN 24ch', 'DS-CNN\n24ch').replace('SNN M3k3 Q8+KD', 'SNN M3k3\nQ8+KD') for r in rows], fontsize=7.5)
    ax.set_ylim(85, 93.5); ax.set_ylabel('test 정확도 (%)'); ax.set_title('(a) 클립 정확도, 시드 3개 평균 ± 표준편차 (base 데이터)', fontsize=9.5, loc='left')
    ax.text(0.5, 0.975, 'U2(unknown 3배) 데이터: SNN 85.4 % / DS-CNN 86.2 % (12클래스, s0)', transform=ax.transAxes, fontsize=7.2, color=INK2, ha='center', va='top')
    ax = axs[1]; style(ax)
    sel = [('SNN U2', 'test_board', C[1], '-', 'SNN M3k3 U2 (ROM)'), ('DS-CNN 24ch U2-style (last 98 frames)', 'test_board', C[0], '-', 'DS-CNN 24ch U2 (구간 끝 98프레임)'),
           ('SNN base', 'test_board', C[1], ':', 'SNN M3k3 base'), ('DS-CNN 24ch base (last 98 frames)', 'test_board', C[0], ':', 'DS-CNN 24ch base')]
    for m, cond, col, ls, lab in sel:
        r = sorted([q for q in dc if q['model'] == m and q['split_cond'] == cond], key=lambda q: float(q['fa_per_h']))
        if not r: continue
        ax.plot([float(q['fa_per_h']) for q in r], [100 * float(q['det']) for q in r], color=col, ls=ls, lw=1.8, marker='o', ms=3, label=lab, zorder=3)
    ax.set_xscale('log'); ax.set_xlabel('오검출 (FA/h, test 보드 조건 스트림, 마진 스윕)'); ax.set_ylabel('키워드 검출률 (%)'); ax.legend(frameon=False, fontsize=7.5, loc='lower right')
    ax.set_title('(b) 시스템 수준: 같은 VAD 구간 파이프라인에서 검출률–오검출', fontsize=9.5, loc='left')
    fig.tight_layout()
    save(fig, 'fig04_accuracy.png', '정확도 비교. (a) Speech Commands 12/10클래스 test 클립 정확도(시드 3개 평균 ± 표준편차): DS-CNN 24ch 89.08/91.66 %, SNN M3k3 8비트 QAT+KD 87.67/89.59 %. (b) 같은 VAD 구간 파이프라인에서 판정 마진을 바꾸며 얻은 검출률–오검출 곡선(test 보드 조건 스트림): SNN U2(최종 ROM)와 DS-CNN 24ch(구간 끝 98프레임 입력), base 학습 모델. 선택은 val에서, 보고는 test에서 했다.', out)

# ------------------------------------------------------------------ 5. class confusion
def fig_confusion(out):
    rows = list(csv.DictReader(open(os.path.join(FIG, 'confusion.csv')))); cl = [k for k in rows[0] if k not in ('model', 'true')]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.9))
    for ax, key, title in zip(axs, ('snn', 'dscnn'), ('SNN M3k3 Q8+KD U2 s0 (최종 ROM, 정수 골든)', 'DS-CNN 24ch (base)')):
        m = np.array([[float(r[c]) for c in cl] for r in rows if r['model'] == key]); mn = m / m.sum(1, keepdims=True); acc = np.trace(m) / m.sum()
        ax.imshow(mn, cmap='Blues', vmin=0, vmax=1); ax.set_xticks(range(len(cl))); ax.set_yticks(range(len(cl))); ax.set_xticklabels(cl, rotation=60, fontsize=7); ax.set_yticklabels(cl, fontsize=7)
        for i in range(len(cl)):
            for j in range(len(cl)):
                if m[i, j] > 0 and (i == j or m[i, j] >= 10): ax.text(j, i, int(m[i, j]), ha='center', va='center', fontsize=6, color='white' if mn[i, j] > 0.5 else INK)
        ax.set_title(f'{title} — 정확도 {100 * acc:.1f} %', fontsize=9.5); ax.set_xlabel('판정'); ax.set_ylabel('정답'); ax.grid(False)
        for s in ('top', 'right'): ax.spines[s].set_visible(True)
    fig.tight_layout()
    save(fig, 'fig05_confusion.png', '12클래스 test(4,888클립) 혼동행렬(행 = 정답, 열 = 판정, 행 정규화 색). 왼쪽은 보드에 올린 최종 ROM(U2 s0, `snn/runs/confusion_u2.json`, 정수 골든 추론), 오른쪽은 base DS-CNN 24ch. U2 SNN의 오류는 unknown 열로 몰린다(키워드 행마다 25~58개): unknown 재현율 92.6 % 대 DS-CNN 66.1 %(+26.5 %p), 대신 silence 89.4 % 대 100 %(−10.6 %p), 클립 정확도 85.4 % 대 89.2 %. 두 모델에 공통인 혼동쌍은 off↔up, on↔off, no↔go↔down, left↔yes. 대각선과 10 이상인 칸에 개수를 적었다.', out)

# ------------------------------------------------------------------ 6. resources (implementation results)
def fig_resources(out):
    names = ['hls4ml DS-CNN\nKWS 계층(구현)', 'kws_top\n(본 설계 전체)', 'SNN 코어+판정\n(범위 맞춤)']
    R = {'LUT': [19710, 5173, 4143], 'FF': [34917, 5795, 4865], 'BRAM18K': [111, 39, 32], 'DSP': [63, 9, 0]}
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.6))
    for ax, (k, v) in zip(axs, R.items()):
        style(ax); b = ax.bar(range(3), v, color=[INK2, C[0], C[2]], width=0.62, zorder=3, edgecolor=BG)
        for i, val in enumerate(v): ax.text(i, val + max(v) * 0.02, f'{val:,}', ha='center', fontsize=7.5, color=INK)
        ax.text(1, v[0] * 0.78, f'{v[0] / v[1]:.1f}×' if v[1] else '–', ha='center', fontsize=9, color=C[0], weight='bold')
        ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=6.8); ax.set_title(k, fontsize=9.5); ax.set_ylim(0, max(v) * 1.15)
    fig.suptitle('그림 6. 구현 자원 비교 (Vivado 2023.2, xc7z020clg484-1, 48 MHz, 구현 결과끼리) — LUT 3.8× / FF 6.0× / BRAM18K 2.8× / DSP 7×', fontsize=9.5, y=1.0)
    fig.text(0.01, 0.005, "기준: hls4ml DS-CNN 통합 설계의 KWS 관련 계층(myproject_0 + DMA + GPIO + 인터커넥트 + latch, 특징 추출 미포함). 신청서의 'BRAM 1/5'는 HLS 합성 추정(IP 단독 199) 기준이었고 구현치끼리는 111 → 39로 2.8배(미달).", fontsize=6.8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, 'fig06_resources.png', '구현 자원 비교(구현 결과끼리, 같은 보드·48 MHz·Vivado 2023.2). hls4ml DS-CNN 통합 설계의 KWS 관련 계층 LUT 19,710 / FF 34,917 / BRAM18K 111 / DSP 63 대 본 설계 kws_top(프론트엔드·VAD·프리롤·SNN·판정 포함) 5,173 / 5,795 / 39 / 9 → 3.8× / 6.0× / 2.8× / 7×. 범위를 맞춘 SNN 코어 + 판정만은 4,143 / 4,865 / 32 / 0. 신청서의 BRAM 1/5 목표는 HLS 추정(199) 기준이었고 구현 기준 2.8배로 미달이다.', out)

# ------------------------------------------------------------------ 7. power table (as a figure)
def fig_power(out):
    rows = [['구분', '방법', '무음 / 키워드 / 대화 (W)', '비고'],
            ['hls4ml DS-CNN 기준 IP (+DMA 등)', 'vectorless (활동률 가정)', '≈ 0.116 (IP 0.112)', '클록 0.029 + BRAM 0.030 + DSP 0.025 + 신호 0.017 + 논리 0.011'],
            ['본 설계 kws_top', 'vectorless', '0.029', '같은 방법끼리 ≈ 4× (규모 비교)'],
            ['SNN 코어 + 판정 (범위 맞춤)', 'vectorless', '0.018', ''],
            ['kws_top (게이팅 없음)', 'SAIF 활동량 (넷 매칭 6 %)', '0.021 / 0.019 / 0.019', '클록 트리 0.009, 논리 0.005, 신호 0.002, BRAM 0.002'],
            ['kws_top 게이팅 RTL, 클록 상시 on (= 게이팅 OFF)', 'SAIF', '0.024 / 0.022 / 0.022', 'BUFGCE CE = 1'],
            ['kws_top 게이팅 ON (듀티 가중 추정)', 'SAIF + CE 듀티 (SNN 0~1.2 %, FE 1.2 %)', '0.0060 / 0.0062 / 0.0061', '약 3.7~4× (본 설계 내부 효과, 추정)'],
            ['상시 대기 평균 (무음 90 %)', '추정', '0.021 → 0.006', ''],
            ['기준 IP 클록 성분 (활동 무관 하한)', 'vectorless Clocks 항목', '0.029', '게이팅 없는 한 상시 소비 > 본 설계 전체 추정 0.006'],
            ['보드 입력 전력 차분 (P0~P6)', 'N6705C 데이터 로거', 'TBD(보드)', 'power_log_analyze.py, 해상도 이하면 그렇게 기록']]
    fig, ax = plt.subplots(figsize=(11, 4.3)); ax.axis('off')
    tb = ax.table(cellText=rows[1:], colLabels=rows[0], loc='center', cellLoc='left', colWidths=[0.27, 0.2, 0.17, 0.36])
    tb.auto_set_font_size(False); tb.set_fontsize(7.3); tb.scale(1, 1.55)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor('#d0cfc8'); cell.set_linewidth(0.6)
        if r == 0: cell.set_facecolor('#eef3fa'); cell.set_text_props(weight='bold')
        elif r in (6,): cell.set_facecolor('#fff7f3')
        elif r in (1, 8): cell.set_facecolor('#f6f6f4')
    ax.set_title('그림 7. PL 동적 전력 (Vivado report_power, kws_top; PS7 1.527 W·정적 0.159 W 제외) — 같은 방법끼리만 비교', fontsize=9.5, loc='left')
    fig.text(0.01, 0.02, '해석: 기준 설계와의 비교는 vectorless끼리(≈ 4×)만. SAIF·BUFGCE 게이팅 값(0.022~0.024 → 0.006 W)은 기준 설계에 대응 측정이 없으므로 본 설계 내부 전후 효과로만 제시. 보드 실측은 입력 전력 차분(BOARD_STEP2 §4).', fontsize=6.8, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, 'fig07_power_table.png', 'PL 동적 전력 표(Vivado report_power, kws_top 기준). 기준 hls4ml IP vectorless ≈ 0.116 W 대 본 설계 0.029 W(같은 방법끼리 규모 비교 ≈ 4×). RTL 활동량(SAIF)으로는 0.019~0.021 W이고, BUFGCE 클록 게이팅의 CE 듀티를 가중하면 0.006 W로 추정된다(본 설계 내부 효과, 추정). 기준 IP의 클록 성분 0.029 W는 활동과 무관한 하한이다. 보드 입력 전력 차분은 TBD(보드).', out)

# ------------------------------------------------------------------ 8. VAD example (golden pipeline, ROM U2 s0)
def fig_vad(out):
    import torch; torch.backends.cuda.matmul.allow_tf32 = False
    from snn.golden.pipeline_int_ref import run_pipeline
    from snn.frontend.frontend_ref import scale_input, frame
    from snn.golden.export_snn_mem import load_run, extract
    model, cfg, res = load_run(os.path.join(S, 'runs', 'M3k3e60_U2_q8_kd_s0')); P = extract(model)
    vd = os.path.join(S, 'golden', 'vectors', 'activity_board', 'stream_0'); info = json.load(open(os.path.join(vd, 'info.json')))
    pcm = np.array([int(l, 16) - (1 << 24 if int(l, 16) >= (1 << 23) else 0) for l in open(os.path.join(vd, 'pcm24.hex')) if not l.startswith('//')], dtype=np.int64)
    r = run_pipeline(P, pcm, info['params'], 'E1'); T = r['T']; t0 = json.load(open(os.path.join(S, 'runs', 't0_bound.json')))['T0_proven']
    E0 = np.abs(frame(scale_input(pcm))).sum(-1); gated = E0 < t0; tt = np.arange(T) * 0.01; secs = min(12.0, T * 0.01)
    act = np.array([x['active'] for x in r['rec']]); fed = np.array([x['fed'] > 0 for x in r['rec']])
    fig, axs = plt.subplots(5, 1, figsize=(11, 7.2), sharex=True)
    axs[0].plot(np.arange(len(pcm)) / 16000, pcm / 2 ** 23, color=C[0], lw=0.4); axs[0].set_ylabel('파형')
    for e in info['events']: axs[0].axvspan(e['onset'] * 0.01, e['offset'] * 0.01, color=C[3] if e['label'] < 10 else C[7], alpha=0.15)
    for e in info['events']:
        if e['onset'] * 0.01 < secs: axs[0].text(e['onset'] * 0.01 + 0.05, 0.8, CLASSES[e['label']] if e['label'] < 10 else 'unknown', fontsize=7, color=INK2)
    axs[1].plot(tt, r['E'], color=C[1], lw=1); axs[1].axhline(info['params']['T_on'], color=INK2, lw=0.7, ls='--'); axs[1].axhline(info['params']['T_off'], color=INK2, lw=0.7, ls=':'); axs[1].set_ylabel('E1 = Σ특징')
    axs[1].text(0.01, 0.8, f"T_on {info['params']['T_on']} (--), T_off {info['params']['T_off']} (··)", transform=axs[1].transAxes, fontsize=7, color=INK2)
    axs[2].fill_between(tt, 0, act, step='post', color=C[2], alpha=0.6); axs[2].set_ylabel('VAD 구간'); axs[2].set_yticks([0, 1])
    axs[3].fill_between(tt, 0, fed, step='post', color=C[5], alpha=0.6, label='SNN 프레임 공급(클록 on)'); axs[3].fill_between(tt, 0, gated * 0.5, step='post', color=C[4], alpha=0.7, label='FE T0 게이팅(디지털 무음)')
    axs[3].set_ylabel('SNN on / FE 게이팅'); axs[3].set_yticks([0, 1]); axs[3].legend(frameon=False, fontsize=7, loc='upper left', ncol=1)
    for d in r['dets']:
        if d['t'] * 0.01 > secs: continue
        axs[4].axvline(d['t'] * 0.01, color=C[2] if d['valid'] else C[7], lw=1.5); axs[4].text(d['t'] * 0.01 + 0.06, 0.5, CLASSES[d['cls']], fontsize=7.5, color=INK, rotation=90, va='center')
    axs[4].set_yticks([]); axs[4].set_ylabel('판정'); axs[4].set_xlabel('시간 (s)'); axs[4].set_xlim(0, secs)
    for ax in axs: style(ax)
    duty = fed.mean(); fig.suptitle(f'그림 8. VAD 동작 예시 (합성 보드 조건 스트림, ROM U2 s0, 골든 파이프라인; SNN 가동률 {100 * duty:.1f} %, FE T0 게이팅 {100 * gated.mean():.1f} %)', fontsize=9.5)
    fig.tight_layout()
    save(fig, 'fig08_vad_example.png', f'VAD 동작 예시(합성 보드 조건 스트림 앞 12 s, 골든 파이프라인, ROM U2 s0). 위부터 파형(노랑 = 키워드, 빨강 = unknown 구간), 프레임 에너지 E1과 T_on/T_off, VAD 구간, SNN 프레임 공급(= 클록 on)과 프론트엔드 T0 게이팅, 구간 판정(초록 = 키워드, 빨강 = 비키워드). 전체 스트림에서 SNN 가동률 {100 * duty:.1f} %. T0 게이팅은 합성 스트림의 디지털 무음 구간에서만 걸리며({100 * gated.mean():.1f} %), 보드 캡처처럼 잡음 바닥·DC 오프셋이 있으면 걸리지 않는다.', out)

# ------------------------------------------------------------------ 9. level sweep + stop/up confusion
def fig_level(out):
    SW = {'U2 (최종 ROM)': json.load(open(os.path.join(S, 'runs', 'board_capture', 'level_sweep.json')))['sweep'],
          'U2c s0 (예비 ROM)': json.load(open(os.path.join(S, 'runs', 'board_capture_U2c_s0', 'level_sweep.json')))['sweep'],
          'U2c s1': json.load(open(os.path.join(S, 'runs', 'board_capture_U2c_s1', 'level_sweep.json')))['sweep']}
    LC = json.load(open(os.path.join(S, 'runs', 'board_capture', 'level_confusion.json')))
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.8), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    for (name, sw), col in zip(SW.items(), [C[0], C[1], C[2]]):
        g = [r['gain_db'] for r in sw]
        axs[0].plot(g, [r['acc'] for r in sw], color=col, lw=2, marker='o', ms=3.5, label=name, zorder=3)
        axs[1].plot(g, [r['recall']['stop'] for r in sw], color=col, lw=2, marker='o', ms=3.5, label=name, zorder=3)
    for ax, t in zip(axs[:2], ['(a) 키워드 정확도 (test 100/키워드, 클립 모드)', '(b) stop 재현율']):
        style(ax); ax.set_title(t, fontsize=9, loc='left'); ax.set_xlabel('입력 감쇠 dB (0 = 데이터셋 원 레벨)'); ax.set_ylim(0, 1)
        ax.axvspan(-36, -27, color='#f0f0ee', zorder=0); ax.text(-31.5, 0.93, 'far 20~30 cm', ha='center', fontsize=7, color=INK2)
        ax.axvspan(-13, -6, color='#f0f0ee', zorder=0); ax.text(-9.5, 0.93, 'near 3 cm', ha='center', fontsize=7, color=INK2); ax.legend(frameon=False, fontsize=7, loc='lower right')
    ax = axs[2]; style(ax); gains = ['0', '-12', '-20', '-30']; run = 'M3k3e60_U2_q8_kd_s0'; si, ui = 2, 4
    cats = [('stop', C[2]), ('up', C[1]), ('unknown + silence', INK2), ('그 외', '#c9c8c0')]
    x = np.arange(len(gains)); bottom = np.zeros(len(gains))
    for k, (lab, col) in enumerate(cats):
        vals = []
        for g in gains:
            cm = LC[run][g]['stop']; n = sum(cm)
            v = cm[si] if k == 0 else cm[ui] if k == 1 else cm[10] + cm[11] if k == 2 else n - cm[si] - cm[ui] - cm[10] - cm[11]
            vals.append(100 * v / n)
        ax.bar(x, vals, bottom=bottom, color=col, width=0.6, label=f'stop → {lab}', zorder=3, edgecolor=BG)
        for i, v in enumerate(vals):
            if v >= 4: ax.text(i, bottom[i] + v / 2, f'{v:.0f}', ha='center', va='center', fontsize=7, color='white' if k < 3 else INK)
        bottom += vals
    up_to_stop = [100 * LC[run][g]['up'][si] / sum(LC[run][g]['up']) for g in gains]
    ax.plot(x, up_to_stop, color=C[5], marker='s', ms=4, lw=1.5, label='up → stop (%)', zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([f'{g} dB' for g in gains]); ax.set_ylabel('stop 발화의 판정 비율 (%)'); ax.set_ylim(0, 100)
    ax.set_title('(c) U2: 레벨별 stop 판정 분해 (test stop 411개)', fontsize=9, loc='left'); ax.legend(frameon=False, fontsize=6.8, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3, handlelength=1.2, columnspacing=0.9)
    fig.tight_layout()
    save(fig, 'fig09_level_sweep_confusion.png', '입력 레벨 스윕과 stop↔up 혼동. (a)(b) test 클립 100개/키워드를 0~−36 dB로 감쇠했을 때의 키워드 정확도와 stop 재현율(클립 모드): 최종 ROM U2와 채널 증강 재학습 U2c(s0 예비 ROM, s1). 보드 3 cm(near)·20~30 cm(far) 조건의 레벨 구간을 음영으로 표시. (c) U2 s0에서 test stop 전체(411개)의 레벨별 판정 분해: 레벨이 낮아지면 stop은 up이 아니라 unknown·silence로 빠지며, up → stop 오인은 어느 레벨에서도 1.5 % 이하다. 따라서 코덱 게인 보정이 1순위이고 재학습은 보조다.', out)

# ------------------------------------------------------------------ 10. board ILA capture (mode 3) + golden VAD/decision
def fig_board_capture(out):
    from snn.eval.board_capture_check import read_capture
    sm = json.load(open(os.path.join(S, 'runs', 'board_capture', 'summary.json'))); prm = sm['params']; lead = sm['pad']['lead_s']; trail = sm['pad']['trail_s']
    names = ['stop_near_01', 'stop_far_01', 'up_far_01', 'quiet_01']
    fig, axs = plt.subplots(2, len(names), figsize=(11.5, 5.2), sharex='col')
    for j, n in enumerate(names):
        x18, _ = read_capture(os.path.join(S, 'data', 'board_capture', n + '.csv')); r = sm['files'][n]
        t = np.arange(len(x18)) / 16000; dur = len(x18) / 16000
        axs[0, j].plot(t, x18 / 2 ** 17, color=C[0], lw=0.35); axs[0, j].set_ylim(-1.05, 1.05)
        axs[0, j].set_title(f"{n}\n피크 {r['peak_dbfs24']:.0f} / 바닥 {r['floor_dbfs']:.0f} dBFS, DC {r['dc'] / 64:.0f} LSB18", fontsize=7.6, loc='left')
        E = np.array(r['E1']); te = np.arange(len(E)) * 0.01 - lead
        axs[1, j].plot(te, E, color=C[1], lw=1); axs[1, j].axhline(prm['T_on'], color=INK2, lw=0.7, ls='--'); axs[1, j].axhline(prm['T_off'], color=INK2, lw=0.7, ls=':')
        for ax in axs[:, j]: ax.axvspan(dur, dur + trail, color='#f0f0ee', zorder=0)
        axs[0, j].text(dur + 0.03, 0.8, '0 패딩\n(골든)', fontsize=6.5, color=INK2)
        for d in r['golden']['dets']:
            td = d['t'] * 0.01 - lead; axs[1, j].axvline(td, color=C[2] if d['valid']['0'] else C[7], lw=1.4)
            axs[1, j].text(td - 0.05, E.max() * 0.85 if E.max() > 0 else 1, f"판정 {d['name']}\n(마진 {d['margin']})", fontsize=7, color=INK, ha='right')
        if not r['golden']['dets']: axs[1, j].text(0.03, 0.85, 'VAD 미발동' if not r['golden']['vad_triggered'] else '판정 없음', transform=axs[1, j].transAxes, fontsize=7.5, color=INK2)
        axs[1, j].set_xlabel('시간 (s)'); axs[1, j].set_xlim(0, dur + trail)
        for ax in axs[:, j]: style(ax)
    axs[0, 0].set_ylabel('ILA dbg_cap_sample (18b, ±1 = 풀스케일)'); axs[1, 0].set_ylabel('E1 = Σ특징 (골든)')
    fig.suptitle(f"그림 10. 보드 ILA 모드 3 캡처(레퍼런스 코덱 설정, 게인 보정 전)와 골든 E1·판정 (T_on {prm['T_on']} --, T_off {prm['T_off']} ··)", fontsize=9.5)
    fig.tight_layout()
    save(fig, 'fig10_board_ila_capture.png', '보드 ILA 모드 3 캡처 파형과 같은 샘플을 정수 골든에 넣은 E1·VAD·판정. 코덱 게인 보정 전(레퍼런스 초기화)이라 3 cm(near) 발화는 피크 약 −10 dBFS, 20~30 cm(far) 발화는 약 −30 dBFS로 데이터셋보다 낮고, 조용한 방(quiet)은 바닥 −76 dBFS에 DC 오프셋 약 −1,200 LSB18이 있어 특징은 전부 0(VAD 미발동)이지만 Σ|x18|가 T0보다 커 T0 게이팅은 걸리지 않는다. far 조건의 stop은 E1이 T_on에 못 미쳐 VAD조차 열리지 않는다(게인 보정 대상). 판정 시각은 VAD 구간 종료(홀드 30프레임) 시점이라 캡처 끝의 0 패딩 구간에 찍힌다. 게이팅 ON/OFF CE 파형 등 ILA 화면 캡처 PNG는 TBD(보드).', out)

# ------------------------------------------------------------------ 11. activity (simulation) gating
def fig_activity(out):
    rows = list(csv.DictReader(open(os.path.join(FIG, 'activity.csv'))))
    scen = ['board', 'talk', 'quiet']
    def get(sc, t0, vad, key):
        for r in rows:
            if r['cfg'] == f'{sc} T0={t0} VAD={vad}': return int(r.get(key, 0) or 0) / int(r['frames'])
        return float('nan')
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8)); x = np.arange(len(scen)); w = 0.38
    axs[0].bar(x - w / 2, [get(sc, 0, 1, 'fe_cyc') for sc in scen], w, color=C[0], alpha=0.45, label='T0 게이팅 off', zorder=3, edgecolor=BG)
    axs[0].bar(x + w / 2, [get(sc, 6694, 1, 'fe_cyc') for sc in scen], w, color=C[0], label='T0 게이팅 on (6,694, 손실 없음)', zorder=3, edgecolor=BG)
    for i, sc in enumerate(scen):
        a, b = get(sc, 0, 1, 'fe_cyc'), get(sc, 6694, 1, 'fe_cyc'); axs[0].text(i + w / 2, b + 150, f'{100 * b / a:.0f} %', ha='center', fontsize=7.5, color=INK2)
    axs[0].set_xticks(x); axs[0].set_xticklabels(['board (합성)', 'talk (합성)', 'quiet (합성, 디지털 무음 포함)']); axs[0].set_ylabel('프론트엔드 활성 사이클 / 프레임'); axs[0].set_title('(a) 프론트엔드: T0 게이팅 (시뮬, 합성 스트림)', fontsize=9, loc='left'); axs[0].set_ylim(0, 7800); axs[0].legend(frameon=False, fontsize=7.5, loc='upper center', ncol=2)
    axs[1].bar(x - w / 2, [get(sc, 0, 0, 'snn_cyc') for sc in scen], w, color=C[1], alpha=0.45, label='VAD 없음 (상시 공급)', zorder=3, edgecolor=BG)
    axs[1].bar(x + w / 2, [get(sc, 6694, 1, 'snn_cyc') for sc in scen], w, color=C[1], label='VAD 구간만', zorder=3, edgecolor=BG)
    for i, sc in enumerate(scen):
        a, b = get(sc, 0, 0, 'snn_cyc'), get(sc, 6694, 1, 'snn_cyc'); axs[1].text(i + w / 2, b + 150, f'{100 * b / a:.0f} %', ha='center', fontsize=7.5, color=INK2)
    axs[1].set_xticks(x); axs[1].set_xticklabels(['board (합성)', 'talk (합성)', 'quiet (합성)']); axs[1].set_ylabel('SNN 활성 사이클 / 프레임'); axs[1].set_title('(b) SNN: VAD 게이팅', fontsize=9, loc='left'); axs[1].set_ylim(0, 5300); axs[1].legend(frameon=False, fontsize=7.5, loc='upper center', ncol=2)
    for ax in axs: style(ax)
    fig.text(0.01, 0.005, '프레임 예산 480,000 사이클(10 ms @ 48 MHz). T0 게이팅 수치는 합성 스트림의 디지털 무음 구간에서 얻은 것으로, 잡음 바닥·DC 오프셋이 있는 보드 무음에서는 발동하지 않는다(무음 절감은 VAD + BUFGCE가 담당).', fontsize=6.8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, 'fig11_activity_gating.png', '활동량 테스트벤치(RTL 시뮬레이션)에서 잰 프레임당 활성 사이클. (a) 프론트엔드의 T0 게이팅 전후, (b) SNN의 VAD 게이팅 전후(board/talk/quiet 합성 스트림). T0 게이팅의 76~82 %는 합성 스트림의 디지털 무음에서 얻은 값이며 보드 무음(잡음 바닥 −76 dBFS, DC 오프셋)에서는 발동하지 않으므로, 실제 무음 절감은 VAD와 BUFGCE 클록 게이팅에서 온다.', out)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(FIG, 'final')); ap.add_argument('--skip_vad', action='store_true'); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_system(a.out); fig_frontend(a.out); fig_m3k3(a.out); fig_accuracy(a.out); fig_confusion(a.out); fig_resources(a.out); fig_power(a.out)
    if not a.skip_vad: fig_vad(a.out)
    fig_level(a.out); fig_board_capture(a.out); fig_activity(a.out)
    L = ['# 최종보고서 그림 (300 dpi PNG) — 캡션 초안', '', f'생성: `python snn/docs/make_final_figures.py` (이 폴더). 수치 출처: `snn/runs/vivado_impl.md`, `snn/board/pkg/POWER_COMPARE.md`, `snn/docs/figures/*.csv`, `snn/runs/board_capture*/`, `snn/runs/board_capture/summary.json`.',
         '보드 그림: fig12(`make_fig12_gating.py`, `snn/board/log/gate_on.csv`/`gate_off.csv`), fig13(`snn/eval/playback_compare.py`, `snn/board/log/R0_det.csv`/`R0b_det.csv`). 전력 시계열(`power_result.png`)은 이번 보고서에서 생략(향후 과제, N6705C).', '', '| 파일 | 보고서 위치(안) | 캡션 초안 |', '|---|---|---|']
    where = {'fig01': '2.1 설계 개요', 'fig02': '2.2/2.7 프론트엔드', 'fig03': '2.4 SNN 구조', 'fig04': '3.1 정확도', 'fig05': '3.1 정확도(보조)', 'fig06': '3.5 자원', 'fig07': '3.5 전력',
             'fig08': '2.6 판정/VAD', 'fig09': '3.7 보드·레벨', 'fig10': '3.7 보드 캡처', 'fig11': '3.3 활동량'}
    for f, c in CAP: L.append(f'| `{f}` | {where[f[:5]]} | {c} |')
    L += ['| `fig12_ila_gating_waveform.png` | 3.7 보드 클록 게이팅 | 보드 ILA 캡처(2026-11-25, 게이팅 빌드 10-29, 앱 kws16_vC): yes 클립 재생 중 VAD 구간 시작(트리거 dbg_vad_active 상승, 깊이 16384, 위치 2000)의 FE/SNN 클록 CE. 게이팅 ON(sw[5]=OFF): fe_frame_done(1999) → vad_active·snn_active(2000) → SNN CE(2002), FE CE는 프레임 처리 종료 뒤 OFF_DELAY 2로 2002에 내려가 두 CE가 같은 사이클에 교대한다(겹침·공백 0, 핸드오프 3사이클 = 레지스터 3단, 설계 의도와 일치). 게이팅 OFF(sw[5]=ON, force_on): 두 CE 상시 1, 나머지 신호 타이밍 동일. 오른쪽은 전체 창(0.34 ms). |',
          '| `fig13_board_confusion.png` | 3.7 보드 재생 시험 | 재생 시험 혼동행렬(행 = 발화, 열 = 판정, 무검출 열 포함). 왼쪽: 보드 R0+R0b(마진 0, 2회 합산, 코덱 vC, PC 스피커 8 cm, 체인 이득 +1 dB) 키워드 정검출 71.2 %; 오른쪽: 같은 wav를 디지털로 넣은 골든 83.8 %. 보드의 손실은 거의 무검출 열(키워드 160×2 중 63)로 나타나고 오분류 패턴(no→go/down, on→off, go→yes/right)은 골든과 같다. silence 슬롯(핑크 잡음) 2개가 두 런 모두 down으로 오검출됐다. |',
          '', '## 생략/향후 (보드)', '- 보드 전력 시계열(`power_log_analyze.py --fig`, 상태 P0~P6): 이번 보고서에서 생략, 향후 과제(N6705C 데이터 로거).']
    open(os.path.join(a.out, 'index.md'), 'w').write('\n'.join(L) + '\n'); print('index.md written with', len(CAP), 'captions')

if __name__ == '__main__':
    main()
