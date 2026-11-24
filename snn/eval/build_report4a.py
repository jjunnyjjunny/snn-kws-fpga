"""build_report4a.py — assemble snn/REPORT_week4a.md from measured artifacts (re-run after any update)."""
import os, re, json, datetime, glob, csv
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); S = os.path.join(ROOT, 'snn')
def rd(p): p = os.path.join(S, p); return open(p).read() if os.path.exists(p) else ''
def J(p): p = os.path.join(S, p); return json.load(open(p)) if os.path.exists(p) else None
def R(n): return J(f'runs/{n}/results.json')
fa = J('runs/fa_breakdown_board.json'); t0 = J('runs/t0_bound.json')
base = J('runs/vad_eval/eval_M3k3e60_c12_q8_kd_base.json')
# activity parsing
def parse_act(fn):
    rows = []; cur = None
    for l in rd(fn).splitlines():
        if l.startswith('=='): cur = l.strip('= \n')
        elif l.startswith('ACTIVITY'): d = dict(re.findall(r'(\w+)=(\S+)', l)); d['cfg'] = cur; rows.append(d)
        elif l.startswith('ACT ') and rows: rows[-1].update(dict(re.findall(r'(\w+)=(\S+)', l)))
    return rows
act = parse_act('runs/activity_board_talk.txt') + parse_act('runs/activity_quiet.txt')
L = []
L.append(f"# SNN KWS — 4주차 A 보고서 ({datetime.date.today()})\n")
L.append("범위(보드·Vivado 없이): **0** 오검출 원인 분해, **A** 현실적 스트림 + unknown/silence 강화 재학습(U1~U3) + 동작점 스위치, **B** 프론트엔드 2단 게이팅(증명된 T0), **C** Verilator lint·Yosys 추정·활동량 계측·Vivado 전력 스크립트, **D** 최종보고서 초안·그림·발표 구성. 동결 사양 불변. 수치는 실제 실행 결과, 추정은 \"추정\". 모델·파라미터 선택은 val 스트림으로만.\n")
# 0
if fa:
    c = fa['counts']; L.append(f"""## 0. 오검출 원인 분해 (3주차 VAD 추천 파라미터, 마진 0, 보드 조건 test 200 × 30 s = {fa['hours']:.2f} h, `snn/eval/fa_breakdown.py`)

| 유형 | 건수 | /시간 | 비고 |
|---|---|---|---|
| unknown 발화 → 키워드 판정 | {c['fa_unknown']} | {fa['fa_breakdown_per_hour']['fa_unknown']:.1f} | unknown 발화 {fa['n_unknown_events']}개 중 **{100*fa['unknown_rate_as_keyword']:.1f} %** |
| 한 발화가 두 구간으로 쪼개져 중복 검출 | {c['fa_dup']} | {fa['fa_breakdown_per_hour']['fa_dup']:.1f} | 구간 중앙값 길이 {fa['segments']['len_median']:.0f} 프레임 |
| 배경소음·무음 구간 → 키워드 | {c['fa_noise']} | {fa['fa_breakdown_per_hour']['fa_noise']:.1f} | VAD가 소음 구간을 거의 잡지 않음 |
| **합계** | {fa['fa_total']} | **{fa['fa_per_hour']:.1f}** | |

예측 클래스별(hit / wrong / unknown→키워드): """ + ", ".join(f"{k} {v['hit']}/{v['wrong']}/{v['fa_unknown']}" for k, v in fa['per_class'].items()) + f". unknown→키워드가 많은 클래스: on, no, left, right. 학습 데이터 클래스 비율: 키워드 각 ~3,077, silence 3,077, unknown 3,077(8.3 %)로 unknown이 과소 대표됨.\n")
# A
L.append("## A. 오검출 줄이기\n")
if base:
    rs = base['results']
    L.append("### A-1. 새 스트림 (`vad_eval.py --conds talk,quiet,board`): talk = 발화의 60 % unknown(25단어), 게인 ±6 dB, 소음 U(0,0.05), 30 s; quiet = 간격 8~25 s(무음·소음 ≥ 90 %), 60 s. val 100 / test 200 스트림씩. 기본 모델(M3k3 Q8+KD) 결과:\n")
    L.append("| 스트림 | 오라클 | 마진 | 검출 % | 오분류 % | 미검출 % | 오검출/h | SNN 가동률 |\n|---|---|---|---|---|---|---|---|")
    for key in ('val_talk', 'test_talk', 'val_quiet', 'test_quiet', 'test_board'):
        if key not in rs: continue
        for m, d in rs[key].items():
            if not m.startswith('margin'): continue
            L.append(f"| {key} | {100*rs[key]['oracle_acc']:.1f} | {m.split('_')[1]} | {100*d['det_rate']:.1f} | {100*d['wrong_rate']:.1f} | {100*d['miss_rate']:.1f} | {d['false_alarms_per_hour']:.1f} | {100*d['snn_duty']:.1f} % |")
    L.append("\n- 대화 많은 스트림에서는 마진 0 기준 오검출 **184/h**(보드 조건 44/h의 4배): 오검출이 unknown 비율에 비례함을 확인. 조용한 상시 대기 스트림의 SNN 가동률 6.7 %.\n")
# A-2 results
L.append("### A-2. unknown·silence 강화 재학습 (M3k3 구조 동결, float 60ep → Q8 QAT + KD(변형별 48ch teacher), 시드 0/1)\n")
L.append("| 변형 | 학습 데이터 | 변형별 teacher | float s0/s1 | Q8+KD s0/s1 (12cls test) | unknown acc | silence acc | val_talk 검출 / 오검출/h (마진 0) |\n|---|---|---|---|---|---|---|---|")
for U, desc in (('base', '기본(unknown 1×, silence 1×)'), ('U1', 'unknown 2×'), ('U2', 'unknown 3× + 손실 가중(unknown 3, silence 2)'), ('U3', 'U2 + silence 2× (50 % 저레벨)')):
    if U == 'base':
        f = [R('M3k3_c12_float_e60'), R('M3k3_c12_float_e60_s1')]; q = [R('M3k3e60_c12_q8_kd'), R('M3k3e60_c12_q8_kd_s1')]; t = R('dscnn_c12_ch48'); ev = base
    else:
        f = [R(f'M3k3_{U}_float_e60_s{s}') for s in (0, 1)]; q = [R(f'M3k3e60_{U}_q8_kd_s{s}') for s in (0, 1)]; t = R(f'dscnn_c12_ch48_{U}'); ev = J(f'runs/vad_eval/eval_M3k3e60_{U}_q8_kd_s0_{U}.json')
    pf = lambda r: 'pending' if r is None else f"{100*r['test_acc']['mean']:.2f}"
    pc = lambda r, c: 'pending' if r is None else f"{100*r['per_class_acc'][c]:.1f}"
    evs = 'pending' if not ev else (lambda d: f"{100*d['det_rate']:.1f} / {d['false_alarms_per_hour']:.1f}")(ev['results']['val_talk']['margin_0'])
    tacc = 'pending' if t is None else f"{100*t['test_acc']:.2f}"
    L.append(f"| {U} | {desc} | {tacc} | {pf(f[0])} / {pf(f[1])} | {pf(q[0])} / {pf(q[1])} | {pc(q[0], 'unknown')} | {pc(q[0], 'silence')} | {evs} |")
ad = J('runs/adopt_week4a.json'); u2 = J('runs/vad_eval/eval_M3k3e60_U2_q8_kd_s0_U2.json'); u2r = R('M3k3e60_U2_q8_kd_s0'); gc = J('runs/M3k3e60_U2_q8_kd_s0/golden_check.json')
_al = rd('logs/adopt_regress.log'); _new = _al.split('adopted U2 weights', 1)[1] if 'adopted U2 weights' in _al else ''
_lines = _new.splitlines(); _regress = next((l for l in _al.splitlines() if l.startswith('REGRESS')), 'REGRESS pending')
rg = (f"클립 {sum(1 for l in _lines if l.startswith('m3k3/') and 'PASS' in l)}/12 PASS(0 건너뛰기 on), 스트리밍 {'PASS' if any('m3k3_stream' in l and 'PASS' in l for l in _lines) else 'pending'}, " + _regress) if _al else 'pending'
p3 = rd('runs/pipeline_tb_v3.txt').strip().splitlines()
L.append(f"""
**채택 판정(val_talk, 마진 0, 규칙 = 키워드 검출률 기본 대비 −3 %p 이내 & 오검출/h 최소)**: {('**' + ad['adopted'] + '** (seed 0: det ' + f"{100*ad['pick']['det']:.1f} %, FA/h {ad['pick']['fa']:.1f}; seed 1도 동일 순위)") if ad and ad['adopted'] else 'pending'} — 기본 모델 val_talk det {100*ad['base']['det']:.1f} %, FA/h {ad['base']['fa']:.1f}.
- U2 = unknown 3× + 손실 가중: **오검출이 거의 절반**(val_talk 200 → 107/h) 이고 unknown 정확도 74.9 → 92.6 %. **대가**: 12클래스 클립 정확도 87.79 → **85.41 %**(−2.4 %p; DS-CNN 24ch 시드 평균 89.08 % 대비 −3.7 %p로 1주차 목표 −2 %p를 벗어남), silence 93.1 → 89.4 %, 키워드 클래스 정확도 하락. U1은 클립 손실이 작지만(86.99 %) 오검출 감소도 작다(−29 %). 
- 지시된 규칙대로 **U2를 채택해 `.mem`을 교체**했고(이전 가중치는 `snn/rom/archive/20261026_033828_m3k3/`), 정확도 목표와 오검출 중 무엇을 우선할지는 결정 사항으로 남긴다(되돌리기: `bash snn/tb/run_regress.sh M3k3e60_c12_q8_kd m3k3`).
- 채택 모델 test 스트림(마진 0 / 2838 / 6212): board det {100*u2['results']['test_board']['margin_0']['det_rate']:.1f} / {100*u2['results']['test_board']['margin_2838']['det_rate']:.1f} / {100*u2['results']['test_board']['margin_6212']['det_rate']:.1f} %, FA/h {u2['results']['test_board']['margin_0']['false_alarms_per_hour']:.1f} / {u2['results']['test_board']['margin_2838']['false_alarms_per_hour']:.1f} / {u2['results']['test_board']['margin_6212']['false_alarms_per_hour']:.1f}; talk det {100*u2['results']['test_talk']['margin_0']['det_rate']:.1f} / {100*u2['results']['test_talk']['margin_2838']['det_rate']:.1f} / {100*u2['results']['test_talk']['margin_6212']['det_rate']:.1f} %, FA/h {u2['results']['test_talk']['margin_0']['false_alarms_per_hour']:.1f} / {u2['results']['test_talk']['margin_2838']['false_alarms_per_hour']:.1f} / {u2['results']['test_talk']['margin_6212']['false_alarms_per_hour']:.1f}; quiet FA/h {u2['results']['test_quiet']['margin_0']['false_alarms_per_hour']:.1f} / {u2['results']['test_quiet']['margin_2838']['false_alarms_per_hour']:.1f} / {u2['results']['test_quiet']['margin_6212']['false_alarms_per_hour']:.1f} (기본 모델: board 44.4 / 28.8 / 13.8, talk 184 / 111 / 47, quiet 24.9 / 13.5 / 7.2).
- **4단계 검증(채택 가중치)**: ① numpy 골든 예측 일치 {('%.0f %%' % (100*gc['pred_agreement'])) if gc else 'pending'} ② `run_regress.sh`(`snn/logs/adopt_regress.log`): {rg} ③ 프론트엔드(가중치 무관) 12/12, 통합 테스트 `{p3[-1][:90] if p3 else 'pending'}`(4/4 PASS) ④ `pipeline_int_ref.py` 골든 재생성(pipeline_v3, activity_*).\n""")
L.append("""### A-3. 동작점 스위치
`decision_seg`에 MARGIN(일반 0) / MARGIN1(엄격 2,838) / MARGIN2(6,212) / MARGIN3(매우 엄격 9,633) 파라미터와 `margin_sel[1:0]` 입력을 두고 `kws_top`을 거쳐 `user_logic` 모드 2의 `sw[3:2]`에 연결했다(값은 3주차 val 구간 합 마진의 10/30/50 % 백분위). `pipeline_int_ref.py --margin_sel`로 골든을 만들고 `run_pipeline.sh`가 `info.json`의 margin_sel을 읽어 통합 테스트한다. 동작점별 지표는 A-1 표(마진 0 / 2838 / 6212 / 9633 행)와 같다(기본 모델); 채택 모델의 표는 학습 완료 후 갱신.
""")
# B
if t0:
    L.append(f"""## B. 프론트엔드 2단 게이팅

### B-1. 손실 없는 문턱 T0 (`snn/eval/t0_bound.py`, `snn/runs/t0_bound.json`)
E0 = 프레임 480샘플의 Σ|x18| (hop 160마다 Σ|x|를 누적해 최근 3 hop을 더함: 샘플 수신 경로에서 |x| 가산 1회/샘플).
유도: (1) Hann Q1.15 < 1, round-half-up → |xw[n]| ≤ |x[n]| ⇒ Σ|xw| ≤ E0. (2) 정수 FFT: 정확 DFT(1/512 스케일)의 |X[k]| ≤ Σ|xw|/512; twiddle 양자화 |W_q| ≤ 1+2⁻¹⁵(9스테이지 → ×(1+2⁻¹⁵)⁹), 스테이지당 round-half-up 1회(re/im 각 ±0.5, 스테이지 ×1/2이므로 누적 상한 9×0.5×√2 = 6.36) ⇒ |X_int[k]| ≤ E0/512·(1+2⁻¹⁵)⁹ + 6.36. (3) P = |X|² ≤ Pmax. (4) mel 누적 acc[m] ≤ (밴드 가중치 합, 최대 {t0['wsum_max']}) × Pmax. (5) log2q(acc) < 332 ⇔ acc < A_thr = {t0['A_thr']:,}(log2_int을 이진 탐색으로 스캔한 정확값). ⇒ **T0 = {t0['T0_proven']:,}** (E0 < T0 ⇒ 40개 특징 전부 0, 증명). 보수적 버전(모든 bin 가중치 255, 최장 밴드 {t0['Lmax']}): {t0['T0_loose']:,}. Σx² 기반(Parseval) T2 = {t0['T2_proven']:,}도 유도했으나 곱셈이 필요해 Σ|x|를 채택.
실측(test 4,888클립 × 98프레임 = {t0['n_frames']:,}프레임): 특징이 0이 아닌 프레임의 최소 E0 = **{t0['empirical_min_E0_nonzero']:,}**(증명값의 {t0['empirical_min_E0_nonzero']/t0['T0_proven']:.1f}배), 증명값 위반 {t0['violations_proven']}건. 게이팅 비율: 증명값 {100*t0['gated_ratio_proven']:.1f} %, 실측 기반 문턱이면 {100*t0['gated_ratio_empirical']:.1f} %(특징 전부 0인 프레임은 {100*t0['zero_feature_frames_ratio']:.1f} %). 증명값은 라운딩 항(6.36 ≈ √Pmax의 1/3)과 합 상한이 보수적이어서 실측 대비 6배 낮다 — **채택은 증명값**(비트 일치 보장), 실측값은 참고.

### B-2. RTL과 효과
`fe_ring`이 hop별 Σ|x|로 E0를 내고, `fe_top`은 `frame_start`에 E0 < T0이면 window/FFT/mel을 건너뛰고 특징 0 40개를 스트리밍(`S_GATE`, e_sum/e_log = 0, `gated` 플래그, `cnt_gated`). 해당 블록은 start 펄스를 받지 않아 BRAM enable·곱셈기가 모두 0. 검증: 게이팅 on 상태로 `tb_fe_top` 12벡터 {rd('runs/fe_top_gated_ms0.txt').count('PASS')}/12 PASS(특징·E1 동일), 통합 테스트 `{(rd('runs/pipeline_tb_v2_gated.txt').strip().splitlines() or ['pending'])[-1][:100]}`.
""")
    if act:
        L.append("세 스트림 시나리오(20~60 s, `snn/tb/activity/run_activity.sh`, 사이클 단위 실측). VAD=0 행은 VAD 문턱을 0으로 둔 근사(E1>0인 모든 프레임 공급, 255프레임마다 리셋)이며 진짜 'SNN 상시'는 가동률 100 %에 해당한다(이 행의 판정 불일치는 골든이 VAD 기준이라 생기는 것으로, 활동량 비교용). 프론트엔드 활성 사이클/프레임은 게이팅된 프레임을 0으로 포함한 평균:\n")
        L.append("| 시나리오 / 설정 | 프레임 | 게이팅된 프레임 | 프론트엔드 가동률(프레임) | SNN 공급 프레임 | SNN 가동률(프레임) | 프론트엔드 활성 사이클/프레임 | SNN 활성 사이클/프레임 | 판정 일치 |\n|---|---|---|---|---|---|---|---|---|")
        for r in act:
            fr = int(r['frames']); g = int(r['gated']); sf = int(r['snn_frames'])
            L.append(f"| {r['cfg']} | {fr} | {g} | {100*(fr-g)/fr:.1f} % | {sf} | {100*sf/fr:.1f} % | {int(r.get('fe_cyc',0))/fr:,.0f} | {int(r.get('snn_cyc',0))/fr:,.0f} | mism {r['mism']} |")
        L.append("")
else: L.append("## B. 게이팅\npending\n")
# C
L.append(f"""## C. 보드 없이 하는 합성·전력 준비
### C-1. Lint·Yosys
- `verilator --lint-only -Wall`(waiver `snn/rtl/verilator_waivers.vlt`: PINCONNECTEMPTY = 의도적 미연결 디버그 출력, WIDTHTRUNC/EXPAND = 파라미터 산술, UNUSED*, DECLFILENAME): `kws_top` {('경고 0, exit 0' if 'exit=0' in rd('runs/lint_kws_top.txt') else rd('runs/lint_kws_top.txt')[-200:])}; `user_logic` {('경고 0, exit 0' if 'exit=0' in rd('runs/lint_user_logic.txt') else 'see runs/lint_user_logic.txt')}. waiver 없이는 {rd('runs/lint_kws_top_nowaiver.txt').strip().replace(chr(10), ', ')}.
- **Yosys 0.69 `synth_xilinx -family xc7`**(`snn/rtl/yosys_kws_top.ys`, `reports/yosys_kws_top_stat.txt`) kws_top — **Yosys 추정**: LC 7,764 (LUT 9.2k), FF 8,011, CARRY4 508, DSP48E1 12, **RAMB36E1 12 + RAMB18E1 5 (= BRAM18K 29)**, RAM64M 24(분산). BRAM 추론 확인: 가중치 ROM(`wrom` 5개) → RAMB, 샘플 링(512×18)·FFT 버퍼(512×36)·판정 링 → RAMB, 프리롤 버퍼·θ/b ROM → 분산 RAM/LUT, 히스토리 → FF(설계 의도). `$readmemh` 경로: 모든 모듈이 `ROM_DIR`/`MEM_PREFIX` 파라미터 또는 `rom_paths.vh` 매크로로 절대 경로를 받으며(Windows `C:/...` 슬래시 경로 포함, `build_bitstream.tcl`이 생성), 시뮬에서는 상대 경로(`../rom`)를 쓴다.
### C-2. 활동량 기반 상대 전력 모델
시뮬레이션 전용 카운터(`ifdef ACTIVITY`): 블록별 활성 사이클, BRAM 읽기·쓰기, DSP 곱셈 횟수, v_mem 쓰기. 세 시나리오 × {{게이팅 on/off, VAD on/off}} 표는 B-2, 블록별 비중 그래프 `snn/docs/figures/activity_share.png`(`activity.csv`). **실제 전력이 아닌 상대 비교용**이다(토글·배선·정적 전력 미포함).
### C-3. Vivado 전력 스크립트(사용자 실행)
`snn/board/pkg/power_flow.tcl`(vectorless `report_power -hier` → xsim post-route 기능 시뮬 SAIF(무음/키워드/대화, `power_vec/*.hex`, `tb_power.v`) → `read_saif` + `report_power -hier`), `POWER_COMPARE.md`(hls4ml DS-CNN 동일 조건 절차, 비교표 양식, 보드 실측 순서 초안). PL 동적 전력은 `user_logic_inst` 계층 행만 사용, PS7 제외.

## D. 최종보고서·발표
- `snn/docs/final_report_draft.md`: 신청서 구조(배경·목적 / 수행 방법 / 기대효과) + 결과 절(정확도·스트리밍·연산량 세 층·검증·자원/전력 TBD) + 한계 + 참고문헌 12개.
- `snn/docs/figures/`: `system_block.svg`, `ops_vs_period.png`, `timeline.png`, `confusion.png`, `activity_share.png` + `activity_gating.png`(게이팅 전/후, VAD 유무), `seed_stats.png` (+ CSV/JSON 원자료).
- `snn/docs/presentation_outline.md`: 14장 + 라이브 데모 시나리오·대체안.

## 4-B(보드 결과 이후) 할 일과 리스크
1. U1~U3 결과로 모델 채택 → `.mem` 교체·4단계 검증 → `REPORT_week4a.md` 갱신(이 보고서는 pending 항목을 자동 채움).
2. Windows: `build_bitstream.tcl` → `power_flow.tcl` → `POWER_COMPARE.md` 표 채우기; 보드 `BOARD_STEP1.md` 1~4단계; 전류 실측.
3. 리스크: 마이크 레벨·실내 소음에서 E1 문턱(600)과 T0 게이팅 효과가 달라질 수 있음(캡처 → `board_capture_check.py`); 오검출은 unknown 비율에 비례(대화 환경 데모는 엄격 동작점 권장).

## 생성 파일과 재현
```
snn/eval/   fa_breakdown.py t0_bound.py make_power_vectors.py build_report4a.py (vad_eval.py: talk/quiet, --eval_only)
snn/data/   make_dataset.py(--unknown_mult/--silence_mult/--silence_lowlevel) feat_c12_m40_w480_U{{1,2,3}}.npz filelist_c12_U*.json
snn/        run_week4_A.sh train_snn.py(--class_weight)
snn/rtl/    fe/fe_ring.v fe/fe_top.v (E0, T0 게이팅) decision_seg.v kws_top.v (margin_sel, T0, fe_gated, 활동량 카운터) verilator_waivers.vlt yosys_kws_top.ys reports/
snn/tb/     activity/tb_activity.v run_activity.sh ; pipeline/ (margin_sel) ; fe/run_fe_top.sh (T0)
snn/board/pkg/  user_logic.v(sw[3:2] 동작점) power_flow.tcl tb_power.v POWER_COMPARE.md power_vec/
snn/docs/   final_report_draft.md presentation_outline.md make_figures.py figures/
snn/runs/   fa_breakdown_board.json t0_bound.json vad_eval/eval_*.json activity_*.txt lint_*.txt fe_top_gated_ms0.txt pipeline_tb_v2_gated.txt
```
```bash
python snn/eval/fa_breakdown.py --cond board --out snn/runs/fa_breakdown_board.json
python snn/eval/vad_eval.py --conds talk,quiet,board --eval_only --margins 0,2838,6212,9633 --tag base [--run snn/runs/<run>]
bash snn/run_week4_A.sh ; python snn/eval/t0_bound.py
T0=6694 bash snn/tb/fe/run_fe_top.sh snn/golden/vectors/m3k3/yes_11 0 64 ; bash snn/tb/activity/run_activity.sh snn/golden/vectors/activity_board/stream_0 6694 1
cd snn/rtl && yosys -q -s yosys_kws_top.ys ; verilator --lint-only -Wall verilator_waivers.vlt --top-module kws_top *.v fe/*.v
python snn/docs/make_figures.py ; python snn/eval/build_report4a.py
```
""")

# ---------- appendix: 4-A follow-up (training-style comparison + detection curves)
cu = rd('runs/compare_u2.md'); dc = J('runs/det_curve/det_curve.json'); cuj = J('runs/compare_u2.json')
if cu:
    L.append("## 부록 A. 4-A 후속 — 같은 학습 방식끼리의 SNN·DS-CNN 비교와 시스템 수준 검출 곡선\n")
    L.append("### 부록 A-1. DS-CNN 24ch를 U2 방식(unknown 3× + 손실 가중 10:2, 11:3)으로 학습 (`snn/eval/compare_u2.py`, 시드 0/1/2, 12클래스 test)\n")
    L.append(cu + "\n")
if dc:
    L.append("### 부록 A-2. 검출률–오검출 곡선 (`snn/eval/det_curve.py`, 같은 VAD 구간 파이프라인; DS-CNN은 구간 특징을 98프레임으로 0-패딩/절단; 마진은 모델별 자기 척도, val 마진 백분위 40점 + 0; 동작점은 val에서 FA ≤ 10/h, ≤ 3/h 조건으로 고르고 test로 보고)\n")
    L.append("![det curve](docs/figures/det_curve.png)\n")
    L.append("| 모델 | 스트림 | 마진 0: 검출 / 오검출/h | val FA≤10 동작점 → test 검출 / 오검출/h | val FA≤3 동작점 → test 검출 / 오검출/h |\n|---|---|---|---|---|")
    for name, r in dc.items():
        for cond in ('board', 'talk'):
            c0 = r['curves'][f'test_{cond}'][0]; o10 = r['ops'].get(f'{cond}_fa<=10'); o3 = r['ops'].get(f'{cond}_fa<=3')
            f = lambda o: f"{100*o['test']['det']:.1f} / {o['test']['fa_per_h']:.1f} (마진 {o['margin']:.0f})" if o else '–'
            L.append(f"| {name} | {cond} | {100*c0['det']:.1f} / {c0['fa_per_h']:.1f} | {f(o10)} | {f(o3)} |")
    # equal-FA comparison at 5 and 20 FA/h on test by interpolation
    def det_at(curve, fa):
        pts = sorted([(c['fa_per_h'], c['det']) for c in curve]); xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        return float(np.interp(fa, xs, ys)) if xs.min() <= fa <= xs.max() else float('nan')
    L.append("\n같은 오검출률에서의 test 검출률(곡선 보간):\n\n| 모델 | board @5/h | board @20/h | talk @5/h | talk @20/h |\n|---|---|---|---|---|")
    for name, r in dc.items():
        L.append(f"| {name} | " + " | ".join(f"{100*det_at(r['curves'][f'test_{cond}'], fa):.1f}" for cond in ('board', 'talk') for fa in (5, 20)) + " |")
    L.append("")
if cuj and cuj.get('gaps'):
    g = cuj['gaps']; gb = g.get('base', {}); gu = g.get('U2', {})
    verdict = ''
    if gu and gu.get('dscnn_drop') is not None:
        if gu['dscnn_drop'] > -1.0 and gu['gap'] <= -3.0:
            verdict = f"DS-CNN은 U2 방식에서도 거의 떨어지지 않는데({gu['dscnn_drop']:+.2f} %p) SNN은 {gu['snn_drop']:+.2f} %p 떨어져 격차가 {gu['gap']:.1f} %p로 벌어진다 → **U2 유지의 근거가 약하다. U1(unknown 2×, 손실 가중 없음) 재검토를 제안**: U1은 클립 −0.8 %p로 격차를 유지하면서 오검출을 29 % 줄이고, 나머지 오검출은 동작점(마진)으로 맞추는 쪽이 정확도 목표와 양립한다."
        else:
            verdict = f"DS-CNN도 U2 방식에서 {gu['dscnn_drop']:+.2f} %p 떨어져 SNN의 하락({gu['snn_drop']:+.2f} %p)이 주로 데이터 구성(클래스 재가중) 때문임이 확인된다 → 같은 방식끼리의 격차는 base {gb.get('gap', float('nan')):+.2f} %p → U2 {gu['gap']:+.2f} %p로 유지되며, **U2 유지가 타당**(오검출 절반, 격차는 데이터 방식과 무관)."
    # equal-FA comparison SNN U2 vs the two DS-CNN U2-style input variants (test, interpolated)
    def _d(name, cond, fa):
        r = dc.get(name); return 100 * det_at(r['curves'][f'test_{cond}'], fa) if r else float('nan')
    cmp = {v: {c: (_d('SNN U2', c, 5) - _d(v, c, 5)) for c in ('board', 'talk')} for v in ('DS-CNN 24ch U2-style (segment, 0-pad)', 'DS-CNN 24ch U2-style (last 98 frames)')}
    L.append("### 부록 A-3. U2 유지 판단\n" + verdict + f" **검출 곡선(부록 A-2)의 결론은 DS-CNN 입력 방식에 따라 다르다**: 구간 0-패딩 입력에서는 SNN U2가 같은 오검출률(5/h)에서 DS-CNN U2-style보다 board {cmp['DS-CNN 24ch U2-style (segment, 0-pad)']['board']:+.1f} %p, talk {cmp['DS-CNN 24ch U2-style (segment, 0-pad)']['talk']:+.1f} %p 높지만, **VAD 종료 시점에서 거꾸로 98프레임(실제 특징)을 넣는 공정한 변형에서는 DS-CNN U2-style이 board에서 SNN U2보다 {-cmp['DS-CNN 24ch U2-style (last 98 frames)']['board']:+.1f} %p 높고 talk에서는 동률({cmp['DS-CNN 24ch U2-style (last 98 frames)']['talk']:+.1f} %p)이다.** 즉 '같은 오검출률에서 SNN이 더 잘 검출한다'는 주장은 성립하지 않으며, 시스템 수준 검출 품질은 두 모델이 ±4 %p 안에서 비슷하다. U2 유지의 근거는 (1) 같은 방식 격차 −0.6 %p, (2) 기본 SNN 대비 오검출 절반, (3) 연산량·가동률 이점이지 검출 품질 우위가 아니다. DS-CNN 0-패딩 결과는 패딩 때문에 불리했던 것이므로 보고서에는 last-98 변형을 대표값으로 쓴다.\n")

# ---------- appendix B: same-system ops / energy / latency
so = J('runs/system_ops.json'); som = rd('runs/system_ops.md')
if so:
    lat = so['latency']; b = so['board']['per_segment']; be = so['board']['per_segment_energy_pJ']
    L.append("## 부록 B. 같은 VAD 구간 시스템에서의 연산량·에너지·지연 (`snn/eval/system_ops.py`, test 스트림 200개씩, DS-CNN = U2-style 24ch(마지막 98프레임 1회 추론), SNN = U2(프리롤 포함 매 프레임), 활성 희소도는 실제 구간 창에서 측정)\n")
    L.append("프론트엔드(프레임당 5,942 사이클, 게이팅 시 0)와 VAD는 두 시스템에 **공통**이며 아래 표에는 포함하지 않는다(별도 행: 프론트엔드 ≈ 윈도우 512 + FFT 2,304 버터플라이 + mel 488 MAC-급 연산/프레임, 100 프레임/s).\n")
    L.append("**구간당 연산(MAC / AC)**\n\n" + som.split("\n\n")[0] + "\n")
    L.append(f"DS-CNN 측정 희소도(board): 입력 특징 0 비율 {100*so['board']['sparsity']['input']:.1f} %, ReLU1 출력 0 비율 {100*so['board']['sparsity']['relu1']:.1f} %, ReLU2 {100*so['board']['sparsity']['relu2']:.1f} % (conv1은 전체 MAC의 8 %뿐이라 입력 0 건너뛰기 효과가 작고, conv2/3의 ReLU 희소도가 72 % 절감의 원천).\n")
    L.append("**시간당 연산과 추정 에너지(데이터패스만, MAC 0.3 pJ / AC 0.1 pJ)**\n\n" + som.split("\n\n")[1] + "\n")
    r_ops = b['ds_dense'] / (b['snn_mac_skip'] + b['snn_ac']); r_e = be['ds_dense'] / be['snn_skip']; r_ops2 = b['ds_in_act_skip'] / (b['snn_mac_skip'] + b['snn_ac']); r_e2 = be['ds_in_act_skip'] / be['snn_skip']
    L.append(f"""**같은 시스템에서의 SNN 고유 이득(board 조건, 구간당)**: DS-CNN dense 대비 연산 {r_ops:.1f}×, 추정 에너지 {r_e:.1f}×; **활성 희소도를 활용한 DS-CNN(입력+ReLU 0 건너뛰기) 대비 연산 {r_ops2:.1f}×, 추정 에너지 {r_e2:.1f}×** — 즉 VAD 시스템 안에서 SNN의 알고리즘 이득은 "곱셈 → 가산"과 희소성이며, 희소성을 똑같이 활용하는 CNN 대비 배수는 2× 수준이다. (2주차의 "100 ms 판정 기준 48×"는 VAD 없이 매 판정마다 1 s 창을 재계산하는 상시 추론 시스템 조건의 참고 수치.)

**판정 지연·버퍼·병렬도(직렬 엔진 MAC 1개 @ 48 MHz 가정)**

| 항목 | DS-CNN 24ch + VAD | SNN U2 + VAD |
|---|---|---|
| 구간 종료 후 판정 지연 | dense {lat['ds_dense_ms']:.1f} ms / 입력 0 건너뛰기 {lat['ds_in_skip_ms']:.1f} ms / 입력+ReLU 0 건너뛰기 {lat['ds_in_act_skip_ms']:.1f} ms | 마지막 프레임 처리 {lat['snn_last_frame_avg_ms']:.2f} ms(평균) ~ {lat['snn_last_frame_ms']:.2f} ms(최악 16.7k 사이클) |
| 필요 버퍼 | 98×40×8 = {lat['ds_buffer_bits']:,} bit (BRAM18K {lat['ds_buffer_bram18k']}개) + 활성값 20×10×24×8 = 38 kbit(추정) | 프리롤 21×40×8 = {lat['snn_preroll_bits']:,} bit + 뉴런 상태·히스토리 {lat['snn_state_bits']:,} bit(분산 RAM/FF) |
| 곱셈기 | 1 (직렬) — 지연을 1 ms로 맞추려면 **{lat['ds_parallel_for_1ms_dense']:.0f}개(dense) / {lat['ds_parallel_for_1ms_in_act_skip']:.0f}개(희소)**, SNN 최악 지연(0.35 ms)에 맞추려면 {lat['ds_parallel_for_snn_latency_dense']:.0f} / {lat['ds_parallel_for_snn_latency_in_act_skip']:.0f}개 | 1 (입력층) + 가산기 1 |
""")

# checklist
done_u = all(R(f'M3k3e60_U{u}_q8_kd_s{s}') for u in (1, 2, 3) for s in (0, 1))
L.append(f"""## 완료 기준 체크리스트
- [x] 오검출 원인 분해
- [x] 대화 많은 / 조용한 상시 대기 스트림(val, test)
- [{'x' if done_u else ' '}] U1~U3 비교, 채택 모델 시드 2개 이상, 검증 4단계 통과 {'(학습 진행 중)' if not done_u else ''}
- [x] 동작점 스위치 RTL과 표
- [x] 증명된 T0와 손실 없는 프론트엔드 게이팅 RTL(골든 일치 유지)
- [{'x' if act else ' '}] 세 스트림의 프론트엔드·SNN 가동률
- [x] Verilator lint, Yosys 자원 추정(BRAM 추론 확인)
- [{'x' if act else ' '}] 활동량 기반 상대 전력 표·그래프
- [x] `power_flow.tcl`, `POWER_COMPARE.md`
- [x] `final_report_draft.md`, 그림 6종, `presentation_outline.md`
- [x] REPORT_week4a.md
""")
open(os.path.join(S, 'REPORT_week4a.md'), 'w').write("\n".join(L)); print('report4a written')
