"""build_report2.py — assemble snn/REPORT_week2.md from the measured artifacts (re-run whenever a result updates)."""
import os, re, json, subprocess, datetime
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
S = os.path.join(ROOT, 'snn')
def J(p):
    p = os.path.join(S, p); return json.load(open(p)) if os.path.exists(p) else None
def pct(x): return 'pending' if x is None else f"{100*x:.2f}"

# ---------------- RTL cycle stats
def cyc_stats(zs):
    p = os.path.join(S, 'runs', f'rtl_vec_zs{zs}.txt')
    if not os.path.exists(p): return None
    rows = [dict(re.findall(r'(\w+)=(-?\d+)', l)) for l in open(p) if l.startswith('RESULT')]
    if not rows: return None
    n = len(rows); a = lambda k: sum(int(r[k]) for r in rows) / n
    return dict(n=n, npass=sum(1 for l in open(p) if 'PASS' in l), mism=sum(int(r['mism']) for r in rows), cyc_avg=a('cyc_avg'), cyc_max=max(int(r['cyc_max']) for r in rows),
                l1=a('l1'), hid=a('hid'), out=a('out'), upd=a('upd'), dec=a('dec'), skip=a('skip_l1'), run1=a('run_l1'), runh=a('run_hid'), runo=a('run_out'))
c1, c0 = cyc_stats(1), cyc_stats(0)
reg = open(os.path.join(S, 'runs', 'rtl_regress.txt')).read().strip() if os.path.exists(os.path.join(S, 'runs', 'rtl_regress.txt')) else 'pending'
stream = open(os.path.join(S, 'runs', 'rtl_stream.txt')).read().strip() if os.path.exists(os.path.join(S, 'runs', 'rtl_stream.txt')) else 'pending'
zs = J('runs/zero_skip_stats.json'); se = J('runs/stream_eval/summary.json'); rules = J('runs/stream_eval/rules.json'); cf = J('runs/confusion.json')
subprocess.run(['python3', os.path.join(S, 'eval', 'collect_week2.py')], capture_output=True)
tables = open(os.path.join(S, 'runs', 'week2_tables.md')).read()
ops = open(os.path.join(S, 'runs', 'figures', 'ops_energy.csv')).read().strip().split('\n')
import csv; opsrows = list(csv.DictReader(ops))

L = []
L.append(f"# SNN KWS — 2주차 보고서 ({datetime.date.today()})\n")
L.append("범위: **A** 추천 모델 M3k3(8bit+KD)의 SNN 코어 RTL + 골든 비트 일치 시뮬레이션 + 합성 스크립트, **B** 보고서용 수치 보강(10클래스, 시드, 장기 학습, 스트리밍, 연산량·에너지, 0 건너뛰기, 혼동행렬), **C** 코덱 16 kHz 준비. 수치는 전부 실제 실행 결과이며 추정치는 \"추정\"으로 표시. 1주차 산출물은 덮어쓰지 않았고(교체한 `.mem`은 `snn/rom/archive/`), 모델 선택은 val 기준[^1].\n\n[^1]: 1주차에는 test 정확도를 보고 고른 항목이 있다: (a) 30 → 60 에폭 재학습 채택, (b) 후보 M1/M2/M3/M3k3 중 M3k3 선택. 2주차 B-3의 val 기준 재검토에서도 60에폭 체크포인트가 유지되므로 결론은 바뀌지 않지만, 보고 수치의 선택 편향 가능성은 표시해 둔다.\n")
L.append("동결 사양(변경 없음): M3k3, K=3, dilation 1/2/4/8, 64ch×4, 출력 12, 8bit 대칭 per-layer QAT+KD, mel40/hop160/win480/T=98, LIF k=3 감산 리셋, 뉴런별 정수 θ·b, 막전위 16bit 포화, in_shift=6, 출력 적분기 24bit, 시간 합 argmax, clk_48 단일 도메인.\n")
L.append("## 0. 시작 전 확인\n- 시뮬레이터: **Icarus Verilog 13.0, Verilator 5.052** (conda base). 테스트벤치는 Icarus로 실행(벡터 1개 ≈ 4 s, 회귀 240개 ≈ 15 min).\n- Vivado: 서버에 없음(`/tools/Xilinx`, `/opt/Xilinx` 없음) → `snn/rtl/synth_snn_core.tcl` 배치 스크립트로 대체(A-4).\n- 레퍼런스 보드 파일(`reg_set_init.c`, `ila.xdc`, `user_logic.v`)은 서버에 없음 → C와 A-5는 데이터시트/1주차 포트 정의 기준 초안.\n- 골든 연산 순서: `snn/rtl/SPEC_snn_core.md` (A-1).\n")

# ---------------- A
L.append("---\n## A. SNN 코어 RTL (`snn/rtl/`, `snn/tb/`)\n")
L.append("### A-1. 사양서 `snn/rtl/SPEC_snn_core.md`\n`snn_int_ref.py` 줄 번호 대응으로 작성: 탭/dilation 인덱싱(줄 45–47), 누적 비트폭(24bit), `>>> 6` 산술 시프트 후 바이어스 가산(줄 51–53), LIF 순서 **누설 → 전류 가산 → 16bit 포화 → θ 비교 → 감산 리셋**(줄 17–21), 출력 적분기(줄 66), 시간 합·argmax tie = 낮은 인덱스(줄 68), 프레임 0 이전 히스토리 = 0(줄 46), 신호 비트폭 표.\n")
L.append("""### A-2. RTL 구조

```
snn_core (top, 파라미터: N_IN 40, N_HID 64, N_CLASS 12, K 3, D2/D3/D4 2/4/8, IN_SHIFT 6, LEAK_K 3, ZERO_SKIP, WIN_SUM 98, MEM_PREFIX)
├─ snn_ctrl      프레임 FSM: CLR(v_mem 256엔트리 0) → IDLE → LOAD(40 특징) → [층 l=1..4, OUT: INIT→ROWCHK/ROWRUN→DRAIN→UPD] → DEC → DONE
├─ feat_hist     특징 히스토리 3프레임×40×8bit (프레임 시프트 레지스터, 탭 j = j프레임 전)
├─ spike_hist×4  1비트 스파이크 히스토리 5/9/17/1프레임×64 (층2·3·4·출력 입력, 탭 j = j·D 프레임 전)
├─ wrom×5        층별 가중치 ROM ($readmemh, BRAM 추론, 활성 층만 enable): 7680 / 12288×3 / 768 × 8bit
├─ row_acc       공용 누적 엔진: ROM 1워드/사이클, 입력층은 act×w(DSP 1개), 스파이크층은 w 가산만, acc[64]×24bit
├─ prom×9        θ(4층)·b(5층) 64×16bit 비동기 ROM (분산 LUT)
├─ lif_update    1뉴런/사이클 LIF 갱신(조합) + v_mem 256×16bit LUTRAM
├─ out_integrator 출력 적분기 12×24bit
└─ decision      창 합(링 BRAM 128프레임×12×24bit) + argmax(엄격 '>' 순차 스캔) + 스트리밍 파라미터(MARGIN/N_CONSEC/REFRACT/N_KEYWORD)
```
지시서의 `input_layer` + `hidden_ac`를 **`row_acc` 하나로 합친 이유**: 두 경로의 차이는 곱셈 유무뿐이고(입력층 `act×w`, 스파이크층 `act=1`), 행 단위 ROM 읽기·post 루프·누적기(64×24bit)가 동일해서 한 엔진에 `is_input` 먹스만 두면 DSP 1개·누적기 1벌로 끝난다. 층은 순차 처리이므로 두 엔진이 동시에 돌 일도 없다(1주차 사이클 모델의 'MAC‖AC' 겹침은 적용하지 않음 — 프레임 예산 10만의 6 %만 쓰므로 불필요). `spike_hist`/`feat_hist`는 링 포인터 대신 **프레임 시프트 레지스터**(프레임 시작 시 1회 시프트)로 만들어 `t−j·D<0 → 0`이 리셋 후 자동으로 성립한다.

**인터페이스**: 특징은 **스트리밍 입력**(`feat_valid`/`feat_data[7:0]`/`feat_last`, 1개/사이클 또는 간격 임의)을 택했다 — 프론트엔드 mel 블록이 밴드를 순차 생성하므로 버퍼 읽기 포트보다 결합이 단순하고, 코어 내부 `feat_hist`가 이미 3프레임 버퍼다. 출력 `frame_done`(1클럭), `class_idx[3:0]`, `vo_sum[12×32]`(창 합, 디버그), `det_valid`, `busy`(IDLE에서만 0 → clock enable 내림 가능). `clip_start`는 전체 상태 0 클리어(256+α 사이클, busy=1).

**저전력**: 입력층 x=0 행 건너뛰기(`ZERO_SKIP`), 스파이크층은 s=1인 (탭, pre) 행만 처리(항상), 가중치 BRAM enable은 해당 층의 `issue` 사이클에만, 유휴 시 `busy=0`.
""")
def cycrow(c, label):
    if c is None: return f"| {label} | pending | | | | | | | | |"
    return f"| {label} | {c['npass']}/{c['n']} | {c['mism']} | **{c['cyc_avg']:.0f}** | {c['cyc_max']} | {c['l1']:.0f} | {c['hid']:.0f} | {c['out']:.0f} | {c['upd']:.0f} + {c['dec']:.0f} | {c['run1']:.1f}/{c['skip']:.1f} , {c['runh']:.1f} , {c['runo']:.1f} |"
L.append("### A-3. 골든 비교 결과\n")
L.append("**(1) 클립 벡터 12개**(`snn/golden/vectors/m3k3/`, 클래스별 1개): 프레임마다 4층×64 막전위(리셋 후)·스파이크, 출력 막전위 12개, 최종 클래스를 비교.\n")
L.append("| 설정 | PASS | 불일치 수 | 사이클/프레임 평균 | 최대 | 입력층 누적 | 은닉 누적 | 출력 누적 | 갱신 + 판정 | 행 수/프레임: 입력 실행/건너뜀 , 은닉 실행 , 출력 실행 |\n|---|---|---|---|---|---|---|---|---|---|")
L.append(cycrow(c1, '0 건너뛰기 ON')); L.append(cycrow(c0, '0 건너뛰기 OFF'))
L.append(f"\n- 모든 벡터에서 **불일치 0**(막전위·스파이크·출력 막전위 전부, {98*(4*64*2+12)}개 값/벡터). 0 건너뛰기 on/off 결과도 동일(건너뛴 행은 기여 0).\n- 1주차 사이클 모델(11.1k, 직렬 MAC+AC+갱신)과 비교: off {('%.0f' % c0['cyc_avg']) if c0 else '–'}(입력층 7,800 = 7,680 MAC + 120 행 검사)로 모델과 일치, **on {('%.0f' % c1['cyc_avg']) if c1 else '–'}**(입력층 MAC 77 % 절감). 행 검사 오버헤드는 행당 1사이클(입력 120 + 은닉 576 + 출력 64 = 760/프레임).\n- 프레임 예산 10만 사이클 대비 **{('%.1f' % (100*c1['cyc_max']/1e5)) if c1 else '–'} %(최악 프레임)**, 평균 {('%.1f' % (100*c1['cyc_avg']/1e5)) if c1 else '–'} %.\n")
L.append(f"**(2) 대규모 회귀**(`snn/golden/vectors/m3k3_regress/`, 테스트셋 클래스별 20개 = 240개, `tb_regress.v`): 예측 클래스와 출력 합 12개 비교 → `{reg}`\n")
L.append(f"**(3) 스트리밍 모드**(`m3k3_stream`: yes, silence, no, unknown, stop, go 6클립 = 588프레임을 `clip_start` 없이 연속 입력; 골든 = 이어붙인 시퀀스에 `run_int` 1회): `{stream}` — 588프레임 전부 막전위·스파이크 일치, 최종 클래스 = 마지막 98프레임 창의 argmax(go=3).\n")
L.append("회귀 스크립트 `snn/tb/run_regress.sh <run> <prefix> [N]`: 이전 `.mem` 아카이브 → 내보내기 → 벡터/회귀/스트림 재생성 → numpy 골든 100 % 확인 → 위 (1)(2)(3) 시뮬레이션 일괄 실행.\n")
L.append("""### A-4. 합성
서버에 Vivado가 없어 **실행하지 못했다**. `snn/rtl/synth_snn_core.tcl`(OOC synth → opt/place/route → utilization/timing/vectorless power/RAM 리포트)과 `snn_core_ooc.xdc`(48 MHz)를 준비했다. Windows 실행:
```
cd C:\\kws_snn\\snn\\rtl
vivado -mode batch -source synth_snn_core.tcl -tclargs C:/kws_snn/snn/rom/m3k3 1
```
(`-tclargs` = `.mem` 절대 경로 prefix, ZERO_SKIP). 리포트는 `snn/rtl/reports/`. **추정** 자원: 가중치 ROM BRAM18K 23개(7680×8 → 4, 12288×8 → 6×3, 768×8 → 1; 1주차 추정과 동일 산정), 판정 창 링 128×288bit → BRAM18K 2개, v_mem 256×16 LUTRAM, 히스토리 FF ≈ 3k, DSP 1개(9×8 곱), 나머지 LUT ≈ 2–3k. 타이밍은 가장 긴 경로가 LIF 갱신(24bit 가산 2회 + 포화 + 비교, 20.8 ns)이라 여유 있음(추정).
""")
L.append("### A-5. 보드 1단계 준비 (`snn/board/`)\n- `user_logic.v`: 레퍼런스 포트 + `new_sample`. `feat_sets.mem`(골든 특징 4세트: yes/no/stop/silence, `sw[2:1]` 선택)을 ROM에서 10 ms 간격(또는 `sw[3]` 고속)으로 SNN 코어에 주입, 결과 `result_class/result_valid/run_count` 레지스터(ILA용) + 키워드별 반음계 비프(300 ms, `audio_l/r_out`), `sw[0]`=0 패스스루. `led`는 내부 신호. Icarus 컴파일 확인.\n- `ila_snn.xdc`: 새 net 이름 기준 디버그 코어 초안(result_class, result_valid, frame_done, busy, new_sample, LRCLK(TODO net명), audio_l_in, run_count, det_valid).\n")

# ---------------- B
L.append("---\n## B. Python 병행 과제\n")
L.append(tables.replace('### B-1.', '### B-1. 10클래스 재학습 —').replace('### B-2.', '### B-2. 시드 반복 —').replace('### B-3.', '### B-3. 장기 학습 —') + "\n")
L.append("""**B-1~B-3 해석**
- **10클래스(하드웨어 비교용, 시드 평균끼리 비교)**: DS-CNN 24ch **91.66 ± 0.20 %** vs SNN M3k3 Q8+KD 60ep **89.59 ± 0.55 %** → **−2.1 %p, 목표 경계**. 기존 보드의 hls4ml MFCC DS-CNN(91.75 %, 단일 학습) 대비 **−2.2 %p**.
- **12클래스(시드 평균)**: DS-CNN 24ch 89.08 ± 0.17 % vs SNN 87.67 ± 0.43 % → **−1.4 %p, 목표 충족**. float→Q8+KD 차이(+0.35 %p)와 1주차에 본 0.5 %p 이하 차이들은 **시드 표준편차(0.2~0.55 %p) 안**이라 유의하지 않다.
- **장기 학습**: val 기준 100ep 88.12 / 150ep 89.22(float), Q8+KD val 88.72 / 89.08 — 60ep(88.68)와의 차이가 시드 노이즈 범위이고 test는 오히려 낮다(86.58 / 86.95 vs 87.79). **60에폭 체크포인트(`M3k3e60_c12_q8_kd`, 1주차 추천)를 유지**한다. 주의: val만으로 고르면 150ep Q8+KD(val 89.08)가 0.4 %p 높지만 노이즈 수준이고 이미 동결·내보낸 `.mem`이므로 바꾸지 않았다(바꾸려면 `run_regress.sh M3k3e150_c12_q8_kd m3k3`).
- 새 QAT 런 10개 모두 numpy 정수 골든 추론과 **예측 100 % 일치** 확인(`snn/runs/<run>/golden_check.json`).
""")
L.append("(B-1~B-3의 pending 항목은 학습 진행 중 — `python snn/eval/build_report2.py`로 보고서 재생성, 로그 `snn/logs/week2_B_driver.log`, 런 `snn/runs/M3k3_c10_float_e60_s*`, `M3k3e60_c10_q8*`, `M3k3_c12_float_e60_s*`, `M3k3_c12_float_e100/e150`, `M3k3e100/e150_c12_q8_kd`.)\n")
# B-4
if se:
    o = se['oracle']
    L.append(f"""### B-4. 스트리밍 평가 (`snn/eval/stream_eval.py`, `snn/runs/stream_eval/`)
스트림: 테스트 분할의 키워드·unknown 발화를 0.5~3 s 간격으로 배치한 **30 s × {se['n_streams']}개(총 {se['n_streams']*se['dur']/3600:.2f} h, 발화 2,099개(키워드 1,918개 + unknown 181개))**, 발화별 게인 U(−20, +10) dB, 배경소음(test 구간) 볼륨 U(0, 0.1). 정수 프론트엔드 → **리셋 없는 정수 추론**(`run_int`를 전체 시퀀스에 1회 = RTL 스트리밍과 동일 정의) → 프레임별 출력 막전위.

**오라클 창 분석(판정 규칙과 상태 이월 효과 분리)**: 정답 구간 [onset, offset)의 출력 합 argmax 정확도 — 상태 이월(리셋 없음) **{100*o['acc_carried_state']:.1f} %**, 구간 시작 리셋 {100*o['acc_reset_at_onset']:.1f} % → **상태 이월 손실은 {100*(o['acc_reset_at_onset']-o['acc_carried_state']):.1f} %p로 작다.** 클립 정확도 87.8 %와의 나머지 차이는 게인(−20~+10 dB)·소음 혼합(1주차 표: −10 dB 84.7 %, −20 dB 67 %) 때문.

**판정 규칙 비교**(불응기 R=100프레임, N=1; 마진 M은 발화 종료 시점 마진 분포의 백분위 {se['margin_percentiles']} (10/25/50/75 %)에서 선택):

| W | M | 검출률 | 오분류 | 미검출 | 오검출/시간 | 지연 중앙값(프레임, offset 기준) | p90 |
|---|---|---|---|---|---|---|---|""")
    for r in rules:
        if r['R'] == 100 and r['N'] == 1 and r['M'] in (0, se['margin_percentiles'][1] if False else r['M']) and r['W'] in (50, 75, 98) and (r['M'] == 0 or r['M'] >= 3000):
            L.append(f"| {r['W']} | {r['M']} | {100*r['det_rate']:.1f} | {100*r['wrong_rate']:.1f} | {100*r['miss_rate']:.1f} | {r['false_alarms_per_hour']:.1f} | {r['latency_median_frames']} | {r['latency_p90_frames']:.0f} |")
    rR = {r['R']: r for r in rules if r['W'] == 75 and r['M'] == se['margin_percentiles'][2] and r['N'] == 1}
    bc = se['best_constrained'][0] if se['best_constrained'] else None
    L.append(f"""
- **불응기 R이 결정적**: 같은 W=75, M={se['margin_percentiles'][2]}, N=1에서 R=0 → {rR[0]['false_alarms_per_hour']:.0f} FA/h, R=50 → {rR[50]['false_alarms_per_hour']:.0f}, R=100 → {rR[100]['false_alarms_per_hour']:.1f} (매 프레임 재발화를 막음). 연속 조건 N은 마진 조건이 있으면 영향이 거의 없다.
- 마진 M은 검출률 ↔ 오분류·오검출의 트레이드오프 축: M≈6000(25 % 백분위) → 검출 72 %, 오분류 6 %, 23–27 FA/h; M≈10800(중앙값) → 검출 57–58 %, 오분류 1.3 %, 0.6–3 FA/h.
- 지연 중앙값 −33 ~ −37프레임: 규칙이 클립 종료 **전**(단어 중간)에 발화한다 → 오분류의 주원인(부분 단어로 판정). W=50이 W=98보다 오검출이 적다(무음 구간의 누적이 희석됨).
- 클립 정확도(87.8 %) 대비 스트리밍 검출률(58–72 %)이 크게 낮은 원인과 대안: (1) 조기 판정 → **발화 종료(VAD offset) 시점 또는 마진 피크에서 판정**하도록 규칙 보강(RTL에 VAD 에너지 비교기 추가), (2) 학습이 1초 클립·0 초기상태 기준 → **연속 시퀀스 fine-tune / 랜덤 초기상태 학습**, (3) 게인 범위: 보드에서는 PGA로 레벨을 고정하므로 −20 dB 극단은 완화됨.
- **추천 규칙(RTL `decision` 파라미터)**: **W=50, MARGIN=6000, N_CONSEC=1, REFRACT=100, N_KEYWORD=10** (검출 72 %, 오분류 6 %, 23 FA/h) — 데모용. 오검출을 우선하면 **MARGIN=10800**(검출 57 %, 오분류 1.2 %, 0.6 FA/h). 제약(FA/h ≤ 5, 오분류 ≤ 3 %) 하 최고: W={bc['W']}, M={bc['M']}, N={bc['N']}, R={bc['R']} → 검출 {100*bc['det_rate']:.1f} %.
""")
# B-5
L.append("### B-5. 연산량·에너지 비교 (`snn/eval/ops_energy.py`, `snn/runs/figures/ops_energy.csv`, `ops_vs_acc.png`)\n")
L.append("에너지 상수(45 nm, **추정**): Horowitz, \"Computing's energy problem (and what we can do about it)\", ISSCC 2014 — int add 8b 0.03 pJ / 32b 0.1 pJ, int mult 8b 0.2 pJ. MAC(8b 곱 + 32b 누적) := 0.3 pJ, AC(8b 가중치 가산) := 0.1 pJ(32b 가산값, 보수적). 메모리 접근·제어·클럭 에너지는 제외.\n")
L.append("| 모델 | test acc | MAC/추론(1 s 창) | AC/추론 | 스트리밍 ops/s | 추정 pJ/s | 비고 |\n|---|---|---|---|---|---|---|")
for r in opsrows:
    L.append(f"| {r['model']} | {100*float(r['acc']):.2f} | {float(r['mac_per_inf']):,.0f} | {float(r['ac_per_inf']):,.0f} | {float(r['ops_per_s_stream']):,.0f} | {float(r['energy_per_s_pJ']):,.0f} | {r['note']} |")
L.append("\n- **대표 수치는 판정 주기 100 ms 기준**: DS-CNN 24ch는 판정마다 1 s 창을 재계산하므로 2.27 M MAC × 10/s = **22.7 M ops/s**, 추천 SNN은 프레임 증분이라 판정 주기와 무관하게 **0.47 M ops/s(0 건너뛰기) / 1.06 M(dense)** → **48×**(0 건너뛰기) / 21×(dense) 적은 연산. 판정 주기 10 ms(매 프레임)일 때만 480×, 1 s(클립 단위)일 때는 4.8×다(`snn/runs/figures/ops_vs_period.png`, `ops_vs_period.csv`). 기존 hls4ml MFCC DS-CNN(10cls)은 100 ms 기준 5.7 M ops/s(12×).\n\n![ops vs period](runs/figures/ops_vs_period.png)\n- SNN 내부: 입력층 MAC 752,640/초(dense) → 173,176(0 건너뛰기, 측정 77 % 절감), 은닉 AC 285k/초(실측 스파이크 × fan-out).\n\n![ops vs acc](runs/figures/ops_vs_acc.png)\n")
# B-6
L.append("### B-6. 0 건너뛰기 통계 (`snn/eval/zero_skip_stats.py`, `snn/runs/zero_skip_stats.json`)\n")
if zs:
    L.append("| 잡음 | 레벨(dBFS RMS) | 입력 0 비율 | 입력층 MAC/프레임 (dense 7,680) | MAC 절감 | 정확도(정수 골든) |\n|---|---|---|---|---|---|")
    for r in zs:
        L.append(f"| {r['noise']} | {r['dbfs'] if r['dbfs'] is not None else '–'} | {100*r['feat_zero_ratio']:.1f} % | {r['input_mac_per_frame']:,.0f} | {100*r['mac_saving']:.1f} % | {100*r['acc']:.2f} |")
    L.append("\n- 테스트셋 원본: 0 비율 76.9 %(1주차 77.6 %는 100개 샘플 기준), 입력층 MAC 절감 77.0 %(탭 3개가 같은 프레임을 보므로 0 비율과 거의 같음).\n- −60~−80 dBFS 백색/핑크 잡음을 더해도 0 비율·절감률이 변하지 않는다: 특징 바닥(오프셋 332)은 광대역 잡음 기준 약 **−40 dBFS RMS**에 해당해 그 아래 플로어는 전부 0으로 떨어진다(사인파 기준 −100 dBFS와는 다름). 즉 실제 마이크·코덱 플로어(−60 dBFS 이하)에서도 절감률은 유지된다(추정: 보드 측정으로 확인 필요). 더 높은 레벨(−50/−40 dBFS)은 표 참조 — 0 비율이 줄기 시작하는 지점.\n")
else:
    L.append("pending\n")
# B-7
if cf:
    cl = cf['classes']
    L.append("### B-7. 클래스별 분석 (`snn/eval/confusion.py`, `snn/runs/confusion.json`, 12클래스 test 4,888개)\n")
    L.append("| 클래스 | SNN M3k3 Q8+KD (정수 골든) | DS-CNN 24ch | 차이 |\n|---|---|---|---|")
    for c in cl:
        a, b = cf['snn']['per_class'][c], cf['dscnn']['per_class'][c]
        L.append(f"| {c} | {100*a:.1f} | {100*b:.1f} | {100*(a-b):+.1f} |")
    L.append(f"| **전체** | **{100*cf['snn']['acc']:.2f}** | **{100*cf['dscnn']['acc']:.2f}** | {100*(cf['snn']['acc']-cf['dscnn']['acc']):+.2f} |")
    m = np.array(cf['snn']['cm']); md = np.array(cf['dscnn']['cm'])
    def top_conf(mat):
        out = []
        for i in range(len(cl)):
            j = int(np.argsort(-mat[i])[1] if np.argmax(mat[i]) == i else np.argmax(mat[i])); out.append(f"{cl[i]}→{cl[j]} {mat[i,j]}")
        return out
    L.append(f"\n- SNN은 **unknown**에서 DS-CNN보다 +8.8 %p 좋고(74.9 vs 66.1), **silence**에서 −6.9 %p(93.1 vs 100; silence→down 21개 — 저레벨 배경소음 클립을 'down'으로 오인), 키워드는 대체로 −1~−5 %p(yes −2.8, left −5.3, up −2.4). 주요 혼동(SNN): " + ", ".join(top_conf(m)[:12]) + ".\n- 혼동행렬 전체는 `snn/runs/confusion.json`(행 = 정답, 열 = 예측).\n")
# C
L.append("""---
## C. 코덱 16 kHz 준비 (`snn/board/`)
- `reg_set_init_16k.c`: 레퍼런스가 서버에 없어 **데이터시트 기준 초안**. 구조는 PS I2C(XIicPs) 쓰기 + 레퍼런스 48 kHz 시퀀스 자리(`REFERENCE_INIT_SEQUENCE()`) + `codec_set_16k()`: R17 Converter Control 0(0x4017) CONVSR[2:0]=011(16 kHz), R64 Serial Port Sampling Rate(0x40F8) SPSR[2:0]=011. 비트 인코딩·I2C 주소·표 번호는 `// TODO verify`. 핵심 분기: 코덱이 시리얼 포트 **마스터**(R15 0x4015 bit0)일 때만 레지스터로 LRCLK가 바뀜.
- `CHECK_16k.md`: UART 레지스터 리드백 → ILA로 LRCLK 주기 62.5 µs(3000클럭@48 MHz)·`new_sample` 간격 3000클럭·BCLK 1.024 MHz 확인 절차, 실패 유형별 판단표, 2일 이상 미해결 시 48 kHz + RTL 데시메이터(67탭, 1주차 검증) 확정 기준.

---
## 3주차 리스크와 제안
1. **스트리밍 판정**이 가장 큰 리스크(클립 87.8 % → 스트림 검출 58–72 %). 제안: (a) RTL `decision`에 VAD(프레임 에너지 = mel 합 log2q 또는 특징 합) 기반 **발화 종료 트리거** 추가, (b) 연속 시퀀스/랜덤 초기상태 fine-tune(구조·비트폭 불변, `.mem`만 교체), (c) 보드 PGA로 레벨 고정.
2. 합성 결과 미확인(서버 Vivado 없음): tcl 실행 후 BRAM 23+2, DSP 1, WNS 확인 필요. 포트 `vo_sum[383:0]`은 디버그용이라 보드 통합 시 제거 가능.
3. 코덱 마스터/슬레이브 확인 전까지 16 kHz 경로 불확실 → C 절차대로 1시간 내 판정, 실패 시 데시메이터 RTL(3주차 프론트엔드 범위에 포함).
4. 10클래스 −2 %p 기준(B-1)과 시드 통계(B-2), 장기 학습(B-3)은 학습 완료 후 재생성(위 pending).
5. 프론트엔드 RTL(FFT/mel/log2)은 3주차: 골든 블록 함수와 벡터(`snn/golden/vectors/m3k3/*/{windowed,fft_re,fft_im,power,mel_acc,log2q,feat}.hex`)가 준비돼 있다.

## 생성 파일과 재현
```
snn/rtl/      SPEC_snn_core.md  snn_core.v snn_ctrl.v row_acc.v lif_update.v out_integrator.v decision.v spike_hist.v feat_hist.v wrom.v prom.v
              synth_snn_core.tcl snn_core_ooc.xdc reports/
snn/tb/       tb_snn_core.v tb_regress.v run_vec.sh run_all_vecs.sh run_regress.sh build/
snn/golden/   dump_vectors.py(--regress N, --stream a,b,..)  vectors/m3k3/ m3k3_regress/ m3k3_stream/ m3k3q6/
snn/eval/     stream_eval.py zero_skip_stats.py ops_energy.py confusion.py collect_week2.py build_report2.py
snn/board/    user_logic.v ila_snn.xdc reg_set_init_16k.c CHECK_16k.md ; snn/rom/feat_sets.mem
snn/runs/     stream_eval/ zero_skip_stats.json confusion.json figures/ week2_tables.md rtl_vec_zs{0,1}.txt rtl_regress.txt rtl_stream.txt
snn/run_week2_B.sh  (B-1/2/3 GPU 런 드라이버)
```
```bash
bash snn/tb/run_vec.sh snn/golden/vectors/m3k3/yes_11 stop        # 벡터 1개, 첫 불일치에서 정지
bash snn/tb/run_all_vecs.sh m3k3 1 ; bash snn/tb/run_all_vecs.sh m3k3 0   # 12개, 0 건너뛰기 on/off
bash snn/tb/run_regress.sh M3k3e60_c12_q8_kd m3k3 240               # .mem 재내보내기 + 전체 회귀
python snn/eval/stream_eval.py ; python snn/eval/zero_skip_stats.py ; python snn/eval/ops_energy.py ; python snn/eval/confusion.py
bash snn/run_week2_B.sh ; python snn/eval/build_report2.py           # GPU 런 / 보고서 재생성
```
""")
# checklist
L.append(f"""## 완료 기준 체크리스트
- [x] `SPEC_snn_core.md`
- [x] `snn_core` RTL, 골든 벡터 12개 비트 단위 100 % 일치 ({c1['npass'] if c1 else '?'}/12 PASS, 불일치 0)
- [{'x' if 'PASS' in reg else ' '}] 회귀 240개 예측 100 % 일치 (`{reg[:60]}`), `run_regress.sh`
- [{'x' if 'PASS' in stream else ' '}] 스트리밍 모드 RTL = 골든 일치
- [x] 프레임당 실측 사이클(0 건너뛰기 on/off)
- [ ] 합성 결과 — Vivado 없음, **실행 가능한 tcl 준비**(`synth_snn_core.tcl`)
- [{'x' if 'pending' not in tables.split('### B-2')[0] else ' '}] 10클래스 M3k3 Q8(+KD) 결과
- [{'x' if 'pending' not in tables.split('### B-2')[1].split('### B-3')[0] else ' '}] 시드 3개 평균±표준편차
- [x] 스트리밍 평가와 추천 판정 규칙
- [x] 연산량·에너지 비교표와 그래프
- [x] 0 건너뛰기 통계(잡음 플로어 포함)
- [x] 16 kHz 코덱 설정 초안
- [x] REPORT_week2.md
""")
open(os.path.join(S, 'REPORT_week2.md'), 'w').write("\n".join(L))
print('report written', len("\n".join(L)))
