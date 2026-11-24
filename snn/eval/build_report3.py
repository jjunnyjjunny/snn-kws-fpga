"""build_report3.py — assemble snn/REPORT_week3.md from measured artifacts (re-run after any result updates)."""
import os, re, json, datetime, csv
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); S = os.path.join(ROOT, 'snn')
def rd(p): p = os.path.join(S, p); return open(p).read() if os.path.exists(p) else ''
def J(p): p = os.path.join(S, p); return json.load(open(p)) if os.path.exists(p) else None
def fe_stats(name):
    rows = [dict(re.findall(r'(\w+)=(-?\d+)', l)) for l in rd(f'runs/{name}.txt').splitlines() if l.startswith('FE_')]
    npass = sum(1 for l in rd(f'runs/{name}.txt').splitlines() if 'PASS' in l)
    return rows, npass
vad = J('runs/vad_eval/report.json'); pipe = rd('runs/pipeline_tb_v2.txt') or rd('runs/pipeline_tb.txt'); pipe1 = rd('runs/pipeline_tb.txt'); rt = rd('runs/fe_top_realtime.txt')
max_replay = max([int(m) for m in re.findall(r'max_replay_cyc=(\d+)', pipe + pipe1)] or [0])
b0, p0 = fe_stats('fe_blocks_ms0'); b1, p1 = fe_stats('fe_blocks_ms1'); t0, tp0 = fe_stats('fe_top_ms0'); t1, tp1 = fe_stats('fe_top_ms1')
L = []
L.append(f"# SNN KWS — 3주차 보고서 ({datetime.date.today()})\n")
L.append("범위: **A** 보고서 수치 표현 정리 + 판정 주기별 연산량 그래프, **B** VAD 구간 판정(Python 평가, 파라미터 탐색은 val 스트림, 보고는 test 스트림), **C** 프론트엔드 RTL(블록별·샘플 입력 골든 비트 일치), **D** VAD·프리롤·구간 판정 RTL + 통합 테스트(정수 골든 파이프라인과 프레임 단위 일치), **E** 보드 1단계 패키지, **F** 합성 tcl. 수치는 실제 실행 결과이며 추정치는 \"추정\"으로 표시. 동결 사양(M3k3 8bit+KD `m3k3`, mel40/hop160/win480, 18bit FFT, log2 Q6.4, 오프셋 332, LIF k=3, 16bit 막전위, 24bit 출력 적분기, clk_48)은 바꾸지 않았다 — 바뀐 것은 판정 로직과 SNN 리셋 시점뿐.\n")
L.append("""## 0. 시작 전 — 레퍼런스 분석과 2주차 초안과의 차이
레퍼런스(`FPGA implementation/`, 복사본 `snn/board/ref/`; 두 경로의 소스는 동일):
- Top: `Block_Design_wrapper`(PS7 FCLK0 100 MHz → clk_wiz → `clk_out_48`/`rstn_out_48`, EMIO I2C) + `adau1761_interface`(MCLK = clk_48/2 = 24 MHz, `i2s_clock_interface`, `i2s_data_interface`) + `user_logic`(`.sw(~sw)`, `.rst(!rstn_48)`). **`new_sample`은 선언 없는 암묵 wire로 `adau1761_interface` 출력에만 연결, `user_logic`에는 미연결** — 확인.
- `i2s_data_interface`: BCLK 상승 에지 검출 → LRCLK 하강 에지에서 `audio_l/r_out`과 `new_sample`을 **같은 always 블록**에서 갱신(new_sample=1 사이클에 데이터 유효) — 확인. 지연 2 LRCLK.
- `reg_set_init.c`: `XIicPs_MasterSendPolled(&Iic, [addr_hi, addr_lo, data], 3, 0x3B)` + 50 ms 지연 + 리드백 UART. Clock Control 0x0F(PLL, INFREQ 1024×fs), PLL 00 7D 00 0C 23 01 → 24 MHz/2 × (4 + 12/125) = 49.152 MHz = 1024 × 48 kHz, **Serial Port 0 (0x4015) = 0x01 → 코덱이 I2S 마스터** — 확인. 0x4017/0x40F8은 쓰지 않음(기본 48 kHz).
- `constraints.xdc`: 코덱 9핀 + sw 8핀(LVCMOS18)뿐(LED 없음). `ila.xdc`: 클럭 `Block_Design_wrapper_inst/Block_Design_i/clk_wiz_0/inst/clk_out1`, 프로브 `user_logic_inst/audio_*`, `adau1761_interface_inst/i2s_*`, `…/new_sample`.
- 튜토리얼(47쪽): Vivado GUI → Generate Bitstream → Export Hardware(XSA, include bitstream) → Vitis Classic 2023.2 → 플랫폼(XSA) → Hello World 앱 → C 임포트 → Program Device → Run. `platform.tcl`(xsct) 존재.

2주차 초안 대비 차이(모두 이번 주 패키지에 반영): ① 인스턴스 이름 `user_logic_inst`(초안 `u_user_logic`), ILA 클럭 net, 스위치 반전(`~sw`), 리셋 극성 ② 코덱 C는 자체 I2C 래퍼 대신 레퍼런스의 `XIicPs_MasterSendPolled` 호출·지연·리드백 형식을 그대로 복사하고 **0x4017=0x03, 0x40F8=0x03 두 줄과 리드백만 추가** ③ 코덱 마스터이므로 레지스터 변경만으로 16 kHz 기대(데시메이터는 실패 시에만) ④ 2주차 합성 리포트는 `snn/rtl/reports/`에 없어 건너뜀.
""")
# ---------- A
L.append("""## A. 보고서 수치 표현 (2주차 보고서 수정 완료, `snn/REPORT_week2.md` 재생성)
- **10클래스**: DS-CNN 24ch **91.66 ± 0.20 %** vs SNN M3k3 Q8+KD 60ep **89.59 ± 0.55 %**(시드 3개 평균) → **−2.1 %p, 목표 경계**. 기존 보드 hls4ml MFCC DS-CNN 91.75 % 대비 −2.2 %p. (단일 시드 91.43 % 기준의 "충족" 표현 삭제.)
- **12클래스**: 89.08 ± 0.17 % vs 87.67 ± 0.43 % → −1.4 %p, 목표 충족.
- **연산량 대표 수치는 판정 주기 100 ms 기준**: DS-CNN 24ch 22.7 M ops/s vs SNN 0.47 M ops/s(0 건너뛰기) → **48×**; 10 ms 판정일 때만 480×, 1 s 판정이면 4.8× (`snn/runs/figures/ops_vs_period.png`, `.csv`).
- 1주차의 test 기반 선택(30→60 에폭, M3k3 선택)은 각주로 명시.

![ops vs period](runs/figures/ops_vs_period.png)
""")
# ---------- B
if vad:
    bv, lf = vad['best_val'], vad['lowfa_val']; T = vad['test']
    def row(name, r):
        if isinstance(r, float): return f"| {name} | 오라클 정확도 {100*r:.1f} % | | | | | | |"
        return f"| {name} | {100*r['det_rate']:.1f} | {100*r['wrong_rate']:.1f} | {100*r['miss_rate']:.1f} | {r['false_alarms_per_hour']:.1f} | {r.get('latency_median', r.get('latency_median_frames'))} | {100*r['snn_duty']:.1f} % | {r.get('vad_event_iou', float('nan')):.2f} |" if 'snn_duty' in r else \
               f"| {name} | {100*r['det_rate']:.1f} | {100*r['wrong_rate']:.1f} | {100*r['miss_rate']:.1f} | {r['false_alarms_per_hour']:.1f} | {r.get('latency_median_frames')} | 100 % | – |"
    L.append(f"""## B. VAD 구간 판정 (`snn/eval/vad_eval.py`, `snn/runs/vad_eval/`)

### B-1. 파이프라인(정수, RTL 골든)
1. 프레임 에너지 **E1 = Σ_m f[t][m]**(40개 8bit 특징의 합, 14bit) 채택 — val 탐색에서 E2(mel 누적기 합의 log2q)보다 목적함수가 높았고 RTL 비용이 0(특징 스트림을 더하기만). 2. VAD 상태기계: on `E > T_on` 연속 N_on 프레임, off `E < T_off` 연속 H 프레임(hangover), L_min 미만 구간 폐기, L_max 도달 시 강제 판정. 3. 프리롤 P 프레임: 트리거 시 SNN 리셋 → 버퍼의 최근 P+1 프레임(프리롤 + 트리거 프레임) 연속 입력 → 이후 라이브 프레임. 4. 구간 종료 시 출력 막전위 합의 argmax, silence/unknown은 무검출, 선택적 마진. 5. SNN은 구간(프리롤 포함) 동안만 동작.

### B-2. 평가 (val 스트림 100개 × 2조건으로 탐색, test 스트림 200개 × 2조건으로 보고; 30 s 스트림, 키워드 발화 test w2 1,918 / board 1,891개)
val 탐색 최적(목적함수 det − wrong − 0.002·FA/h, 조건 혼합): **{bv['energy']}, T_on {bv['T_on']}, T_off {bv['T_off']}, N_on {bv['N_on']}, H {bv['H']}, P {bv['P']}, L_min {bv['L_min']}, L_max {bv['L_max']}** → val det {100*bv['det_rate']:.1f} %, wrong {100*bv['wrong_rate']:.1f} %, FA/h {bv['false_alarms_per_hour']:.1f}, SNN 가동률 {100*bv['snn_duty']:.1f} %. 오검출 우선 세트: 같은 파라미터 + 마진 **{lf['margin']}** (val det {100*lf['det_rate']:.1f} %, FA/h {lf['false_alarms_per_hour']:.1f}).

| 조건 | 방식 | 검출률 % | 오분류 % | 미검출 % | 오검출/h | 지연 중앙값(프레임, offset 기준) | SNN 가동률 | VAD-정답 IoU |
|---|---|---|---|---|---|---|---|---|""")
    for cond, lab in (('w2', '2주차 조건 (게인 −20..+10 dB, 소음 0..0.1)'), ('board', '보드 조건 (게인 ±6 dB, 소음 0..0.05)')):
        r = T[cond]
        L.append(row(f"{lab} — 오라클 창(리셋, 정답 구간 합)", r['oracle_acc']))
        L.append(row(f"{lab} — 2주차 슬라이딩(W=50, M=6000, R=100)", r['sliding_w2rule']))
        L.append(row(f"{lab} — **VAD 구간 판정(추천)**", r['vad_recommended']))
        L.append(row(f"{lab} — VAD 구간 판정(오검출 우선, 마진 {r['vad_low_fa']['margin']})", r['vad_low_fa']))
    rb = T['board']['vad_recommended']; rw = T['w2']['vad_recommended']; ob = T['board']['oracle_acc']
    L.append(f"""
- **보드 조건에서 VAD 구간 판정 검출률 {100*rb['det_rate']:.1f} % = 오라클 {100*ob:.1f} %와 {100*(ob-rb['det_rate']):.1f} %p 차이** (검출+오분류 = {100*(rb['det_rate']+rb['wrong_rate']):.1f} %: 거의 모든 발화가 구간으로 잡히고, 구간 내 정확도 {100*rb.get('seg_acc', rb['det_rate']/max(rb['det_rate']+rb['wrong_rate'],1e-9)):.1f} %). 슬라이딩 규칙 대비 검출률 +{100*(rb['det_rate']-T['board']['sliding_w2rule']['det_rate']):.1f} %p, 미검출 {100*rb['miss_rate']:.1f} %, 지연 중앙값 {rb['latency_median']} 프레임(발화 끝 직후 판정).
- 남은 문제는 **오검출**: 마진 0에서 {rb['false_alarms_per_hour']:.0f}/h(2주차 조건 {rw['false_alarms_per_hour']:.0f}/h). 주원인은 unknown 발화(모델의 unknown 정확도 74.9 % → 25 %가 키워드로 나옴)와 소음 구간. 마진으로 트레이드오프(아래 val 스윕). B-3 조건(오라클 대비 −5 %p 이상 또는 75 % 미만)은 **해당하지 않아 구간 단위 fine-tune은 하지 않았다**.
- SNN 가동률 {100*rb['snn_duty']:.1f} %(보드 조건) — 프레임의 ~{100-100*rb['snn_duty']:.0f} %에서 SNN 정지(VAD·프론트엔드만 동작).

val 마진 스윕(추천 파라미터, 구간 합 top1−top2 마진 백분위 기준):

| 마진 | 검출 % | 오분류 % | 오검출/h |
|---|---|---|---|""")
    for m in vad.get('margin_sweep_val', []): L.append(f"| {m['margin']} | {100*m['det_rate']:.1f} | {100*m['wrong_rate']:.1f} | {m['false_alarms_per_hour']:.1f} |")
    L.append(f"""
**RTL `decision_seg`/`vad` 파라미터(추천)**: `T_ON={bv['T_on']}, T_OFF={bv['T_off']}, N_ON={bv['N_on']}, H={bv['H']}, P={bv['P']}, L_MIN={bv['L_min']}, L_MAX={bv['L_max']}, MARGIN=0, ENERGY_SEL=1`; 오검출 우선: `MARGIN={lf['margin']}`.
""")
else: L.append("## B. VAD 구간 판정\npending (`snn/logs/vad_eval2.log`)\n")
# ---------- C
L.append(f"""## C. 프론트엔드 RTL (`snn/rtl/fe/`, 테스트 `snn/tb/fe/`)

| 모듈 | 역할 | 골든 비교 |
|---|---|---|
| `fe_in_scale` | `sat18((pcm24 + 2^5) >>> 6)` | (x_db.hex 경유) |
| `fe_ring` | 18bit 샘플 링 512(BRAM), 480샘플 후 첫 `frame_start`, 이후 160샘플마다 | 프레임 정의 = x[160t : 160t+480] |
| `fe_window` | ring × Hann Q1.15 → `sat18((x·w + 2^14) >>> 15)` → 비트역순 주소로 버퍼 적재, 480..511 = 0 | `windowed.hex` |
| `fe_fft512` | radix-2 DIT in-place 9스테이지 ×1/2, twiddle ROM, 버터플라이 `sat18((a<<15 ± b·W + 2^15) >>> 16)`, 스테이지 경계마다 파이프 드레인 | `fft_re/im.hex` |
| `fe_power` | re²+im² 36bit | `power.hex` |
| `fe_mel_log` | 희소 mel 표 → 50bit 누적 → leading-one + 5bit LUT → Q6.4 → `sat8(log2q−332)`; E1 = Σf, E2 = log2q(Σacc) | `mel_acc.hex`, `log2q.hex`, `feat.hex` |
| `fe_top` | 순차 제어(window → fft → mel), 특징 스트림(`feat_valid/data/last`) + E1/E2 + `frame_done` | `feat.hex` (pcm24.hex 입력) |

**블록별 비트 일치**(12벡터 × 98프레임, `tb_fe_blocks.v`): MULT_SERIAL=0 → {p0}/12 PASS, MULT_SERIAL=1 → {p1}/12 PASS, 불일치 0(윈도우 480값·FFT 512복소값·파워 257·mel 40·log2 40·특징 40, 매 프레임).
**샘플 입력 → 특징**(`tb_fe_top.v`, pcm24.hex를 `new_sample`로 입력, 가속 간격 64클럭): MULT_SERIAL=0 {tp0}/12 PASS, =1 {tp1}/12 PASS (98프레임 특징 + E1 전부 일치). 실시간 간격(3000클럭) 1벡터: `{rt.strip().split('vec=')[-1] if rt else 'pending'}`.

**프레임당 사이클(측정, `fe_top`)**:

| FFT 설정 | 윈도우 | FFT | mel+log2 | 합계 | DSP(추정) |
|---|---|---|---|---|---|
| MULT_SERIAL=0 (버터플라이 2사이클, 곱셈기 4) | {t0[0]['win'] if t0 else '–'} | {t0[0]['fft'] if t0 else '–'} | {t0[0]['mel'] if t0 else '–'} | **{t0[0]['cyc_avg'] if t0 else '–'}** | 윈도우 1 + twiddle 4 + 파워 2 + mel 2 = 9 |
| MULT_SERIAL=1 (버터플라이 4사이클, 곱셈기 1) | {t1[0]['win'] if t1 else '–'} | {t1[0]['fft'] if t1 else '–'} | {t1[0]['mel'] if t1 else '–'} | **{t1[0]['cyc_avg'] if t1 else '–'}** | 윈도우 1 + twiddle 1 + 파워 2 + mel 2 = 6 |

프레임 예산 48만 사이클(10 ms)의 1.2~2.2 %. 샘플 링 여유(512−480 = 32샘플 = 96k 사이클) 대비 윈도우 적재 516사이클.

**DSP48E1 매핑 계획(추정, 합성으로 확인 필요)**:

| 연산 | 폭 | DSP | 비고 |
|---|---|---|---|
| 윈도우 x·Hann | 18 × 17(Q1.15 unsigned→signed) | 1 | 25×18 한 개 |
| twiddle 복소 곱 | 18 × 16 × 4 | 4 (또는 1 직렬) | `MULT_SERIAL` |
| 파워 re², im² | 18 × 18 × 2 | 2 | 18×18 한 개씩 |
| mel P × w | 36 × 8 | 2 | 36 = 18+18 분할, 또는 LUT shift-add(8사이클) |
| SNN 입력층 x·W | 9 × 8 | 1 | 2주차 |
| 합계 | | 10 (또는 7) | 7020 DSP 220 |
""")
# ---------- D
L.append(f"""## D. VAD·판정 RTL과 통합 (`snn/rtl/vad.v`, `preroll_buf.v`, `decision_seg.v`, `kws_top.v`; 골든 `snn/golden/pipeline_int_ref.py`; 테스트 `snn/tb/pipeline/`)
- `vad.v`: B-1 상태기계(파라미터 T_ON/T_OFF/N_ON/H/P/L_MIN/L_MAX), `strobe`(fe frame_done)마다 1스텝, `seg_start(n_pre = min(P+1, 본 프레임 수))`/`seg_end(keep, forced)`.
- `preroll_buf.v`: 32프레임 × 40 × 8bit 링(분산 RAM). 매 프레임 특징 기록; `replay(n)`으로 최근 n프레임을 오래된 순으로 `snn_core`에 공급(프레임 사이 `busy` 대기), 프레임마다 `out_frame_done`.
- `decision_seg.v`: 구간 합(12×32bit) 누적(SNN 프레임 완료마다 `vo_cur` 가산) → 종료 시 argmax(동점 낮은 인덱스) + 키워드 필터 + MARGIN → `det_valid/det_any/det_class`. 2주차 슬라이딩 판정은 `snn_core` 안에 그대로 남아 있음(비교용, `det_valid`/`vo_sum` 포트).
- `kws_top.v`: fe_top → VAD → (start: `clip_start` + 프리롤 재생 | active: 1프레임 재생) → 판정. `snn_active`(클럭 enable 요청), 가동률 카운터(`cnt_frames`, `cnt_snn_frames`, `cnt_snn_busy_cycles`, `cnt_total_cycles`), `overrun`(재생 중 다음 프레임 이벤트 도착 시 sticky), 보드 모드 1용 외부 특징 주입 포트(`ext_*`).
- **프리롤 처리 시간**: P=20 → 트리거 프레임에 21프레임 연속 처리 = 21 × 최악 16.7k = **351k 사이클(추정 최악) < 프레임 주기 480k** → FIFO 불필요. 통합 시뮬 실측 최대(스트림 8개, `max_replay_cyc`) **{max_replay:,} 사이클**(무음 프리롤 프레임이 많아 평균 ~3.7k/프레임). P ≤ 27까지 여유.
- **통합 테스트**(`tb_pipeline.v`, 보드 조건 8 s 스트림 4개, pcm24 → kws_top, 골든 파이프라인과 프레임 단위로 VAD 상태·구간 경계·판정·공급 프레임 수·에너지·구간 합 비교):

```
{pipe.strip() if pipe else 'pending'}
```
(첫 번째 파라미터 세트 T_on 1000/T_off 600/N_on 2/L_max 98로 만든 스트림 4개도 동일하게 통과: `snn/runs/pipeline_tb.txt`)
""")
# ---------- E, F
L.append("""## E. 보드 1단계 패키지 (`snn/board/pkg/`)
| 파일 | 내용 |
|---|---|
| `ZSK_Design_Top.v` (+`.diff`) | 레퍼런스 복사본에 `wire new_sample;` 선언과 `user_logic` 포트 연결 2줄만 추가 |
| `user_logic.v` | 레퍼런스 포트 + `new_sample`. `sw[1:0]` 모드: 0 패스스루 / 1 골든 특징 주입(`feat_sets.mem` 5세트 yes·no·stop·go·silence, `sw[4:2]` 선택, 결과 레지스터 + 비프) / 2 라이브(kws_top) / 3 오디오 캡처(`cap_sample[17:0]`, `cap_valid=new_sample`). `sw[5]` 고속, `sw[6]` 뮤트, `sw[7]` 홀드. `led`는 내부 신호 |
| `rom_paths.vh` | `$readmemh` 절대 경로 매크로(빌드 tcl이 덮어씀) |
| `ila_kws.xdc` | 레퍼런스 클럭 net 기준, 프로브 18개(모드, new_sample, 18bit 샘플, VAD, det, frame_done, snn_active, 카운터, 에너지, 결과 레지스터, overrun, led, LRCLK/BCLK), 깊이 16384 + storage qualification 활성 |
| `build_bitstream.tcl` | 프로젝트 생성 → `ref/Block_Design.tcl` → wrapper → 소스/`.mem`/xdc → synth/impl/bitstream → `.bit .ltx .xsa` + 리포트 |
| `reg_set_init.c` | 레퍼런스 그대로(48 kHz) |
| `reg_set_init_16k.c` | 레퍼런스 + `0x4017=0x03`, `0x40F8=0x03` 쓰기 + 리드백 UART(근거 주석, `TODO verify`) |
| `vitis_xsct.tcl` | xsct 흐름(플랫폼 생성 → 앱 → 소스 임포트 → 빌드 → 프로그램/실행) |
| `BOARD_STEP1.md` | 4단계 체크리스트(패스스루 → 모드 0/1 → 16 kHz LRCLK → 캡처 CSV) + 실패 시 확인표 |
| `snn/eval/board_capture_check.py` | 캡처 CSV → 레벨/특징 분포/예측/PGA 권고 |

## F. 합성
서버에 Vivado 없음. `snn/rtl/synth_kws_top.tcl`(kws_top OOC: fe_top + vad + preroll + snn_core + decision_seg, 48 MHz) 준비, 2주차 `synth_snn_core.tcl`과 같은 리포트 세트. 보드 전체는 `build_bitstream.tcl`이 `reports/`에 utilization/timing/power를 남긴다. 추정 자원: BRAM18K ≈ 23(SNN 가중치) + 2(판정 링) + 1(샘플 링 512×18) + 1(FFT 버퍼 512×36) + 1(특징 ROM 19.6 KB, 모드 1) ≈ 28, DSP ≈ 10, LUT 5–8k (추정).

## 4주차 리스크와 제안
1. **오검출**(보드 조건 44/h @ 마진 0): 데모에서는 마진을 켜고(val 스윕 표), 근본 대책은 unknown 학습 강화(unknown 샘플 비율↑, 배경소음 클립을 silence로 더 학습) + 구간 단위 fine-tune(B-3 조건 미충족이라 보류).
2. 보드 마이크 레벨/노이즈 플로어가 학습 분포와 다를 수 있음 → 모드 3 캡처 → `board_capture_check.py`로 확인 후 PGA 조정, 필요하면 T_ON 재설정(ILA `energy` 프로브).
3. 합성 미확인(타이밍·자원) → PC에서 `build_bitstream.tcl` 실행이 첫 작업. `$readmemh` 경로 문제가 가장 흔한 실패 원인(`rom_paths.vh`).
4. 실시간 여유: 프론트엔드 6k + SNN 최대 재생 351k(추정) < 480k. `overrun` 프로브로 확인.
5. 전력 측정(4주차): vectorless 리포트 → 보드 전류 측정(SNN 가동률 ~29 %의 효과 확인).

## 생성 파일과 재현
```
snn/rtl/fe/        fe_in_scale.v fe_ring.v fe_buf.v fe_window.v fe_fft512.v fe_power.v fe_mel_log.v fe_top.v
snn/rtl/           vad.v preroll_buf.v decision_seg.v kws_top.v synth_kws_top.tcl (+ 2주차 snn_core 등, vo_cur 포트 추가)
snn/tb/fe/         tb_fe_blocks.v run_fe_blocks.sh tb_fe_top.v run_fe_top.sh ; snn/tb/pipeline/ tb_pipeline.v run_pipeline.sh
snn/golden/        pipeline_int_ref.py (SnnStepper in snn_int_ref.py) ; vectors/pipeline/stream_{0..3}/
snn/eval/          vad_eval.py ops_vs_period.py board_capture_check.py build_report3.py
snn/board/ref/     레퍼런스 원본 ; snn/board/pkg/  보드 패키지(E) ; snn/rom/feat_sets.mem (5 클립)
snn/runs/          vad_eval/ fe_blocks_ms{0,1}.txt fe_top_ms{0,1}.txt fe_top_realtime.txt pipeline_tb.txt figures/ops_vs_period.*
```
```bash
python snn/eval/vad_eval.py --n_val 100 --n_test 200 --workers 20            # B (val 탐색 + test 보고)
bash snn/tb/fe/run_fe_blocks.sh snn/golden/vectors/m3k3/yes_11 0 stop        # C 블록 테스트 (0/1 = MULT_SERIAL)
bash snn/tb/fe/run_fe_top.sh snn/golden/vectors/m3k3/yes_11 0 64             # C 샘플 입력 (64 = 가속, 3000 = 실시간)
python snn/golden/pipeline_int_ref.py --dump 4 --dur 8 --T_on .. --P ..      # D 골든 스트림
bash snn/tb/pipeline/run_pipeline.sh snn/golden/vectors/pipeline/stream_0    # D 통합 테스트
python snn/eval/build_report3.py
```
""")
# checklist
L.append(f"""## 완료 기준 체크리스트
- [x] 레퍼런스 파일 분석과 2주차 초안과의 차이 정리
- [x] 10클래스 "경계" 표기, 100 ms 기준 대표 수치, `ops_vs_period.png`
- [{'x' if vad else ' '}] VAD 구간 판정 평가(두 조건), 추천 파라미터, SNN 가동률
- [x] `pipeline_int_ref.py` 정수 골든
- [{'x' if p0 == 12 and p1 == 12 else ' '}] 프론트엔드 RTL 블록별 비트 일치(12벡터 × 98프레임, 불일치 0)
- [{'x' if tp0 == 12 else ' '}] `fe_top` 샘플 입력 → 특징 일치
- [{'x' if pipe and 'FAIL' not in pipe and 'PASS' in pipe else ' '}] VAD·프리롤·decision RTL + 통합 테스트 골든 일치
- [x] 보드 패키지(Top 수정본, user_logic, ila_kws.xdc, build_bitstream.tcl, Vitis C 2종, BOARD_STEP1.md, board_capture_check.py)
- [x] 합성 tcl (결과는 PC 실행 필요)
- [x] REPORT_week3.md
""")
open(os.path.join(S, 'REPORT_week3.md'), 'w').write("\n".join(L)); print('report3 written')
