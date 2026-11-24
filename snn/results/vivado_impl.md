# Vivado 2023.2 구현 결과 (PC, 2026-10-28) — ZSK_Design_Top (레퍼런스 블록 디자인 + I2S + KWS user_logic + ILA)

빌드: `snn/board/pkg/build_bitstream.tcl` (`source` in Vivado Tcl Shell, argv = C:/kws/snn), part xc7z020clg484-1, clk_48 단일 클록.
산출: ZSK_Design_Top.bit / .ltx / .xsa, reports/impl_util.rpt, impl_util_hier.rpt, impl_timing.rpt, impl_power_vectorless.rpt.

## 전체 (ILA 포함) — impl_util.rpt
| 자원 | 사용 | 가용 | % |
|---|---|---|---|
| Slice LUTs | 8,692 | 53,200 | 16.34 |
| Slice Registers | 11,118 | 106,400 | 10.45 |
| Block RAM Tile (RAMB36 88 + RAMB18 11) | 93.5 | 140 | 66.79 |
| DSPs | 9 | 220 | 4.09 |

타이밍(impl_timing.rpt): All user specified timing constraints are met. WNS 3.013 ns / TNS 0 (30,859 endpoints), WHS 0.033 ns, WPWS 3.000 ns @ clk_48 (20.833 ns).

## 계층별 (impl_util_hier.rpt, 구현 후)
| 인스턴스 | LUT 합 | 논리 LUT | LUTRAM | SRL | FF | RAMB36 | RAMB18 | DSP |
|---|---|---|---|---|---|---|---|---|
| user_logic_inst (전체) | 8,160 | 7,344 | 104 | 712 | 10,104 | 88 | 10 | 9 |
| u_ila (ILA, 측정용) | 2,794 | 2,338 | 0 | 456 | 3,958 | 65 | 1 | 0 |
| **u_kws (kws_top = 설계 본체)** | **5,173** | 4,813 | 104 | 256 | **5,795** | **15** | **9** | **9** |
| u_fe (fe_top) | 906 | 890 | 16 | 0 | 785 | 0 | 6 | 9 |
| u_buf / u_fft / u_mel / u_ring / u_win | 0 / 331 / 296 / 187 / 78 | | | | 0 / 383 / 155 / 160 / 69 | 0 | 2 / 2 / 1 / 1 / 0 | 0 / 4 / 4 / 0 / 1 |
| u_snn (snn_core) | 3,837 | 3,493 | 88 | 256 | 4,400 | 15 | 2 | 0 |
| u_acc / u_ctrl / u_dec(decision) | 960 / 1,475 / 376 | | | | 1,576 / 89 / 449 | 0 / 0 / 1 | 0 / 0 / 1 | 0 |
| u_dec (decision_seg) / u_vad / u_pb | 306 / 51 / 66 | | | | 465 / 44 / 36 | 0 | 0 / 0 / 1 | 0 |

kws_top 기준 BRAM18K 환산 = 15×2 + 9 = 39 (Yosys 추정 29: fe_buf 분리 +1 RAMB36, ILA 제외). SNN 입력층 곱셈은 Vivado가 DSP 대신 LUT로 구현(u_acc 960 LUT, DSP 0; Yosys 추정 DSP 12 → Vivado 9).

## 전력 (impl_power_vectorless.rpt, 벡터 없는 추정 — 참고치)
| 항목 | W |
|---|---|
| Total On-Chip | 1.857 |
| Dynamic | 1.698 (PS7 모델값 1.527 포함) |
| Device Static | 0.159 |
| user_logic_inst | 0.044 (u_ila 0.015, **u_kws 0.029**) |
vectorless는 활동률을 가정한 값이라 게이팅 효과를 반영하지 않음 → SAIF 기반 값은 `power_flow.tcl`(무음/키워드/대화)로 별도.

## 이력
- 1차 합성: LUT 57,175 > 53,200 (DRC UTLZ-1). 원인 fe_buf(512×36 TDP, 단일 always)가 FF 18.4k + 멀티플렉서로 구현(계층 표: u_buf 9.9k, u_fft 37.2k LUT). fe_buf를 re/im 512×18 두 메모리·포트별 always·ram_style=block으로 재작성(FE/통합 골든 전부 일치 유지) → 8,692 LUT.
- ILA(ila_kws, 21 프로브 146비트, 깊이 16384): 합성 기준 LUT 2,943 / FF 3,996 / RAMB36 65 + RAMB18 1 — 설계 자원이 아니므로 보고서에서는 분리해 기록.

## SAIF 기반 전력 (2026-10-29, `power_flow.tcl`, reports_power/)
방법: RTL 동작 xsim(테스트벤치 `tb_power.v`, PL 부분만, 48 MHz, 오디오 3 s, 모드 2 라이브) → SAIF(`xsim.simulate.saif_all_signals`) → `read_saif -strip_path tb_power/dut` → 구현 넷리스트 `report_power -hier all`. 넷 매칭 6 %(레지스터 출력; 나머지는 확률적 전파). 포스트 라우트 시뮬레이션은 xsim에서 시나리오당 10 h 이상이라 포기. PC 보안 정책이 xvhdl을 막아 블록 디자인(VHDL)은 시뮬레이션에서 제외(클록·리셋은 벤치가 구동).

동적 전력 (W, PL만; PS7 모델값 1.527 W·칩 정적 0.158 W는 설계와 무관):
| 시나리오 | user_logic_inst | **u_kws (설계)** | u_fe | u_snn | u_ila (측정용) |
|---|---|---|---|---|---|
| vectorless (활동률 가정) | 0.044 | 0.029 | 0.010 | 0.017 | 0.015 |
| SAIF 무음(배경 잡음) | 0.036 | 0.021 | 0.004 | 0.016 | 0.014 |
| SAIF 키워드 1개 | 0.034 | 0.019 | 0.003 | 0.015 | 0.014 |
| SAIF 대화(unknown+키워드) | 0.035 | 0.019 | 0.003 | 0.015 | 0.014 |

u_kws 구성(키워드): 클록 트리 0.009, 논리 0.005, 신호 0.002, BRAM 0.002, DSP <0.001 (합 0.019).

해석:
- 설계 본체(kws_top)의 PL 동적 전력은 **약 19~21 mW**이며, 그중 절반(9 mW)이 클록 분배다. 시나리오 간 차이(≤ 2 mW)는 리포트 해상도(1 mW)와 6 % 매칭의 확률적 전파 오차 안에 있어 **VAD·게이팅 효과는 이 추정에서는 분해되지 않는다**(무음 0.021 ≥ 키워드 0.019는 잡음 수준).
- 이유: 현재 RTL은 활동이 없을 때 데이터패스 토글과 BRAM enable만 멈추고 플립플롭 클록은 그대로다. 활동량 모델(4-A, SNN 가동률 4~39 %, FE 76~82 %)이 전력으로 이어지려면 BUFGCE 등 클록 게이팅이 추가로 필요하다(향후 과제).
- vectorless 0.029 W → SAIF 0.019 W: 실제 활동이 Vivado 기본 가정(토글률 12.5 %)보다 낮음.

## 클록 게이팅 (4-A 후속 4, 2026-10-29 RTL 완료, Vivado 측정 대기)
설계(`snn/rtl/clk_gate_buf.v`): `clk_gate_buf` = BUFGCE(합성) / 동작 모델(시뮬레이션, CE를 클록 low에서 샘플), `clk_gate_ctrl` = 항상 켜진 영역의 CE 핸드셰이크(req → 다음 사이클 ce=1 → 그다음 사이클 ready=1 → start 발행; req 해제 후 2사이클 뒤 ce=0; rst·force_on이면 항상 on).
- 게이트 클록 2개: (1) `snn_core` 전체 — 시퀀서가 S_WAKE 상태에서 ready를 기다린 뒤 clip_start/replay를 발행, req = 시퀀서 busy | 프리롤 replay | snn_busy | ext_en. (2) 프론트엔드 window/FFT/power/mel + 버퍼 — fe_top이 S_WAKE에서 ready 후 win_start, req = WAKE/WIN/FFT/MEL 상태. in_scale·링 버퍼·E0·VAD·프리롤·판정·user_logic은 게이팅하지 않음.
- 게이팅 on/off 선택: 모드 2에서 sw[5] (BUFGCE CE 강제 1). ready 타이밍이 on/off 동일하므로 동작·골든 결과는 동일(회귀로 확인). ILA probe21/22 = fe/snn CE.
- 비용: 프레임당 FE 사이클 5,942 → 5,945(+3, S_WAKE), SNN 세그먼트 시작/프레임 공급 전 +2~3 사이클. Yosys 추정 LC 7,763 → 7,821, BUFGCE 2.
- 측정 절차(PC): `build_bitstream.tcl` 변형 `nogate_popt`(기존 RTL + power_opt_design) / `""`(게이팅 RTL) → `power_flow.tcl <variant> <gate_off>` → `reports_power*`. 확인 항목: impl_clock_interaction.rpt(동일 클록), impl_cdc.rpt, impl_clock_util.rpt(BUFG 수), WNS.
- 회귀(서버 Icarus, 2026-10-29, `snn/runs/clkgate/`): FE 블록 24/24, fe_top 12클립 × (게이팅 on 직렬/병렬 + off) 36/36, SNN 클립 24/24·스트리밍·회귀 240/240, 통합 6 스트림 × 게이팅 on/off 12/12 — 전부 비트 일치, on/off 사이클 수 동일. Verilator 경고 0.
- 클록 enable 듀티(활동량 TB, 실시간 48 MHz 환산): board 20 s — FE 9.08M / SNN 5.02M 사이클 → 0.95 % / 0.52 % ; talk 20 s — 9.53M / 3.60M → 0.99 % / 0.38 % ; quiet 60 s — 29.4M / 1.51M → 1.02 % / 0.05 %. (CE 사이클 = 블록 busy 사이클과 일치: 일할 때만 클록이 켜짐.)
- Vivado 구현(게이팅 RTL, 10-30): 타이밍 제약 전부 만족(WNS 값 TBD), BUFGCTRL 6/32 (기본 4 + BUFGCE 2: `u_kws/G_CG.u_scb` SNN 클록 부하 4,697, `u_fe/G_CG.u_cb` FE 처리 클록 부하 636). report_clock_interaction: PL 클록은 clk_out1 하나뿐(게이트 클록도 같은 클록으로 분석, Clean), report_cdc: 기존 dbg_hub TCK↔clk 항목 외 새 항목 없음.
- 1차 전력 측정(10-30, SAIF, kws_top W): 기존 RTL + power_opt_design = 무음 0.023 / 키워드 0.022 / 대화 0.022 (power_opt 없는 기존 0.021/0.019/0.019 → **power_opt_design 효과 없음**, 오차 범위). 게이팅 RTL: on 0.024/0.022/0.023, off(sw[5]) 0.024/0.022/– → **on/off 동일** = BUFGCE CE 확률이 전력 계산에 전달되지 않음(게이트 클록·CE 넷이 넷리스트에서 `G_CG.u_scb/...`로 이름이 바뀌어 SAIF 매칭 실패, Clocks 행 0.021 불변). 조치: `power_flow.tcl`이 SAIF에서 CE 신호 T0/T1/TC를 읽어 BUFGCE CE 핀에 `set_switching_activity -static_probability/-toggle_rate`로 직접 주입(gate_off에서는 확률 1.0) → 재측정(10-30): Vivado는 BUFGCE CE에 switching activity를 받지 않음(Power 33-410, set_case_analysis만 허용) → `power_flow.tcl`이 CE 0/1 네 상태(ce11/ce10/ce01/ce00)를 리포트하고 SAIF 실측 듀티로 가중(듀티 가중 추정). xsim은 PC 정책 때문에 셸당 1회만 실행 가능 → 9 s 한 번에 돌리고 구간별 SAIF(`saif_multi.tcl`).
- **결과(kws_top PL 동적, W)**: 클록 on(ce11 = 게이팅 off) 무음 0.024 / 키워드 0.022 / 대화 0.022 → 듀티 가중 추정 **0.0060 / 0.0062 / 0.0061** (ce00 0.006). SNN 코어 0.017~0.018 → 0.0020~0.0022(듀티 0 / 1.21 / 0.78 %), FE 0.004 → 0.0020(듀티 1.23 %). 칩 Clocks 행 0.021 → 0.014. 게이팅 빌드 WNS +3.856 ns. 상시 대기(무음 90 %) 평균 추정 0.0208 → 0.0060 W. 표: `board/pkg/POWER_COMPARE.md` 4절.

## 기준 설계(hls4ml DS-CNN 통합, 중간 단계) 비교 (2026-10-30, `hls4ml_power.tcl`, PC `C:/kws/hls4ml_ref/reports/`)
체크포인트 `impl_1/ZSK_Design_Top_routed.dcp`(서버 사본 `DSCNN/impl_1/`), 같은 2023.2·xc7z020clg484-1·48 MHz·Vivado 기본 vectorless 설정. KWS 관련 계층(Block_Design_i 아래 myproject_0 + axi_dma_0 + axi_mem_intercon + ps7_0_axi_periph + GPIO 4 + kws_out_latch_0): LUT 19,710 / FF 34,917 / BRAM18K 111 / DSP 63, PL 동적 ≈ 0.116 W(IP 0.112 = 클록 0.029 + 신호 0.017 + 논리 0.011 + BRAM 0.030 + DSP 0.025; periph 0.002; 나머지 ≤ 0.003). 전체 설계 LUT 21,716 / FF 38,211 / BRAM 타일 73 / DSP 63, WNS +3.666 ns. 우리 kws_top 5,173 / 5,795 / 39 / 9, vectorless 0.029 W → 자원 3.8× / 6.0× / 2.8× / 7×, vectorless끼리 ≈ 4.0× (규모 비교; 기준 vectorless와 우리 SAIF는 직접 나누지 않음, 동작 반영은 SAIF·게이팅). 상세 `board/pkg/POWER_COMPARE.md` 3·5절.
