# POWER_COMPARE — 기존 hls4ml DS-CNN 설계와의 PL 전력·자원 비교 절차 (Windows Vivado 2023.2)

## 1. 이 설계(SNN KWS)
1. `build_bitstream.tcl` 실행 후 `power_flow.tcl`:
   ```
   vivado -mode batch -source power_flow.tcl -tclargs C:/kws_snn/snn silence keyword talk
   ```
   - `reports_power/vectorless_hier.rpt`: 활동량 없는 추정(기본 toggle rate 12.5 %).
   - `reports_power/<scenario>_saif_hier.rpt`: post-route 기능 시뮬레이션(xsim)에서 SAIF를 만들어 적용한 값. 시나리오 벡터는 `python snn/eval/make_power_vectors.py`로 생성(`power_vec/silence.hex`, `keyword.hex`, `talk.hex`, 각 3 s).
2. 읽는 법: `-hier` 리포트에서 **`user_logic_inst` 행 = PL KWS 동적 전력**(그 아래 `u_kws/u_fe`, `u_kws/u_snn`, `u_kws/u_vad`, `u_kws/u_dec` 등 계층별). `Block_Design_wrapper_inst`(PS7)와 `adau1761_interface_inst`는 제외. 정적(Device Static)은 설계와 무관하게 비교표에 따로 적는다.

## 2. 기존 hls4ml DS-CNN 프로젝트 (1단계 디자인, Windows `C:\kws_hls`)
1. Vivado에서 해당 프로젝트 열기 → `open_run impl_1`(구현이 없으면 `launch_runs impl_1 -to_step write_bitstream; wait_on_run impl_1`).
2. 클럭이 48 MHz가 아니면(1단계는 50 MHz) 비교를 위해 **같은 48 MHz 제약으로 재구현**하거나, 동적 전력을 주파수 비례(48/50)로 환산해 표기한다(환산한 값은 "환산"이라고 쓴다).
3. Tcl 콘솔:
   ```
   report_power -hier -file C:/kws_hls/reports/vectorless_hier.rpt
   report_utilization -hierarchical -file C:/kws_hls/reports/impl_util_hier.rpt
   ```
4. SAIF 기반 비교까지 하려면 hls4ml IP의 AXIS 입력에 같은 MFCC 벡터(골든 `golden_vectors.h`)를 넣는 xsim 테스트벤치가 필요하다(1단계 C-sim 벡터 재사용). vectorless만으로도 1차 비교는 가능하다.

## 3. 비교표 양식

| 항목 | hls4ml DS-CNN (1단계) | SNN KWS (이 설계) | 비고 |
|---|---|---|---|
| 클럭 | 48 MHz (clk_out1 20.833 ns, IP 포함 모든 KWS 논리) | 48 MHz | 동일 클록 (`clocks.txt`) |
| LUT / FF | 19,710 / 34,917 (KWS 관련 합계: myproject_0 18,072 / 31,649 + DMA 442 / 661 + 인터커넥트 805 / 974 + GPIO·latch 391 / 1,633) | 5,173 / 5,795 (kws_top, ILA 제외) | 기준 `C:/kws/hls4ml_ref/reports/impl_util_hier.rpt` 2026-10-30 ; 우리 `impl_util_hier.rpt` 2026-10-28 ; **3.8× / 6.0×** |
| BRAM18K / DSP | 111 (IP RAMB36 44 + RAMB18 21 = 109, DMA 2) / 63 | 39 (RAMB36 15 + RAMB18 9) / 9 | **2.8× / 7×** |
| (범위 맞춤) SNN 코어 + 판정만: u_snn + decision_seg | – | LUT 4,143 / FF 4,865 / BRAM18K 32 / DSP 0, vectorless 0.018 W (u_snn 0.017 + u_dec 0.001) | 기준 KWS 계층(특징 추출 제외)과 가장 가까운 범위 ; 프론트엔드 u_fe는 906 / 785 / 6 / 9, 0.010 W |
| PL 동적 전력, vectorless | **≈ 0.116 W** (myproject_0 0.112 + ps7_0_axi_periph 0.002 + DMA·GPIO·latch·mem_intercon ≤ 0.003: 각 1 mW 미만이라 리포트에 행이 없고 Block_Design_i 잔차로 산출) | **0.029 W** (kws_top), 0.044 W (user_logic 전체, ILA 0.015 포함) | 같은 Vivado 기본 설정(25 °C, 250 LFM, 토글 12.5 %) ; **≈ 4.0× (같은 vectorless 가정끼리, 규모 비교)** ; IP 구성: 클록 0.029 / 신호 0.017 / 논리 0.011 / BRAM 0.030 / DSP 0.025 |
| PL 동적 전력, SAIF 무음 | – | 0.021 W (kws_top) | RTL SAIF, 넷 매칭 6 % ; 차이는 1 mW 해상도 이내 |
| PL 동적 전력, SAIF 키워드 1개 | – (SAIF 테스트벤치 없음) | 0.019 W (kws_top) | 클록 0.009 / 논리 0.005 / 신호 0.002 / BRAM 0.002 |
| PL 동적 전력, SAIF 대화 | – | 0.019 W (kws_top) | |
| 정적 전력 | 0.160 W (칩 전체) | 0.158 W (칩 전체) | 동일 디바이스, 설계 무관 |
| 판정 주기당 연산 | 566,640 MAC / 추론(1단계 hls4ml 24ch, MFCC 49×10 입력, 10클래스; 1 s 창 재계산) | 0.47 M ops/s (0 건너뛰기) ; 같은 log-mel 98×40 입력의 DS-CNN 24ch는 2,265,888 MAC/추론 | 2주차 B-5 (입력이 달라 두 DS-CNN 값은 다른 모델) |

- 비교 범위: 기준 설계의 KWS 계층(myproject_0 + DMA + GPIO + 인터커넥트 + latch)은 **특징 추출을 포함하지 않는다**(MFCC는 PS/사전 계산, 골든 벡터를 DMA로 주입). 우리 kws_top은 FFT·mel 프론트엔드(LUT 906 / FF 785 / BRAM18K 6 / DSP 9, vectorless 0.010 W), VAD, 프리롤, 판정까지 포함한다. 같은 범위에 가장 가까운 행은 'SNN 코어 + 판정'(u_snn + decision_seg: LUT 4,143 / FF 4,865 / BRAM18K 32 / DSP 0, vectorless 0.018 W)이다.
- 신청서의 자원 비교는 HLS 합성 추정치(IP 단독 LUT 36,020 / FF 40,233 / BRAM18K 199 / DSP 63) 기준이었다. 최종보고서는 **구현 결과끼리** 재비교한다: LUT 3.8× / FF 6.0× / BRAM18K 2.8× / DSP 7×. 신청서의 **'BRAM 1/5' 목표는 구현 기준 2.8배(111 → 39)로 미달**이다(HLS 추정 기준으로 계산하면 199/39 = 5.1배였음). Vivado 구현에서 IP의 BRAM이 199 → 109로 줄었기 때문이다.
- 전력은 같은 방법끼리만 비교한다(기준 vectorless ↔ 우리 vectorless). 기준 설계 IP의 **클록 성분 29 mW**(vectorless 리포트 Clocks 항목: 클록 네트·FF 클록 핀의 토글만으로 정해지며 데이터 활동과 무관, 클록 게이팅 없음)는 IP가 켜져 있는 한 소비되는 **하한**이고, 이 값만으로도 우리 게이팅 설계 kws_top의 전체 동적 전력 추정치 **6 mW**(SAIF 듀티 가중)보다 크다. 근거: Clocks 성분은 토글률 가정의 영향을 받지 않으므로 vectorless 값이 곧 실제 클록 전력에 가깝다. 한계: (1) 기준 설계도 BUFGCE로 IP 클록을 게이팅하면 이 하한은 줄어든다(동작 시간 비율만큼), (2) 6 mW는 RTL 활동량(넷 매칭 6 %)을 입힌 추정이지 보드 실측이 아니다, (3) 기준 설계의 BRAM 30 mW·DSP 25 mW 성분은 활동 의존이라 하한에 넣지 않았다.

## 4. 보드 실측 절차 초안
- **측정 위치**: ZSK 보드의 PL 전원 레일(VCCINT 1.0 V) 전류 — 보드 매뉴얼에서 전류 측정용 점퍼/션트(또는 INA 계열 전력 모니터) 유무 확인 필요 **(TODO: ZSK 매뉴얼 확인)**. 없으면 보드 전체 입력(5 V/12 V) 전류를 USB 전력계로 측정하고 모드 간 **차이**만 사용한다(PS·DDR 전력은 공통).
- **순서**(각 60 s, 전류 평균):
  1. 레퍼런스 비트스트림 패스스루 → 기준값 P0
  2. KWS 비트스트림 모드 0(패스스루) → P1 (KWS 로직 정적+유휴)
  3. 모드 2 라이브, 무음(마이크 가림) → P2 (프론트엔드 게이팅 on)
  4. 모드 2 라이브, 무음, 게이팅 off(`T0=0` 빌드 또는 스위치) → P3
  5. 모드 2 라이브, 대화(unknown 위주 60 s) → P4
  6. 모드 2 라이브, 키워드 반복(10 s마다) → P5
  7. (비교) 1단계 hls4ml 비트스트림, 같은 입력 → P6
- 보고: P2−P1(게이팅 대기 전력), P3−P2(게이팅 효과), P4−P1(대화 중 SNN+프론트엔드), P6과의 차이. 전류계 분해능(mA)과 반복 3회 표준편차를 함께 적는다.

## 4. 클록 게이팅 전후 (4-A 후속 4, 2026-10-30 측정)
방법: RTL xsim 9 s(무음·키워드·대화 각 3 s) → 구간별 SAIF → 구현 넷리스트 `report_power`. Vivado는 BUFGCE CE에 확률적 활동량을 받지 않으므로(Power 33-410) CE를 `set_case_analysis`로 0/1 고정한 네 상태를 리포트하고, SAIF에서 실측한 클록 on 듀티 d로 시간 평균을 계산: P = P00 + d_snn·(P10−P00) + d_fe·(P01−P00) — **듀티 가중 추정**. 넷 매칭 6 %(레지스터 출력), 나머지 확률 전파. PL 동적 전력(W), PS7 모델값·ILA(측정용) 제외.

| 설계 / 상태 | 무음 | 키워드 1개 | 대화 | 출처 |
|---|---|---|---|---|
| 기존 RTL(게이팅 없음), kws_top | 0.021 | 0.019 | 0.019 | 10-29 `reports_power/` (게이팅 전 빌드) |
| 기존 RTL + power_opt_design, kws_top | 0.023 | 0.022 | 0.022 | `reports_power_nogate_popt/` — 효과 없음 |
| 게이팅 RTL, 클록 모두 on (ce11 = sw[5] 게이팅 off), kws_top | 0.024 | 0.022 | 0.022 | `*_saif_ce11_hier.rpt` |
| 게이팅 RTL, 클록 모두 off (ce00), kws_top | 0.006 | 0.006 | 0.006 | `*_saif_ce00_hier.rpt` |
| **게이팅 RTL, 듀티 가중 추정, kws_top** | **0.0060** | **0.0062** | **0.0061** | `*_gating_estimate.md` |
| └ 프론트엔드 u_fe: on / 추정 | 0.004 / 0.0020 | 0.004 / 0.0020 | 0.004 / 0.0020 | FE 처리 클록 듀티 1.23 / 1.24 / 1.23 % |
| └ SNN 코어 u_snn: on / 추정 | 0.018 / 0.0020 | 0.017 / 0.0022 | 0.017 / 0.0021 | SNN 클록 듀티 0 / 1.21 / 0.78 % |
| 칩 전체 Clocks 행: on → off | 0.021 → 0.014 | 0.021 → 0.014 | 0.021 → 0.014 | 게이트 클록 트리 2개 = 7 mW |

- 게이팅 후 kws_top 약 **6 mW**: 항상 켜진 부분(I2S 쪽 링 버퍼·E0·VAD·프리롤·판정·제어 ≈ 2 mW)과 클록이 멈춘 SNN/FE 블록의 잔여(BRAM 등 각 2 mW).
- 게이팅 이득: 같은 비트스트림의 sw[5] on/off 비교 0.022~0.024 → 0.006 W(**약 3.7~4.0×**), 게이팅 전 RTL 대비 0.019~0.021 → 0.006 W(약 3.2~3.5×).
- 상시 대기(무음 90 % + 대화 10 %) 평균 **추정**: 게이팅 전 0.9×0.021 + 0.1×0.019 = 0.0208 W → 게이팅 후 0.9×0.0060 + 0.1×0.0061 = **0.0060 W**.
- 비용: BUFGCTRL 4 → 6, LUT/FF 소폭 증가, WNS +3.013 → +3.856 ns(제약 만족), 프레임당 +3 사이클, 골든 일치 유지(회귀 전부 통과).

## 5. 기준 설계(hls4ml DS-CNN, 중간 단계) 전력 측정 절차와 결과 (2026-10-30)
- 비교 조건: 양쪽 모두 **vectorless**(Vivado 기본 토글률 12.5 %, 정적 확률 0.5, 25 °C, 250 LFM, 방열판 없음, typical/commercial), 같은 2023.2, 같은 xc7z020clg484-1. 설정은 `hls4ml_power.tcl`이 `power_flow.tcl`의 vectorless 부분과 동일하게 고정하고, 리포트의 Settings 절을 화면에 찍어 우리 리포트와 대조한다.
- 체크포인트: 중간 단계 통합 프로젝트(hls4ml IP + AXI DMA + AXI GPIO ×4 + AXI Interconnect + kws_out_latch)의 `impl_1/*_routed.dcp`를 `C:/kws/hls4ml_ref/`에 복사. **주의**: 서버에 올라온 `ZSK_Audio_Test_Design.runs`(2026-03-03)는 레퍼런스 오디오 테스트 설계(LUT 2,008 / FF 3,294 / BRAM 17.5 / DSP 0, hls4ml IP·DMA 없음)라 기준 설계가 아니다. 이 설계의 vectorless 값은 공통 부분(오디오 패스스루 + ILA) 참고치: Clocks 7 mW, Signals 2, Logic 1, BRAM 6(ILA), MMCM 121(공통), PS7 1.527(모델), 정적 0.150 W.
- 서버에 복사된 `DSCNN/kws_hls/`는 Vitis HLS 프로젝트(IP export `myproject_prj/solution1/impl/ip/xilinx_com_hls_myproject_1_0.zip`)와 Vitis C 앱(`main.c`: myproject_0 + axi_dma_0 + gpio_ctrl/gpio_res01/gpio_res23/gpio_res4)뿐이고 Vivado 통합 프로젝트(.xpr)·routed.dcp는 없다. IP 단독 **HLS 합성 추정**(`myproject_csynth.rpt`, 2026-06-10, 목표 클록 20 ns = 50 MHz, 추정 17.3 ns): LUT 36,020 / FF 40,233 / BRAM18K 199 / DSP 63, 지연 636,608~637,569 사이클(12.7 ms/추론, dataflow). 이 값은 Vivado 구현치가 아니므로 표에는 "HLS 추정"으로만 병기한다.
- **기준 설계 체크포인트 확인(2026-10-30, 서버 `DSCNN/impl_1/`)**: PC 경로 `C:/FPGA_implementation/ZSK_Audio_Test_Design/vivado_2023_2/ZSK_Audio_Test_Design.runs/impl_1/ZSK_Design_Top_routed.dcp`. hwh 모듈: myproject_0(hls4ml IP), axi_dma_0, axi_mem_intercon, ps7_0_axi_periph, gpio_ctrl / gpio_res01 / gpio_res23 / gpio_res4, kws_out_latch_0, xlconcat_0, clk_wiz_0, proc_sys_reset_0, PS7 + 오디오 레퍼런스(adau1761_interface, user_logic) + ILA. 클록: KWS 논리는 clk_out1 20.833 ns(48 MHz, BUFG 부하 40,047)로 우리와 동일, FCLK0 100 MHz는 PS 전용(부하 1). 구현 리포트(Vivado 기본 설정, 25 °C/250 LFM/typical/commercial = 우리 vectorless와 동일 설정): 전체 LUT 21,716 / FF 38,211 / BRAM 타일 73 / DSP 63, WNS +3.666 ns, 전체 동적 1.785 W 중 PS7 1.530 / MMCM 0.121 / **myproject_0 0.112 W** / ps7_0_axi_periph 0.002 / ILA 0.009 / dbg_hub 0.002 (구현 단계 요약 리포트는 1 mW 미만 계층을 생략 → DMA·GPIO·latch·mem_intercon 값과 계층별 자원은 `hls4ml_power.tcl` 결과로 채움).
- 실행(Vivado Tcl 셸):
```
set argv [list C:/FPGA_implementation/ZSK_Audio_Test_Design/vivado_2023_2/ZSK_Audio_Test_Design.runs/impl_1/ZSK_Design_Top_routed.dcp C:/kws/hls4ml_ref/reports] ; source C:/kws/snn/board/pkg/hls4ml_power.tcl
```
  산출: `reports/vectorless_hier.rpt`, `vectorless_summary.rpt`, `impl_util_hier.rpt`, `impl_util.rpt`, `impl_timing.rpt`, `impl_clock_util.rpt`, `clocks.txt`(클록 주기·소스; 48 MHz = 20.833 ns 존재 여부를 화면에 표시).
- 포함 규칙: 기준 설계의 "KWS 관련 PL 동적 전력" = `Block_Design_i` 아래 hls4ml IP(myproject) + axi_dma + axi_gpio ×4 + axi_interconnect(또는 smartconnect) + kws_out_latch. 오디오 패스스루(adau1761_interface, user_logic), ILA/dbg_hub, PS7, 칩 정적 전력은 양쪽 모두 제외. 우리 쪽은 `user_logic_inst/u_kws`(kws_top) vectorless 0.029 W.
- 해석 주의: vectorless는 두 설계 모두 기본 토글률 가정이라 게이팅·유휴 효과가 반영되지 않는다(기준 설계는 PS가 추론을 시작할 때만 동작하지만 vectorless는 이를 모른다). 따라서 이 비교는 "같은 가정에서의 하드웨어 규모 차이"이고, 동작 기반 저전력 근거는 우리 설계의 SAIF·게이팅 결과(4절)다.

| 항목 | hls4ml DS-CNN 기준(KWS 관련 계층) | SNN KWS (kws_top) | 비고 |
|---|---|---|---|
| 포함 계층 | Block_Design_i/{myproject_0, axi_dma_0, axi_mem_intercon, ps7_0_axi_periph, gpio_ctrl, gpio_res01, gpio_res23, gpio_res4, kws_out_latch_0} ; 제외: adau1761_interface_inst, user_logic_inst(오디오 패스스루), u_ila_0, dbg_hub, PS7, clk_wiz_0(MMCM 0.121 W 공통), 정적 | user_logic_inst/u_kws ; 제외: ILA, I2S, PS7, 정적 | |
| 클록 | 48 MHz (20.833 ns, clk_out1) — HLS 목표는 20 ns였으나 통합은 48 MHz | 48 MHz (20.833 ns) | 동일 |
| LUT / FF | 19,710 / 34,917 (IP 18,072 / 31,649 ; HLS 추정 36,020 / 40,233보다 작음) | 5,173 / 5,795 | 3.8× / 6.0× |
| BRAM18K / DSP | 111 / 63 (IP 109 / 63 ; HLS 추정 199 / 63) | 39 / 9 | 2.8× / 7× |
| PL 동적 전력, vectorless | ≈ 0.116 W (IP 0.112 + periph 0.002 + 나머지 ≤ 0.003) | 0.029 W | ≈ 4.0×, 같은 설정 |
| (범위 맞춤) SNN 코어 + 판정 | – | LUT 4,143 / FF 4,865 / BRAM18K 32 / DSP 0, 0.018 W | 특징 추출 제외 범위끼리 |
- 해석: 이 비교는 "같은 기본 토글률 가정에서의 하드웨어 규모 차이"다. 기준 설계는 PS가 DMA로 추론을 시작할 때만 동작하지만 vectorless는 이를 모르고 상시 동작으로 계산하며, 우리 설계의 VAD·T0·클록 게이팅도 반영되지 않는다. 동작 기반 저전력 근거는 4절(SAIF + 게이팅: kws_top 0.022~0.024 → 0.006 W)이다. 두 설계의 전력을 동작 조건까지 포함해 비교하려면 기준 설계에도 SAIF 테스트벤치(DMA로 MFCC 주입)가 필요하다.
