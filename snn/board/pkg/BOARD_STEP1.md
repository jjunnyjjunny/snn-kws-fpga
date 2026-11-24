# BOARD_STEP1 — 보드 1단계 체크리스트 (Windows PC, Vivado/Vitis 2023.2, ZSK + ADAU1761)

패키지: `snn/board/pkg/` (이 폴더) + `snn/board/ref/`(교수님 레퍼런스 원본, 수정 없음) + `snn/rtl/`, `snn/rtl/fe/`, `snn/rom/`.
PC에 `snn/` 전체를 `C:\kws\snn\`로 복사했다고 가정한다(경로에 한글·공백 없이, 짧게 — 튜토리얼 30쪽 주의와 동일).

## 0. 준비
- [ ] Vivado 2023.2, Vitis(Classic) 2023.2 설치, ZSK JTAG(USB) + UART(115200) + 헤드폰/마이크 연결.
- [ ] `C:\kws\snn\board\pkg\` 에 `ZSK_Design_Top.v`, `user_logic.v`, `rom_paths.vh`, `ila_kws.xdc`, `build_bitstream.tcl`, `reg_set_init.c`, `reg_set_init_16k.c`, `vitis_xsct.tcl` 확인.

## 1. 레퍼런스 원본으로 패스스루 확인 (보드·코덱 생존 확인)
- [ ] 레퍼런스 프로젝트(`FPGA implementation/ZSK_Audio_Test_Design`)의 비트스트림 + `reg_set_init.c`로 튜토리얼 7장대로 실행.
- 기대: 마이크/라인 입력이 헤드폰으로 들림(sw[2]=bypass), UART에 레지스터 리드백 출력(0x4000=0F, 0x4015=01 …).
- 실패 시: I2C 배선/주소(0x3B), 코덱 전원, 튜토리얼 5~7장 재확인. 여기서 안 되면 아래로 가지 않는다.

## 2. 우리 비트스트림 + 48 kHz C로 모드 0·1 확인
빌드(둘 중 하나; 경로는 영문·공백 없음, 압축 푼 위치에 맞게 `C:/kws`을 바꿔 쓴다):
- Windows cmd (settings64.bat 실행 후):
```
cd /d C:\kws\snn\board\pkg
vivado -mode batch -source build_bitstream.tcl -tclargs C:/kws/snn
```
- Vivado Tcl Shell(`Vivado%` 프롬프트) 안에서는 **슬래시 경로 + source** 로 실행한다. 역슬래시 `\`는 Tcl 이스케이프라 `cd C:\kws...`가 깨지고, 셸 안에서 `vivado -mode batch`를 치면 자식 Vivado가 다른 폴더에서 떠서 스크립트를 못 찾는다:
```
cd C:/kws/snn/board/pkg
set argv [list C:/kws/snn]
source build_bitstream.tcl
```
- 산출: `ZSK_Design_Top.bit`, `.ltx`(ILA), `ZSK_Design_Top.xsa`, `reports\impl_util.rpt / impl_timing.rpt / impl_power_vectorless.rpt`.
- [ ] `impl_timing.rpt`에서 WNS ≥ 0 확인(48 MHz). BRAM ≤ 60개, DSP ≤ 20개 예상(추정).
Vitis(xsct):
```
xsct vitis_xsct.tcl C:/kws/snn/board/pkg C:/kws/vitis_ws 48k run
```
(GUI를 쓰려면 튜토리얼 7장과 동일하게 XSA → 플랫폼 → Hello World 앱 → `reg_set_init.c` 임포트 → Program Device → Run.)
- [ ] **모드 0**(sw[1:0]=00): 패스스루 — 1단계와 같은 소리. (sw는 Top에서 반전되므로 "스위치 ON = 1".)
- [ ] **모드 1**(sw[1:0]=01): 골든 특징 주입. sw[4:2]로 세트 선택: 0=yes, 1=no, 2=stop, 3=go, 4=silence. 1초 처리 + 0.5초 휴지 반복.
  - 클록 게이팅(4-A 후속 4): 모드 2에서 **sw[5]=ON이면 게이팅 OFF**(BUFGCE CE 강제 1). ILA `probe21`(fe_clk_ce)·`probe22`(snn_clk_ce)로 게이팅 동작을 눈으로 확인: 무음에서 probe22가 0, 발화 구간에서만 1.
  - ILA(`ZSK_Design_Top.ltx`, 프로브는 전부 `dbg_*` 레지스터): `dbg_result_valid`=1, `dbg_result_class` = 0/1/2/3/10, `dbg_run_count` 증가. 헤드폰: 클래스별 음정의 300 ms 비프(silence는 무음). sw[6]=뮤트, sw[7]=결과 홀드.
  - 실패 시: `snn_active`/`fe_frame_done` 프로브, `rom_paths.vh`의 `.mem` 경로(합성 로그에 `$readmemh` 경고가 있으면 ROM이 0으로 초기화된 것).
- [ ] **모드 2**(sw[1:0]=10, 48 kHz 상태): 아직 48 kHz 입력이므로 프레임/특징이 3배 빠른 시간축 → 인식은 기대하지 않음. `dbg_fe_frame_done`이 3.33 ms마다 뜨는지(16 kHz 설정 전) 확인만.

## 3. 16 kHz C로 LRCLK 62.5 µs 확인
```
xsct vitis_xsct.tcl C:/kws/snn/board/pkg C:/kws/vitis_ws 16k run
```
- [ ] UART: `Address: 4017, Read Data: 03`, `Address: 40f8, Read Data: 03`.
- [ ] ILA 프로브 `i2s_lr`(probe19) 주기 = 3000 클럭(62.5 µs), `dbg_cap_valid`(=new_sample, 1클럭 지연) 간격 3000 클럭, `i2s_bclk` 주기 ≈ 47 클럭(1.024 MHz).
- [ ] 모드 0 패스스루 소리가 여전히 정상(피치 변화 없음 — ADC/DAC 둘 다 16 kHz).
- 실패 판단: `CHECK_16k.md`(2주차) 표 참조. 레지스터는 03인데 LRCLK가 20.8 µs이면 코덱이 마스터가 아닌 것(레퍼런스 0x4015=0x01이라 가능성 낮음) → 48 kHz + RTL 데시메이터로 전환(`fe_decim` 자리, 1주차 67탭 계수 `snn/rom/decim_coef.mem`).
- [ ] 16 kHz에서 **모드 2(라이브)**: "yes/no/stop/go…" 발화 → `dbg_det_valid` 펄스 + `dbg_det_class`(또는 `dbg_live_class`·`dbg_live_count` 누적) + 비프. `dbg_vad_active`가 말할 때만 1, `dbg_cnt_snn_frames/dbg_cnt_frames` ≈ 0.2~0.35(SNN 가동률), `dbg_overrun`=0 확인.

## 4. 모드 3 오디오 캡처 → CSV
- [ ] sw[1:0]=11. Hardware Manager에서 ILA 캡처 모드를 **qualified**, 조건 `dbg_cap_valid == 1`, 깊이 16384(= 1.024 s @16 kHz)로 설정, 트리거는 `dbg_cap_valid` 상승 또는 즉시.
- [ ] 마이크에 1초 동안 키워드를 말하고 캡처 → 파형 창에서 `dbg_cap_sample[17:0]`를 CSV로 내보내기(Export ILA Data → CSV). `board_capture_check.py --col dbg_cap_sample`.
- [ ] CSV를 `snn/data/board_capture/<이름>.csv`로 서버에 업로드 → `python snn/eval/board_capture_check.py snn/data/board_capture/<이름>.csv`
  - 리포트: 레벨(dBFS, 피크/RMS), 특징 0 비율, 학습 분포와의 차이, 1초 창 예측, 추천 PGA 조정.

## 실패 시 공통 확인
| 증상 | 확인 |
|---|---|
| 비트스트림 빌드 실패 | 경로 길이/한글, `Block_Design.tcl`의 Vivado 버전(2023.2 고정), `$readmemh` 경로(`rom_paths.vh`) |
| ILA 프로브 net 없음 | 합성 후 net 이름 변경: `ila_kws.xdc`의 `user_logic_inst/<net>`을 `get_nets -hier *<net>*`로 확인해 수정 |
| 모드 1 결과가 항상 같은 클래스 | `.mem`이 0으로 초기화됨(합성 로그 `$readmemh` 경고) → 절대 경로 확인 |
| 모드 2 검출 없음 | `dbg_energy` 프로브로 말할 때 E가 T_ON(600)보다 큰지; 작으면 PGA 게인(0x400A/0x400C 레코드 믹서 게인) 상향 |
| 오검출 많음 | `MARGIN` 파라미터 상향(user_logic 파라미터), 또는 T_ON 상향 |

## 6. 클록 게이팅 전후 비교 (4-A 후속 4)
빌드 변형(`build_bitstream.tcl` 두 번째 인자): `""` 기본(게이팅 RTL, BUFGCE 2개) / `nogate` 게이팅 없는 RTL / `popt` power_opt_design 추가 / `nogate_popt`.
```
set argv [list C:/kws/snn nogate_popt] ; source build_bitstream.tcl      ;# 1. 빠른 실험: 기존(게이팅 없음) + power_opt
set argv [list C:/kws/snn]             ; source build_bitstream.tcl      ;# 2. 게이팅 RTL (reports/, ZSK_Design_Top.bit 덮어씀)
set argv [list C:/kws/snn nogate_popt 0] ; source power_flow.tcl        ;# SAIF 전력: 변형, 게이팅 off 스위치(0/1)
set argv [list C:/kws/snn "" 0]          ; source power_flow.tcl        ;# 게이팅 RTL, 게이팅 on
set argv [list C:/kws/snn "" 1]          ; source power_flow.tcl        ;# 게이팅 RTL, sw[5]로 게이팅 off (같은 비트스트림)
```
확인: `reports[_variant]/impl_clock_interaction.rpt`(게이트 클록이 같은 clk_48로 분석되는지, unsafe 없음), `impl_cdc.rpt`(새 CDC 경고 없음), `impl_clock_util.rpt`(BUFG 개수), `impl_timing.rpt` WNS ≥ 0.
보드: 같은 비트스트림에서 sw[5]만 바꿔 PL 전류(가능하면)·ILA probe21/22 비교.

## 5. 입력 게인 실험 (4-B B-2, C 코드만 변경, 비트스트림 그대로)
보드 캡처 분석(`snn/runs/board_capture/summary.md`): 20~30 cm 발화 peak −31~−43 dBFS(데이터셋 test 평균 −6 dBFS), 3 cm −11~−17 dBFS; 조용한 방 바닥 −73~−76 dBFS; 특징 바닥(log offset 332)은 정현파 −55 dBFS. 즉 입력 레벨이 데이터셋보다 15~35 dB 낮아 /s/ 같은 약한 성분이 특징에서 0으로 잘린다 → 코덱 게인을 올리는 실험. 파일은 `gen_codec_variants.py`가 `reg_set_init_16k.c`에서 생성(값만 바꾸고, PLL lock 대기 추가, 바뀐 레지스터 리드백 추가).
```
xsct C:/kws/snn/board/pkg/vitis_xsct.tcl C:/kws/snn/board/pkg C:/kws/vitis_vA 16k_vA run     ;# vA (워크스페이스는 변형마다 새 이름)
```
| 변형 | 바뀐 레지스터(기대 리드백) | 뜻 | 모드 2에서 볼 것(20~30 cm 보통 목소리) | 무음 E1 기대값(골든, 조용한 방 바닥 −73~−76 dBFS 기준) |
|---|---|---|---|---|
| v0 | 없음. 4002 끝 바이트 03, 400B/400D **05** | 레퍼런스 경로(LINN/RINN → 레코드 믹서 0 dB, LAUX 0 dB 섞임, PGA 뮤트) + PLL lock 대기 | 콜드 부팅(전원 껐다 켜고) 첫 실행에서 400B/400D가 05인지. `dbg_energy` 피크 지금과 같은 ≈ 900 | 게인 0: 0이 아닌 밴드 0, **E1 = 0** (T0 게이팅은 DC 오프셋 −1,200 LSB18 때문에 안 걸림: Σ\|x18\| ≈ 5.8e5 ≫ 6,694) |
| vA | 400A/400C **0F**, 400B/400D **00**, 4019 **23** | 믹서 게인 +6 dB(최대), LINE IN(AUX) 뮤트, ADC HPF 켜기 | 피크 E1 ≈ 1.3~2×(추정, 0이 아닌 밴드마다 +32), 패스스루 잡음 감소 여부, stop 5회 | +6 dB: 밴드 0, **E1 = 0**; HPF로 DC 제거(바닥 −74 dBFS, 그래도 Σ\|x18\| ≈ 1e4 > 6,694라 T0 게이팅은 안 걸림) |
| vB | 400A/400C **01**, 400B/400D **08**, 400E/400F **83**, 4019 23 | 믹서 입력 뮤트, 차동 PGA 경로 부스트 0 dB + 볼륨 +12 dB | 소리가 정상이면 잭이 PGA(LINN/LINP) 경로에 연결된 것: E1 ≈ 2~3×. **소리가 없거나 잡음뿐이면 PGA 경로 불가** → vA로 확정 | +12 dB: 밴드 0, **E1 = 0** |
| vC | 400B/400D **10**, 400E/400F **63** (나머지 vB와 같음) | PGA 부스트 +20 dB + 볼륨 +6 dB ≈ +26 dB | 20~30 cm에서 E1 1500~3000(데이터셋 수준), yes/no/stop/go 각 5회. 3 cm는 클리핑(찌그러짐) 가능 | +26 dB: 밴드 최대 1~6, **E1 최대 46~66, 중앙값 16~30** → T_off 360 대비 5~7×, T_on 600 대비 9~13× 여유, VAD 미발동. 합격선: E1 max < 180 |
| vD | vC + 4011 **33**, 4012 **2C**, 4013 **34**, 4014 **28**, 400E/400F **43** | PGA + ALC(코덱 AGC, 노이즈 게이트) | 거리를 바꿔도 E1 피크가 비슷해지는지, 무음에서 E=0 유지되는지. **ALC 비트 배치는 데이터시트로 확인 전**(리드백이 쓴 값과 다르면 알려 줄 것) | ALC 게인이 +32 dB를 넘으면 백색잡음 −70~−73 dBFS에서 E1 p99 215~534(주의~불합격), +38 dB면 VAD 상시 발동. 노이즈 게이트가 동작해야 함 → 무음 E1 max < 180 확인 필수 |

각 변형에서 기록: UART 리드백 전체, 모드 0 소리(먹먹함·잡음), **모드 2 무음(마이크 가림·조용한 방) 30 s 동안 `dbg_energy` 최대값**, 모드 2 `dbg_energy` 피크(3 cm / 10 cm / 30 cm 각 3회), stop·yes·no·go 5회씩 판정. **무음 E1 합격선: 30 s 최대 < 180(T_off 360의 절반), 180~360은 주의(T_on 600에 2× 미만 여유), 4프레임 연속 > 600이면 불합격(VAD 상시 발동)** — 표 전체는 `snn/runs/board_capture/silence_e1.md`(소스: quiet_01 캡처 DC 포함/제거, 백색잡음 −76/−73/−70 dBFS × 게인 0~+38 dB). 순서 v0 → vA → vB → (vB가 되면) vC → vD. vC에서 30 cm가 데이터셋 수준이면 vC를 BOARD_STEP2의 기준 설정으로 쓴다.

## 7. ROM(.mem) 교체 후 재빌드 (B-3 재학습 가중치 반영)
가중치 ROM은 `$readmemh`로 합성 시 읽히므로 .mem만 바꿔도 **합성·구현을 다시** 해야 한다. 절차(서버에서 .mem 교체·회귀 통과 후):
1. 서버: `snn/rom/m3k3_*.mem`(새 가중치, 이전 것은 `snn/rom/archive/<날짜>_m3k3/`)과 `snn/golden/vectors/` 갱신본을 포함해 `snn_hw.tgz` 재생성 → PC `C:\kws\snn`에 덮어쓰기(rom/ 과 golden/ 만 바뀜).
2. Vivado Tcl 셸: `set argv [list C:/kws/snn] ; source build_bitstream.tcl` (프로젝트 폴더 `vivado_kws`를 지우고 새로 만든다. 지난 빌드 기준 합성+구현 약 25~40분, 추정).
3. 산출 `ZSK_Design_Top.bit/.ltx/.xsa` 교체 → Hardware Manager 재프로그램. XSA의 PS 설정은 그대로이므로 Vitis 플랫폼은 다시 만들지 않아도 되지만, 새 XSA로 `xsct vitis_xsct.tcl ... 16k_vC run`을 다시 돌리면 가장 확실하다.
4. 모드 1 골든 주입 5세트 확인(새 가중치의 기대 클래스는 `snn/golden/vectors/m3k3/*/info.json` 또는 서버 회귀 로그 참조) → 모드 2.
