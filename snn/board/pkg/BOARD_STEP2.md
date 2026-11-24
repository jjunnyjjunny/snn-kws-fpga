# BOARD_STEP2 — 보드 라이브 측정 프로토콜 (최종보고서용, 4-B B-4)

전제: BOARD_STEP1의 1~4단계 통과(모드 1 골든 주입 5세트 일치, 16 kHz LRCLK 3000 clk, 모드 3 캡처 가능), 입력 게인 실험(BOARD_STEP1 §5)에서 고른 C 변형으로 코덱 초기화. 기록 양식 `snn/board/log/board_log_template.csv`(280슬롯, 23분) / `_short.csv`(50슬롯, 4분), 채점 `snn/eval/board_log_confusion.py`.

## 0. 공통: ILA를 자동 로거로 쓰기 (비트스트림 변경 없음)
모드 2에서 `dbg_det_valid`는 유효 검출마다 1클록 펄스이고, 같은 ILA 행에 `dbg_det_class`, `dbg_cnt_frames`(10 ms 프레임 카운터 = 타임스탬프), `dbg_live_count`, `dbg_cnt_snn_frames`가 함께 찍힌다. 따라서
1. Hardware Manager → hw_ila_1 → Settings: Capture mode **BASIC**, Capture condition `probe4 (dbg_det_valid) == 1`, Trigger mode BASIC + Trigger condition 같은 `probe4 == 1`, Trigger position 0, Data depth 1024.
2. 세션 시작 직전 **Run Trigger**. 이후 모든 검출이 한 행씩 쌓인다(최대 1024행 = 한 세션 충분). 세션이 끝나면 **Stop**(파형 창 정지) → `Export ILA data → CSV` 로 `snn/board/log/<세션>_det.csv` 저장. 행이 없으면(검출 0) 창은 비어 있음 → 그 세션은 "검출 0"으로 기록하고, 가동률 읽기용으로 Trigger condition을 `probe1 == 1`로 바꿔 한 번 즉시 트리거해 `dbg_cnt_frames`/`dbg_cnt_snn_frames`를 읽는다.
3. 발화는 **5초 간격 메트로놈**(폰 타이머)으로 템플릿 순서대로. 시작 신호 시각에 ILA가 돌고 있으면 `--t0` 생략 가능(첫 검출 − 2.5 s를 0번 슬롯 시작으로 간주). 첫 슬롯이 무반응일 수 있으면 시작 신호 직전에 `probe1==1`로 한 번 즉시 트리거해 그때의 `dbg_cnt_frames` 값을 적어 두고 `--t0 <값>`으로 준다.
4. 채점: `python snn/eval/board_log_confusion.py --log snn/board/log/<세션>.csv --det snn/board/log/<세션>_det.csv --cadence 5 --out snn/board/log/<세션>_result.md` → 혼동행렬(발화 단어 × 판정 클래스/miss/multi), 키워드 정검출·오분류·무반응, unknown/none 슬롯의 오검출, 거리별 정검출.

## 1. 인식률 — 두 가지 시험을 구분한다
| | 1A 사람 발화 시험 | 1B 재생 시험 |
|---|---|---|
| 목적 | **데모**·실사용 감각(화자 1명, 거리·억양 변동) | **정량**·재현 가능(같은 test 클립, 같은 레벨 → 골든 상한과 직접 비교) |
| 입력 | 사람이 5 s 메트로놈으로 발화 | `snn/board/play/playback_test.wav`를 스피커로 재생(210슬롯 × 2.5 s = 8.75 분 + 리드 3 s) |
| 정답 | `board/log/board_log_template*.csv` | `snn/board/play/playback_answers.csv`(생성 시 고정, seed 1) |
| 채점 | `snn/eval/board_log_confusion.py` | `snn/eval/playback_score.py`(오프셋 자동 정렬) |
| 비교 기준 | 4-A test 스트림(조건 다름을 명시) | 같은 wav의 디지털 골든 `playback_golden_result.md`(상한) |

### 1A. 사람 발화 시험 (데모)
| 세션 | 동작점 sw[3:2] | 양식 | 내용 | 시간 |
|---|---|---|---|---|
| A0 | 00 (마진 0) | template.csv | yes/no/stop/go × 20회 × {10 cm, 30 cm} + up/down/left/right/on/off × 10회(10 cm) + unknown 20단어 × 2회 + 무발화 슬롯 20 | 23분 |
| A2 | 10 (마진 6212) | template.csv | 같은 280슬롯 | 23분 |
| A1, A3 | 01, 11 | template_short.csv | yes/no/stop/go × 10회(10 cm) + unknown 10단어 | 4분씩 |

- unknown 20단어(Speech Commands의 비키워드): bed, bird, cat, dog, happy, house, marvin, sheila, tree, wow, zero, one, two, three, four, five, six, seven, eight, nine.
- 거리: 입과 마이크 사이 10 cm / 30 cm(자 사용), 보통 목소리, 방 소음은 평소 실험실.
- 같은 세션 안에서 코덱 C 변형·스위치를 바꾸지 않는다. 세션 사이에는 ILA를 다시 Run Trigger.
- 보고 표: 동작점별 {키워드 정검출률, 오분류율, 무반응률, unknown 오검출률, 거리별 yes/no/stop/go 재현율}. 4-A test 스트림 결과(val_talk 규칙으로 고른 U2: 검출 0.894, FA/h 106.9)와 나란히 두되, 조건이 다름(실제 마이크·화자 1명)을 명시. 데모용이므로 "정확도"로 보고하지 않고 "사람 발화 반응"으로 적는다.

### 1B. 재생 시험 (정량, 재현 가능)
패키지(`snn/board/play/`, 생성 `make_playback_test.py`): `playback_test.wav`(16 kHz mono 16-bit; test 클립 yes/no/stop/go × 25, up/down/left/right/on/off × 10, unknown 40, silence 10 = 210슬롯, 2.5 s 간격, 클립은 슬롯 시작 0.5 s 뒤, 리드 3 s), `playback_answers.csv`, 보정용 `cal_tone_1k_-6dBFS.wav`(1 kHz, 피크 −6 dBFS, 10 s)·`cal_ref_clips.wav`(기준 'yes' 클립 5개), 골든 상한 `playback_golden_det.csv` / `playback_golden_result.md`.

1. **스피커·볼륨 보정**(레벨을 데이터셋과 맞추는 것이 이 시험의 핵심): 스피커(폰/노트북)를 마이크 앞 10~20 cm에 고정 → `cal_tone_1k_-6dBFS.wav` 재생 → 모드 3(ILA `dbg_cap_valid==1`, 16384행 = 1.02 s) 캡처 → CSV 저장 → `python snn/eval/capture_level.py <csv>` → 피크가 **−10 ~ −6 dBFS** 에 들 때까지 볼륨·거리 조정(클리핑 −1 dBFS 이상 금지). 톤이 맞으면 `cal_ref_clips.wav`로 한 번 더 확인(클립 피크 평균이 같은 창에 들면 끝; 톤과 클립은 crest factor가 달라 2~4 dB 차이 가능 → 클립 기준을 우선). 보정 캡처 CSV 2개를 `snn/board/log/cal_<날짜>.csv`로 보관하고 결과 줄을 세션 기록에 적는다.
2. **ILA 로거 설정**은 §0과 같다(캡처 조건 `dbg_det_valid == 1`, depth 1024 ≥ 210 슬롯). 재생 직전에 Run Trigger. 가능하면 재생 시작 순간에 `probe1==1` 즉시 트리거로 `dbg_cnt_frames`를 읽어 `--t0`로 준다(없어도 자동 정렬된다: 같은 클래스 적중이 최대가 되는 오프셋을 0.05 s 격자로 찾음).
3. **세션**: R0(sw[3:2]=00, 마진 0)와 R2(10, 마진 6212)에서 각 1회 전체 재생(8.75 분). 추가로 R0을 한 번 더(R0b) 재생해 재현성(두 번의 정검출률 차이)을 적는다. 재생 중 말하지 않는다.
4. **채점**: `python snn/eval/playback_score.py --det snn/board/log/R0_det.csv [--t0 N] --out snn/board/log/R0_result` → 키워드 정검출/오분류/무반응, unknown·silence 오검출(→ FA/h), 슬롯 정확도, 혼동행렬. 골든 상한은 `python snn/eval/playback_score.py --det snn/board/play/playback_golden_det.csv --margin 0`(또는 `--margin 6212`)로 같은 형식으로 나오므로 두 표를 나란히 둔다.
5. **보고**: 동작점별 {보드 정검출률 / 골든 정검출률, 오분류, 무반응, 비키워드 오검출, 슬롯 정확도}와 보정 레벨(캡처 피크 dBFS). 보드와 골든의 차이는 (스피커·마이크·방 채널 + 코덱 레벨) 효과다. 레벨이 창 안인데 차이가 크면 보정 캡처와 재생 중 모드 3 캡처 1~2개를 서버로 보내 골든에 넣어 확인한다.

## 2. 오검출 (조용한 방 10분, 대화 10분)
| 세션 | 조건 | 동작점 | 측정 |
|---|---|---|---|
| Q0 / Q2 | 조용한 방(장비 팬 소음 포함), 발화 없음 | 00 / 10 | 10분 동안 비프 횟수 + ILA 검출 행 수 |
| T0 / T2 | 30 cm 거리에서 한국어·영어 대화(유튜브 뉴스 등)를 보통 음량으로 재생 | 00 / 10 | 10분, 같은 방식 |

채점: `python snn/eval/board_log_confusion.py --det snn/board/log/Q0_det.csv --fa_minutes 10` → FA/h와 검출 클래스·시각 목록. 비프 횟수(귀)와 행 수가 다르면 적어 둔다(비프는 sw[6]로 끌 수 있음).

## 3. 가동률·게이팅
- 모든 세션 CSV의 마지막 행에 `dbg_cnt_snn_frames / dbg_cnt_frames`가 있다 → 스크립트가 SNN 가동률(프레임 기준)을 출력. 세션 종료 시 ILA에 행이 없으면 §0-2의 즉시 트리거로 한 번 읽는다.
- 게이팅 증거: 세션 중 한 번, Trigger `probe3 (dbg_vad_active)` 상승 에지, depth 16384, 캡처 모드 ALWAYS로 파형을 잡아 `probe21 fe_clk_ce` / `probe22 snn_clk_ce` / `probe3` / `probe7 snn_active`를 한 화면에 저장(보고서 그림). sw[5]=ON(게이팅 OFF)으로 같은 캡처를 한 번 더: 두 ce 프로브가 항상 1이어야 한다.
- 카운터는 24비트(ILA용): `cnt_frames`는 46시간, `cnt_snn_frames`도 충분. 리셋은 보드 리셋 버튼(rst_48).

## 4. 전력 (보드 실측, Keysight N6705C 데이터 로거)
ZSK의 ADAU1761/핀 배치는 ZedBoard(xc7z020clg484, AC_* 핀 동일)와 같다. ZedBoard 계열은 **온보드 전류 모니터가 없고**, XADC는 내부 전압(VCCINT/VCCAUX/VCCBRAM)과 온도만 읽을 수 있어 **PL 전류를 XADC로 재는 것은 불가**(VP/VN에 외부 션트를 달면 가능하지만 하드웨어 개조). 따라서 보드 **입력 전력의 차분 측정**만 가능하다.

- 장비: Keysight N6705C DC 전원 분석기. 보드 12 V 입력 잭을 N6705C 출력 채널(12 V, 전류 한계 2 A 이상)로 공급하고 **데이터 로거**(Data Logger)로 전압·전류를 기록한다. 로거 주기 0.1 s(권장; 0.05~0.5 s 모두 가능), 전체 길이 약 23분(7상태 × 60 s × 3회 + 전환 5 s). 끝나면 CSV로 내보낸다(시간, 전압, 전류; 전력 열이 있으면 그대로 사용). 어떤 열 이름이든 스크립트가 자동 인식하며, 안 되면 `--cols 시간,전압,전류` 열 번호로 지정한다.
- 측정 상태(같은 비트스트림, 같은 C 변형, 이어폰 연결 유지, sw[6] 비프 OFF). 각 60 s 유지, P0→P6 순서를 **3회 반복**(반복 간 차이 = 드리프트·재현성):

| 상태 | 모드 | 입력 | 게이팅 sw[5] | 뜻 |
|---|---|---|---|---|
| P0 | 0 패스스루 | 무음 | – | 기준(KWS 로직 유휴) |
| P1 | 2 라이브 | 무음 대기(조용한 방) | ON(sw[5]=OFF) | 상시 대기 전력 |
| P2 | 2 라이브 | 연속 키워드 발화(2.5 s마다) | ON | 발화 중 전력 |
| P3 | 2 라이브 | 연속 대화(unknown 문장 낭독) | ON | 대화 중 전력 |
| P4 | 2 라이브 | 무음 대기 | OFF(sw[5]=ON) | 게이팅 효과(P1 − P4) |
| P5 | 2 라이브 | 연속 키워드 발화 | OFF | 게이팅 효과(P2 − P5) |
| P6 | 2 라이브 | 재생 시험 wav 재생(1B 조건) | ON | 실사용 조건 평균 |

- 절차: ① 로거 시작, 로거 화면의 경과 시간(또는 시계)을 기준으로 ② 상태를 바꿀 때마다 `snn/board/log/power_marks.txt`에 `상태 시작시각` 한 줄을 손으로 적는다(양식 `power_marks_template.txt`, 초 단위 또는 `HH:MM:SS.s`). ③ 60 s 유지 후 다음 상태. ④ 3회 반복 후 로거 정지, CSV 내보내기 → `snn/board/log/power_dlog.csv`.
- 분석: 구간 앞 5 s(전환·정착)와 뒤 2 s를 빼고 구간별 평균 전력, 반복 간 표준편차, 기준(P0) 대비 차분과 게이팅 쌍 차분을 표로 낸다.

```
python snn/eval/power_log_analyze.py --log snn/board/log/power_dlog.csv --marks snn/board/log/power_marks.txt \
       --dur 60 --skip 5 --trim 2 --ref P0 --pairs P1-P4,P2-P5 --resolution 1 --fig --out snn/board/log/power_result
```
  출력 `power_result.md/.json/.png`: 상태별 반복 수·전압·전류·전력 평균·반복 간 표준편차·구간 내 표준편차·기준 대비 차분(± 불확도), 쌍 차분(P1 − P4, P2 − P5), 기준 드리프트, 전력 시계열과 구간 음영 그림. `--resolution`은 로거 전력 분해능(mW; 12 V × 전류 분해능, N6705C 2 A 레인지 전류 분해능 약 0.1 mA → 1.2 mW)이다.
- 판정: |차분| < max(2 × 불확도, 분해능)이면 **"측정 해상도 이하"**로 기록하고 Vivado 추정표와 섞지 않는다(보고서 3.5절에 별도 표). 기대 크기: Vivado 추정으로 PL 동적 전력 차이는 수 mW(게이팅 ON/OFF 약 15 mW 추정)이며 PS7(1.5 W 이상)에 묻힌다. 12 V에서 1 mA = 12 mW.

| 상태 | 전압 V | 전류 mA | 전력 mW | 반복 간 표준편차 | 기준 대비 차분 mW | 비고 |
|---|---|---|---|---|---|---|
| P0 모드 0 | | | | | 기준 | |
| P1 대기, 게이팅 ON | | | | | | |
| P2 발화, 게이팅 ON | | | | | | |
| P3 대화, 게이팅 ON | | | | | | |
| P4 대기, 게이팅 OFF | | | | | | P1 − P4 = |
| P5 발화, 게이팅 OFF | | | | | | P2 − P5 = |
| P6 재생 시험, 게이팅 ON | | | | | | |

## 5. (선택) 자동 로깅 — PS가 읽는 방식
§0의 ILA 로깅으로 인식률·오검출 세션은 모두 처리되므로 비트스트림 변경 없이 끝낼 수 있다. ILA 없이 UART로 바로 찍고 싶다면: user_logic의 `det_valid/det_class/live_count`를 AXI GPIO(블록 디자인에 IP 추가, 32비트 입력 1개)로 PS에 연결하고 reg_set_init C에 폴링 루프(live_count가 바뀌면 `xil_printf("%d %d\r\n", frames, class)`)를 추가하는 것. 비용: Block_Design.tcl 수정 + 재합성·재구현(약 30분, 추정) + XSA 재생성 + Vitis 플랫폼 재생성. ROM 교체 재빌드(B-3)가 생기면 그때 묶어서 하는 것이 맞고, 단독으로는 하지 않는다.

## 6. 기록·제출물
`snn/board/log/`에 세션별 `<세션>.csv`(양식), `<세션>_det.csv`(ILA), 결과 `.md`; 재생 시험 `R0_det.csv`/`R2_det.csv`/`R0b_det.csv` + 보정 캡처 `cal_*.csv`; ILA 파형 PNG 3장(게이팅 ON/OFF, 16 kHz LRCLK); 전력 로그 `power_dlog.csv` + `power_marks.txt`(→ `power_result.md`). 받으면 final_report_draft 3.2(보드 인식·오검출: 1B 재생 시험이 정량 수치, 1A는 데모 반응), 3.5(보드 전력 표), 발표 13~14장을 채운다.
