# snn-kws-fpga — A Spiking Neural Network for Low-Power Keyword Spotting, Implemented Directly in FPGA RTL
 
A keyword spotter in which the whole path — microphone → integer front end (log-mel) → spiking neural network (SNN) → segment decision — is **hand-written integer RTL** running in the PL of a Zynq-7020 (ZSK board, ADAU1761 codec).
A Python integer golden model is the reference, and the RTL **matches it bit for bit** (simulation regression + golden-feature injection on the board).
 
- Dataset: Google Speech Commands v2, 12 classes (yes/no/stop/go/up/down/left/right/on/off + silence + unknown)
- Model: M3k3 — causal dilated convolution (K=3, dilation 1/2/4/8) + LIF neurons, 64 × 4 layers + FC 64→12, 8-bit QAT + knowledge distillation, 45,312 parameters
- Front end: 16 kHz, 30 ms window / 10 ms hop, Hann · FFT512 (18-bit) · mel40 · log2 (Q6.4), 8-bit features
- Decision: the SNN runs only inside segments opened by a frame-energy hysteresis VAD; argmax of the segment sum; operating-point switch (margin 0/2838/6212/9633)
- Low power: VAD event-driven operation + two BUFGCE clock gates (SNN core, front-end processing block), single 48 MHz clock
Demo video: https://drive.google.com/file/d/1R1LHB-HFQxO7KCmjMQhrcChLzLRV3T3r/view?usp=sharing
 
## 1. System architecture
 
![Figure 1. System block diagram](snn/docs/figures/final/fig01_system_block.png)
 
24-bit PCM from the codec → integer front end (40 8-bit features per frame) → VAD (E1 = Σ features, 20-frame pre-roll) → SNN core (event-driven, zero-input skipping) → segment decision → LED/beep. The weight ROMs (`.mem`) are generated from the golden model. All 13 figures with captions are in [`snn/docs/figures/final/index.md`](snn/docs/figures/final/index.md).
 
## 2. Repository layout
 
```
snn/
  frontend/   Integer front-end golden model (frontend_ref.py, fe_config.py), ROM generation (export_roms.py), verification
  golden/     SNN integer golden (snn_int_ref.py), full-pipeline golden (pipeline_int_ref.py), .mem/vector generation
              vectors/  Golden vectors for RTL regression (12 final-ROM m3k3 clips + streaming, pipeline and activity streams)
  models/     PyTorch training models (LIF, QLinear, M3k3); baseline/ DS-CNN comparison models
  data/       Dataset → integer-feature npz (make_dataset.py), split lists (filelist_*.json)
  train_snn.py, run_*.sh   Training (float → 8-bit QAT+KD), per-week run scripts
  eval/       Stream/VAD evaluation (vad_eval.py), detection curves, confusion matrices, level sweeps, playback scoring
              (playback_score.py, playback_compare.py), power-log analysis (power_log_analyze.py), rail currents (power_rails.py)
  rtl/        Hand-written Verilog: kws_top.v (top), fe/ (front end), snn_core.v · lif_update.v · row_acc.v …,
              vad.v, preroll_buf.v, decision_seg.v, clk_gate_buf.v (BUFGCE gating); OOC synthesis tcl/xdc, Yosys scripts
  tb/         Icarus/Verilator testbenches and regression scripts (clips, streaming, front end, pipeline, activity)
  rom/        Final ROM (U2, prefix m3k3) — 14 .mem files + front-end ROMs (Hann, twiddle, mel40, log2), format README
  board/pkg/  Board package: user_logic.v (replacement for the reference design's user_logic), ila_kws.xdc,
              build_bitstream.tcl, vitis_xsct.tcl, codec init C (reg_set_init_16k_v0/vA/vB/vC.c), power-flow tcl,
              board procedures (BOARD_STEP1/2.md)
  board/play/ Playback-test generation (make_playback_test.py), answer key, digital golden results, calibration-tone wav
  board/results/  Board playback-test scoring (R0/R2/R0b), per-slot comparison, power-log template
  docs/       Report figure scripts; figures/final/ 13 PNGs at 300 dpi + captions
  results/    Run summary (summary.md), Vivado implementation and power (vivado_impl.md), model comparison, confusion json
```
 
**Dependencies (not included in this repository)**: the Speech Commands v2 dataset, training checkpoints (.pt), Vivado/Vitis build outputs, and the **Libertron ZSK audio reference design** (`ZSK_Design_Top.v`, `adau1761_interface.v`, I2S, constraints, codec init C). To build for the board, take that reference design, replace only `user_logic` with `board/pkg/user_logic.v`, and apply `board/pkg/ZSK_Design_Top.v.diff` to the top level.
 
## 3. Reproduction
 
Environment: Python 3.11, PyTorch (CUDA), numpy, soundfile, librosa; Icarus Verilog 13 / Verilator 5 (simulation); Vivado/Vitis 2023.2 (board). Run every command from the repository root (the scripts assume `snn/...` paths).
 
1. **Integer golden model and ROMs**
```
python snn/frontend/export_roms.py            # front-end ROMs (.mem)
python snn/frontend/verify_frontend.py --n 100 # check the integer front end against the float reference
```
2. **Dataset → features** (pass the Speech Commands v2 path as an argument to `snn/data/make_dataset.py`)
```
python snn/data/make_dataset.py --classes 12 --n_mels 40 --win 480 --aug_copies 2
```
3. **Training** (60 float epochs → 8-bit QAT + KD; U2 = 3× unknown data + loss weighting)
```
python snn/train_snn.py --model M3 --k 3 --epochs 60 ...        # see run_week1.sh / run_week4_A.sh
bash snn/run_qat.sh <float_run>
python snn/golden/export_snn_mem.py --run snn/runs/<run> --prefix m3k3   # .mem + params.json
python snn/golden/dump_vectors.py   --run snn/runs/<run> --prefix m3k3   # RTL golden vectors
```
   `snn/rom/m3k3_*.mem` in this repository is the final ROM loaded on the board (U2 s0), so steps 4 and 5 work even if you skip this step.
4. **RTL simulation** (bit-exact regression against the golden model)
```
cd snn/tb && ./run_all_vecs.sh m3k3 1          # 12 clips, zero-skip on
cd snn/tb/pipeline && ./run_pipeline.sh ../../golden/vectors/pipeline_v3/stream_0 48   # front end + VAD + SNN + decision
cd snn/tb/fe && ./run_fe_blocks.sh             # front end, block by block
cd snn/tb/activity && ./run_activity.sh        # activity and clock-CE duty
```
5. **Vivado build and board** (Windows, follow `BOARD_STEP1.md`)
```
vivado -mode batch -source snn/board/pkg/build_bitstream.tcl -tclargs <repo>/snn
xsct snn/board/pkg/vitis_xsct.tcl <repo>/snn/board/pkg <ws> 16k_vC run      # app with codec setting vC (+26 dB)
```
   Switches: SW1·SW0 select the mode (passthrough / golden injection / live / capture), SW3·SW2 the margin, SW5 disables gating, SW6 mutes the beep. The playback test and power logging are described in `BOARD_STEP2.md`; scoring is done with `snn/eval/playback_score.py`.
 
## 4. Key results
 
**Accuracy** (Speech Commands v2 test, per clip)
 
| Model | 12-class | 10-class | Notes |
|---|---|---|---|
| DS-CNN 24ch (float, same integer-feature input) | 89.08 ± 0.17 % | 91.66 % | 3 seeds |
| SNN M3k3 8-bit QAT+KD (base) | 87.67 ± 0.43 % | 89.59 % | 3 seeds, integer golden inference |
| SNN M3k3 U2 (final ROM, unknown-augmented) | 85.41 % (s0) | – | unknown recall 92.6 % (DS-CNN 66.1 %) |
 
**Implementation resources and power** (Vivado 2023.2, xc7z020clg484-1, 48 MHz, implemented results compared)
 
| | hls4ml DS-CNN KWS layers (stage 1) | This design, kws_top (FE+VAD+SNN+decision) | Ratio |
|---|---|---|---|
| LUT / FF | 19,710 / 34,917 | 5,173 / 5,795 | 3.8× / 6.0× |
| BRAM18K / DSP | 111 / 63 | 39 / 9 | 2.8× / 7× |
| PL dynamic power, vectorless (estimate) | ≈ 0.116 W | 0.029 W | ≈ 4× |
| PL dynamic power, SAIF activity (estimate) | – | 0.019–0.024 W (gating off) → **≈ 0.006 W** (gating on, duty-weighted) | 3.7–4× |
 
All power figures are Vivado model estimates; measuring board power (N6705C, `snn/eval/power_log_analyze.py`) is future work. Clock-gating CE behavior was confirmed on the board with the ILA (`fig12`).
 
**Board playback test** (210 slots of test-clip wav played through a PC speaker, codec setting vC, 8 cm, chain gain +1 dB; `snn/board/results/`)
 
| | Margin 0 | Margin 6212 |
|---|---|---|
| Digital golden (same wav), keyword detection | 83.8 % (FA 3/50) | 56.9 % (FA 0/50) |
| Board R0 / R0b | 70.0 % / 72.5 % (FA 9/50 / 6/50) | R2: 41.2 % (FA 1/50) |
| R0–R0b per-slot agreement | 93.8 % | |
 
Most of the board loss is missed detections of low-level clips; the acoustic path and the noise floor are the likely causes (the golden model is insensitive to ±3 dB level changes). Quick test with live human speech: 16/20.
 
## 5. License and attribution
- Speech Commands v2: Google, CC BY 4.0. The regression vectors in `snn/golden/vectors/` are integer features and PCM derived from clips in this dataset.
- The ZSK board reference design (Libertron) is not included and is required separately to build for the board.
- Code, RTL and documents in this repository: public release for competition submission.
 
