#!/bin/bash
# run_week2_B.sh — week-2 GPU jobs (B-1 10-class retrain, B-2 seeds, B-3 long training), launched in
# parallel with float -> QAT dependency chains. Logs: snn/logs/<name>.log, results: snn/runs/<name>/.
# All model selection inside a run is by validation accuracy (best.pt = best val).
cd "$(dirname "$0")/.."
PY=${PY:-python}
D12=snn/data/feat_c12_m40_w480.npz; D10=snn/data/feat_c10_m40_w480.npz
KD12="--kd_logits snn/runs/dscnn_c12_ch48/teacher_logits_train.npy --kd_alpha 1.0 --kd_tau 4"
KD10="--kd_logits snn/runs/dscnn_c10_ch48/teacher_logits_train.npy --kd_alpha 1.0 --kd_tau 4"
M3="--model M3 --K 3 --hidden 64 --bs 512"
tr(){ gpu=$1; name=$2; shift 2; [ -f snn/runs/$name/results.json ] && return; CUDA_VISIBLE_DEVICES=$gpu $PY snn/train_snn.py --name $name "$@" > snn/logs/$name.log 2>&1; }
ds(){ gpu=$1; name=$2; shift 2; [ -f snn/runs/$name/results.json ] && return; CUDA_VISIBLE_DEVICES=$gpu $PY snn/baseline/train_dscnn.py --name $name "$@" > snn/logs/$name.log 2>&1; }
waitfor(){ until [ -f "$1" ]; do sleep 60; done; }

# ---- B-1 / B-2: 10-class teacher + DS-CNN 24ch seeds (GPU 7, fast) ---------------------------
( ds 7 dscnn_c10_ch48 --data $D10 --epochs 100 --ch 48
  ds 7 dscnn_c10_s1 --data $D10 --epochs 100 --seed 1; ds 7 dscnn_c10_s2 --data $D10 --epochs 100 --seed 2
  ds 7 dscnn_c12_s1 --data $D12 --epochs 100 --seed 1; ds 7 dscnn_c12_s2 --data $D12 --epochs 100 --seed 2 ) &

# ---- B-1 + B-2: M3k3 10-class, 60 epochs, seeds 0/1/2 -> Q8 (s0) and Q8+KD (all) (GPU 7) --------
for s in 0 1 2; do
( tr 7 M3k3_c10_float_e60_s$s $M3 --data $D10 --mode float --epochs 60 --lr 3e-3 --seed $s
  INIT=snn/runs/M3k3_c10_float_e60_s$s/best.pt
  [ $s == 0 ] && tr 7 M3k3e60_c10_q8_s0 $M3 --data $D10 --mode int --bits 8 --init $INIT --epochs 15 --lr 5e-4 --seed 0
  waitfor snn/runs/dscnn_c10_ch48/teacher_logits_train.npy
  tr 7 M3k3e60_c10_q8_kd_s$s $M3 --data $D10 --mode int --bits 8 --init $INIT --epochs 15 --lr 5e-4 --seed $s $KD10 ) &
done

# ---- B-2: M3k3 12-class seeds 1/2 (seed 0 = M3k3_c12_float_e60 / M3k3e60_c12_q8_kd from week 1) (GPU 6)
for s in 1 2; do
( tr 6 M3k3_c12_float_e60_s$s $M3 --data $D12 --mode float --epochs 60 --lr 3e-3 --seed $s
  tr 6 M3k3e60_c12_q8_kd_s$s $M3 --data $D12 --mode int --bits 8 --init snn/runs/M3k3_c12_float_e60_s$s/best.pt --epochs 15 --lr 5e-4 --seed $s $KD12 ) &
done

# ---- B-3: long training 100 / 150 epochs -> Q8+KD (20 epochs) (GPU 6) ------------------------------
for e in 100 150; do
( tr 6 M3k3_c12_float_e$e $M3 --data $D12 --mode float --epochs $e --lr 3e-3 --seed 0
  tr 6 M3k3e${e}_c12_q8_kd $M3 --data $D12 --mode int --bits 8 --init snn/runs/M3k3_c12_float_e$e/best.pt --epochs 20 --lr 5e-4 --seed 0 $KD12 ) &
done
wait
echo B_ALL_DONE
