#!/bin/bash
# run_week4_A.sh — A-2: U1/U2/U3 datasets (train re-sampled, val/test unchanged) -> 48ch teacher -> M3k3 float 60ep -> Q8+KD (seeds 0,1)
cd "$(dirname "$0")/.."
PY=${PY:-python}
M3="--model M3 --K 3 --hidden 64 --bs 512"
tr(){ gpu=$1; name=$2; shift 2; [ -f snn/runs/$name/results.json ] && return; CUDA_VISIBLE_DEVICES=$gpu $PY snn/train_snn.py --name $name "$@" > snn/logs/$name.log 2>&1; }
ds(){ gpu=$1; name=$2; shift 2; [ -f snn/runs/$name/results.json ] && return; CUDA_VISIBLE_DEVICES=$gpu $PY snn/baseline/train_dscnn.py --name $name "$@" > snn/logs/$name.log 2>&1; }
# datasets (sequential, CPU)
[ -f snn/data/feat_c12_m40_w480_U1.npz ] || $PY snn/data/make_dataset.py --classes 12 --aug_copies 2 --unknown_mult 2 --tag _U1 > snn/logs/make_U1.log 2>&1
[ -f snn/data/feat_c12_m40_w480_U2.npz ] || $PY snn/data/make_dataset.py --classes 12 --aug_copies 2 --unknown_mult 3 --tag _U2 > snn/logs/make_U2.log 2>&1
[ -f snn/data/feat_c12_m40_w480_U3.npz ] || $PY snn/data/make_dataset.py --classes 12 --aug_copies 2 --unknown_mult 3 --silence_mult 2 --silence_lowlevel --tag _U3 > snn/logs/make_U3.log 2>&1
echo DATASETS_DONE
run_variant(){ gpu=$1; U=$2; CW=$3
  D=snn/data/feat_c12_m40_w480_$U.npz
  ds $gpu dscnn_c12_ch48_$U --data $D --epochs 100 --ch 48
  KD="--kd_logits snn/runs/dscnn_c12_ch48_$U/teacher_logits_train.npy --kd_alpha 1.0 --kd_tau 4"
  for s in 0 1; do
    tr $gpu M3k3_${U}_float_e60_s$s $M3 --data $D --mode float --epochs 60 --lr 3e-3 --seed $s $CW
    tr $gpu M3k3e60_${U}_q8_kd_s$s $M3 --data $D --mode int --bits 8 --init snn/runs/M3k3_${U}_float_e60_s$s/best.pt --epochs 15 --lr 5e-4 --seed $s $KD $CW
  done
}
( run_variant 6 U1 "" ) &
( run_variant 7 U2 "--class_weight 10:2,11:3" ) &
( run_variant 7 U3 "--class_weight 10:2,11:3" ) &
wait
echo A2_ALL_DONE
