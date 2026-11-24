#!/bin/bash
# run_week4_B.sh — B-3 channel-robust retraining (4-B): U2 sampling (unknown 3x, class weights 10:2,11:3) + input-channel
# augmentation (make_dataset.py --channel_aug: random low-pass / high shelf, weak reverb, gain U(-30,+10) dB, noise floor
# U(-80,-62) dBFS) -> 48ch DS-CNN teacher -> M3k3 float 60 ep -> Q8 QAT + KD 15 ep, seeds 0 (GPU 6) and 1 (GPU 7).
# Then: numpy golden check, stream evaluation (talk/quiet/board, margins 0/2838/6212/9633), level sweep. Marker: B3_ALL_DONE.
cd "$(dirname "$0")/.."
PY=${PY:-python}
M3="--model M3 --K 3 --hidden 64 --bs 512"
tr(){ gpu=$1; name=$2; shift 2; [ -f snn/runs/$name/results.json ] && return; CUDA_VISIBLE_DEVICES=$gpu $PY snn/train_snn.py --name $name "$@" > snn/logs/$name.log 2>&1; }
ds(){ gpu=$1; name=$2; shift 2; [ -f snn/runs/$name/results.json ] && return; CUDA_VISIBLE_DEVICES=$gpu $PY snn/baseline/train_dscnn.py --name $name "$@" > snn/logs/$name.log 2>&1; }
U=U2c; D=snn/data/feat_c12_m40_w480_$U.npz
[ -f $D ] || $PY snn/data/make_dataset.py --classes 12 --aug_copies 2 --unknown_mult 3 --channel_aug --test_gains=-30,-20,-10,6,10 --tag _$U > snn/logs/make_$U.log 2>&1
[ -f $D ] || { echo "DATASET FAILED"; exit 1; }
echo DATASET_DONE $(date)
ds 6 dscnn_c12_ch48_$U --data $D --epochs 100 --ch 48
echo TEACHER_DONE $(date)
KD="--kd_logits snn/runs/dscnn_c12_ch48_$U/teacher_logits_train.npy --kd_alpha 1.0 --kd_tau 4"
CW="--class_weight 10:2,11:3"
P="--T_on 600 --T_off 360 --N_on 4 --H 30 --P 20 --L_min 25 --L_max 130"
seed_run(){ gpu=$1; s=$2
  tr $gpu M3k3_${U}_float_e60_s$s $M3 --data $D --mode float --epochs 60 --lr 3e-3 --seed $s $CW
  tr $gpu M3k3e60_${U}_q8_kd_s$s $M3 --data $D --mode int --bits 8 --init snn/runs/M3k3_${U}_float_e60_s$s/best.pt --epochs 15 --lr 5e-4 --seed $s $KD $CW
  r=M3k3e60_${U}_q8_kd_s$s
  [ -f snn/runs/$r/golden_check.json ] || CUDA_VISIBLE_DEVICES=$gpu $PY snn/golden/snn_int_ref.py --run snn/runs/$r --data $D > snn/logs/golden_$r.log 2>&1
  [ -f snn/runs/vad_eval/eval_${r}_$U.json ] || CUDA_VISIBLE_DEVICES="" $PY snn/eval/vad_eval.py --conds talk,quiet,board --n_val 100 --n_test 200 --workers 12 --eval_only --margins 0,2838,6212,9633 --run snn/runs/$r --tag $U > snn/logs/vad_eval_$r.log 2>&1
  CUDA_VISIBLE_DEVICES="" $PY snn/eval/level_sweep.py --run snn/runs/$r --data $D --out snn/runs/board_capture_${U}_s$s --fig snn/runs/board_capture_${U}_s$s > snn/logs/level_sweep_$r.log 2>&1
  echo SEED_${s}_DONE $(date)
}
( seed_run 6 0 ) &
( seed_run 7 1 ) &
wait
echo B3_ALL_DONE $(date)
