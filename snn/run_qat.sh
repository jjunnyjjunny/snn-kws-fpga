#!/bin/bash
# run_qat.sh <GPU> <model M1|M2|M3> <classes 10|12> [extra args]  — QAT 4/6/8-bit fine-tune from the float run
# (+ KD variants with the 48-ch DS-CNN teacher on 12 classes). Logs: snn/logs/<name>.log
cd "$(dirname "$0")/.."
PY=${PY:-python}
G=$1; M=$2; CL=$3; shift 3
D=snn/data/feat_c${CL}_m40_w480.npz
HID=128; [[ $M == M3 ]] && HID=64
INIT=snn/runs/${M}_c${CL}_float/best.pt
run(){ name=$1; shift; CUDA_VISIBLE_DEVICES=$G nohup $PY snn/train_snn.py --name $name "$@" > snn/logs/$name.log 2>&1 & }
for B in 8 6 4; do
  run ${M}_c${CL}_q${B} --model $M --data $D --mode int --bits $B --hidden $HID --init $INIT --epochs 15 --lr 5e-4 "$@"
done
if [[ $CL == 12 ]]; then
  for B in 8 4; do
    run ${M}_c${CL}_q${B}_kd --model $M --data $D --mode int --bits $B --hidden $HID --init $INIT --epochs 15 --lr 5e-4 \
        --kd_logits snn/runs/dscnn_c12_ch48/teacher_logits_train.npy --kd_alpha 1.0 --kd_tau 4 "$@"
  done
fi
