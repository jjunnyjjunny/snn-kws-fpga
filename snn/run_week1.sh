#!/bin/bash
# run_week1.sh — reproduction of all week-1 steps (env: conda `snn`). Steps are idempotent-ish;
# long trainings are launched with nohup, logs in snn/logs/<name>.log, results in snn/runs/<name>/.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
G=${GPU:-6}
run(){ name=$1; shift; CUDA_VISIBLE_DEVICES=$G nohup $PY "$@" --name $name > snn/logs/$name.log 2>&1 & }

step=${1:-all}
if [[ $step == all || $step == 1 ]]; then       # front-end golden model, ROMs, verification
  $PY snn/frontend/test_fft.py
  $PY snn/frontend/decimator.py
  $PY snn/frontend/export_roms.py
  $PY snn/frontend/verify_frontend.py --n 100
fi
if [[ $step == all || $step == 2 ]]; then       # datasets (each ~3-4 min with 32 workers)
  $PY snn/data/make_dataset.py --classes 12 --n_mels 40 --win 480 --aug_copies 2
  $PY snn/data/make_dataset.py --classes 10 --n_mels 40 --win 480 --aug_copies 2
  $PY snn/data/make_dataset.py --classes 12 --n_mels 40 --win 480 --aug_copies 2 --no_gain_aug
  $PY snn/data/make_dataset.py --classes 12 --n_mels 32 --win 480 --aug_copies 2
  $PY snn/data/make_dataset.py --classes 12 --n_mels 40 --win 400 --aug_copies 2
fi
D12=snn/data/feat_c12_m40_w480.npz; D10=snn/data/feat_c10_m40_w480.npz
if [[ $step == all || $step == 3 ]]; then       # DS-CNN baselines (PyTorch)
  run dscnn_c12_e100 snn/baseline/train_dscnn.py --data $D12 --epochs 100
  run dscnn_c12_ch48 snn/baseline/train_dscnn.py --data $D12 --epochs 100 --ch 48
  run dscnn_c10_e100 snn/baseline/train_dscnn.py --data $D10 --epochs 100
  run dscnn_c12_nogain snn/baseline/train_dscnn.py --data snn/data/feat_c12_m40_w480_nogain.npz --epochs 100
  run dscnn_c12_m32 snn/baseline/train_dscnn.py --data snn/data/feat_c12_m32_w480.npz --epochs 100
  run dscnn_c12_w400 snn/baseline/train_dscnn.py --data snn/data/feat_c12_m40_w400.npz --epochs 100
fi
if [[ $step == all || $step == 4 ]]; then       # SNN float
  run M1_c12_float snn/train_snn.py --model M1 --data $D12 --mode float --epochs 30
  run M2_c12_float snn/train_snn.py --model M2 --data $D12 --mode float --epochs 30
  run M3_c12_float snn/train_snn.py --model M3 --data $D12 --mode float --epochs 30 --hidden 64
  run M1_c12_float_k2 snn/train_snn.py --model M1 --data $D12 --mode float --epochs 30 --k 2
  run M1_c12_float_k4 snn/train_snn.py --model M1 --data $D12 --mode float --epochs 30 --k 4
  run M1_c10_float snn/train_snn.py --model M1 --data $D10 --mode float --epochs 30
  run M2_c10_float snn/train_snn.py --model M2 --data $D10 --mode float --epochs 30
  run M3_c10_float snn/train_snn.py --model M3 --data $D10 --mode float --epochs 30 --hidden 64
fi
# step 5 (QAT / KD) and 6 (golden, export, vectors): see the commands appended below once step 4 finished
if [[ $step == 5 ]]; then                        # QAT / KD for the candidates that finished float training
  bash snn/run_qat.sh 6 M1 12; bash snn/run_qat.sh 7 M1 10
  bash snn/run_qat.sh 6 M3 12; bash snn/run_qat.sh 7 M3 10
  # M3k3 (K=3) variants, see snn/logs/*.log for the exact commands used (bs 512, lr 3e-3 float / 5e-4 QAT)
fi
if [[ $step == 6 ]]; then                        # golden check, .mem export, test vectors (recommended + 2nd + M1)
  for r in M3k3_c12_q8_kd M3k3_c12_q6 M3k3_c12_q4 M3_c12_q8_kd M1_c12_q8; do
    CUDA_VISIBLE_DEVICES=$G $PY snn/golden/snn_int_ref.py --run snn/runs/$r --data $D12
  done
  $PY snn/frontend/export_roms.py
  for pr in "m3k3 M3k3_c12_q8_kd" "m3 M3_c12_q8_kd" "m1 M1_c12_q8"; do set -- $pr
    CUDA_VISIBLE_DEVICES=$G $PY snn/golden/export_snn_mem.py --run snn/runs/$2 --prefix $1
    CUDA_VISIBLE_DEVICES=$G $PY snn/golden/dump_vectors.py --run snn/runs/$2 --prefix $1 --data $D12
  done
  $PY snn/collect_results.py
fi
