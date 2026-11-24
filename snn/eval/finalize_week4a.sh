#!/bin/bash
# finalize_week4a.sh — after the U1~U3 driver: numpy golden check of each Q8+KD run, stream evaluation on val/test
# talk/quiet/board with the 4 operating points, then rebuild tables/report. Survives session restarts (nohup).
cd "$(dirname "$0")/../.."
PY=${PY:-python}
until grep -q A2_ALL_DONE snn/logs/week4_A_driver.log 2>/dev/null; do sleep 300; done
for U in U1 U2 U3; do for s in 0 1; do r=M3k3e60_${U}_q8_kd_s$s
  [ -f snn/runs/$r/results.json ] || continue
  [ -f snn/runs/$r/golden_check.json ] || CUDA_VISIBLE_DEVICES=6 $PY snn/golden/snn_int_ref.py --run snn/runs/$r --data snn/data/feat_c12_m40_w480_$U.npz > snn/logs/golden_$r.log 2>&1
  [ -f snn/runs/vad_eval/eval_${r}_$U.json ] || CUDA_VISIBLE_DEVICES="" $PY snn/eval/vad_eval.py --conds talk,quiet,board --n_val 100 --n_test 200 --workers 12 --eval_only --margins 0,2838,6212,9633 --run snn/runs/$r --tag $U > snn/logs/vad_eval_$r.log 2>&1
done; done
$PY snn/eval/build_report4a.py; $PY snn/docs/make_figures.py > /dev/null 2>&1
echo FINALIZE4A_DONE
