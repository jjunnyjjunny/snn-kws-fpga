#!/bin/bash
# finalize_week2.sh — wait for the B-driver to finish, golden-check the new QAT runs, rebuild tables/report.
cd "$(dirname "$0")/../.."
PY=${PY:-python}
until grep -q B_ALL_DONE snn/logs/week2_B_driver.log 2>/dev/null; do sleep 300; done
for r in M3k3e60_c10_q8_s0 M3k3e60_c10_q8_kd_s0 M3k3e60_c10_q8_kd_s1 M3k3e60_c10_q8_kd_s2 M3k3e60_c12_q8_kd_s1 M3k3e60_c12_q8_kd_s2 M3k3e100_c12_q8_kd M3k3e150_c12_q8_kd; do
  D=snn/data/feat_c12_m40_w480.npz; [[ $r == *c10* ]] && D=snn/data/feat_c10_m40_w480.npz
  [ -f snn/runs/$r/results.json ] && [ ! -f snn/runs/$r/golden_check.json ] && CUDA_VISIBLE_DEVICES=6 $PY snn/golden/snn_int_ref.py --run snn/runs/$r --data $D > snn/logs/golden_$r.log 2>&1
done
$PY snn/collect_results.py > /dev/null; $PY snn/eval/build_report2.py
grep -h pred_agreement snn/runs/M3k3e*_q8*/golden_check.json | sort | uniq -c
echo FINALIZE_DONE
