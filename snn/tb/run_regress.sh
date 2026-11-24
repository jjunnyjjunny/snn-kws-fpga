#!/bin/bash
# run_regress.sh [run=M3k3e60_c12_q8_kd] [prefix=m3k3] [N=240] — full regression after (re-)exporting the weights:
#   1. archive the previous .mem of <prefix> to snn/rom/archive/<date>/, export the run's weights,
#   2. regenerate the 12 golden clip vectors, the N-sample regression set and the streaming vector,
#   3. simulate: 12 clips (bit-exact, zero-skip on and off), streaming, regression (prediction + sums).
set -e
cd "$(dirname "$0")"
RUN=${1:-M3k3e60_c12_q8_kd}; P=${2:-m3k3}; N=${3:-240}
PY=${PY:-python}
ROOT=$(readlink -f ..); D12=$ROOT/data/feat_c12_m40_w480.npz
[[ $RUN == *c10* ]] && D12=$ROOT/data/feat_c10_m40_w480.npz
if [[ $RUN == *_U[0-9]*_* ]]; then U=$(echo $RUN | sed 's/.*_\(U[0-9][a-z]*\)_.*/\1/'); D12=$ROOT/data/feat_c12_m40_w480_$U.npz; fi   # week-4 unknown/silence variants
export CUDA_VISIBLE_DEVICES=${GPU:-6}
if [ "${SKIP_EXPORT:-0}" != 1 ]; then
  if ls $ROOT/rom/${P}_* >/dev/null 2>&1; then A=$ROOT/rom/archive/$(date +%Y%m%d_%H%M%S)_$P; mkdir -p $A; cp $ROOT/rom/${P}_* $A/; echo "archived old $P .mem -> $A"; fi
  (cd $ROOT/.. && $PY snn/golden/export_snn_mem.py --run snn/runs/$RUN --prefix $P > snn/logs/export_$P.log)
  (cd $ROOT/.. && $PY snn/golden/dump_vectors.py --run snn/runs/$RUN --prefix $P --data $D12 > snn/logs/vectors_$P.log)
  (cd $ROOT/.. && $PY snn/golden/dump_vectors.py --run snn/runs/$RUN --prefix $P --data $D12 --regress $N > snn/logs/regress_$P.log)
  (cd $ROOT/.. && $PY snn/golden/dump_vectors.py --run snn/runs/$RUN --prefix $P --data $D12 --stream yes_11,silence_13,no_3,unknown_2,stop_10,go_50 > snn/logs/stream_$P.log 2>&1 || \
   $PY snn/golden/dump_vectors.py --run snn/runs/$RUN --prefix $P --data $D12 --stream $(ls $ROOT/golden/vectors/$P | grep _ | head -6 | tr '\n' ',' | sed 's/,$//') > snn/logs/stream_$P.log)
  (cd $ROOT/.. && $PY snn/golden/snn_int_ref.py --run snn/runs/$RUN --data $D12 | grep -E "pred_agreement|vout_elementwise")
fi
export MEM_PREFIX=$ROOT/rom/$P
echo "== clip vectors (zero-skip on)";  ./run_all_vecs.sh $P 1 | sed "s|vec=$ROOT/golden/vectors/||" | awk '{print $2,$3,$6,$7,$8}'
echo "== clip vectors (zero-skip off)"; ./run_all_vecs.sh $P 0 | sed "s|vec=$ROOT/golden/vectors/||" | awk '{print $2,$3,$6,$7,$8}'
echo "== streaming";                    ./run_vec.sh ../golden/vectors/${P}_stream count | grep RESULT | sed "s|vec=$ROOT/golden/vectors/||" | awk '{print $2,$3,$6,$7,$8}'
echo "== regression ($N samples)"
VEC=$ROOT/golden/vectors/${P}_regress; B=build/${P}_regress; mkdir -p $B
NS=$(python3 -c "import json;print(json.load(open('$VEC/index.json'))['n'])"); NC=$(python3 -c "import json;print(json.load(open('$VEC/index.json'))['n_class'])")
cat > $B/tb_cfg.vh <<EOV
\`define VEC_DIR "$VEC"
\`define N_SAMPLES $NS
\`define T_FRAMES 98
\`define N_CLASS $NC
\`define ZERO_SKIP 1
\`define MEM_PREFIX "$MEM_PREFIX"
\`define VERBOSE 0
EOV
RTL=../rtl
iverilog -g2005 -I $B -o $B/sim.vvp tb_regress.v $RTL/wrom.v $RTL/prom.v $RTL/spike_hist.v $RTL/feat_hist.v $RTL/row_acc.v \
    $RTL/lif_update.v $RTL/out_integrator.v $RTL/decision.v $RTL/snn_ctrl.v $RTL/snn_core.v
vvp -n $B/sim.vvp | grep -E "REGRESS|MISMATCH"
