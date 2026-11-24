#!/bin/bash
# run_vec.sh <vector dir> [stop|count] [ZERO_SKIP=1] [VERBOSE=0] — simulate one golden vector with Icarus.
# Prints one RESULT line (PASS/FAIL, mismatch count, measured cycles per frame and per block).
set -e
cd "$(dirname "$0")"
VEC=$(readlink -f "$1"); MODE=${2:-count}; ZS=${3:-1}; VERB=${4:-0}
RTL=../rtl; ROM=$(readlink -f ../rom); PREFIX=${MEM_PREFIX:-$ROM/m3k3}
EXP=$(python3 -c "import json;print(json.load(open('$VEC/info.json'))['pred_sum'])")
NC=$(python3 -c "import json;print(len(json.load(open('$VEC/info.json'))['vout_sum']))")
TF=$(python3 -c "import json;print(json.load(open('$VEC/info.json')).get('T',98))")
B=build/$(basename $VEC)_zs$ZS; mkdir -p $B
cat > $B/tb_cfg.vh <<EOV
\`define VEC_DIR "$VEC"
\`define EXP_PRED $EXP
\`define STOP_ON_MISMATCH $([ "$MODE" == stop ] && echo 1 || echo 0)
\`define T_FRAMES $TF
\`define N_CLASS $NC
\`define ZERO_SKIP $ZS
\`define MEM_PREFIX "$PREFIX"
\`define CLIP_START 1
\`define VERBOSE $VERB
EOV
iverilog -g2005 -I $B -o $B/sim.vvp tb_snn_core.v $RTL/wrom.v $RTL/prom.v $RTL/spike_hist.v $RTL/feat_hist.v $RTL/row_acc.v \
    $RTL/lif_update.v $RTL/out_integrator.v $RTL/decision.v $RTL/snn_ctrl.v $RTL/snn_core.v
vvp -n $B/sim.vvp | grep -v "^VCD"
