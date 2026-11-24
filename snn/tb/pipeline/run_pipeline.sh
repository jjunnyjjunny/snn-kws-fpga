#!/bin/bash
# run_pipeline.sh <vector dir> [SAMPLE_CYC=48] — integration test of kws_top against pipeline_int_ref.py (params from info.json)
set -e; cd "$(dirname "$0")"
VEC=$(readlink -f "$1"); SC=${2:-48}; ROM=$(readlink -f ../../rom); RTL=../../rtl
read NS T ND TON TOFF NON H P LMIN LMAX EN MG < <(python3 -c "
import json; i=json.load(open('$VEC/info.json')); p=i['params']
print(i['n_samples'], i['T'], len(i['dets']), p['T_on'], p['T_off'], p['N_on'], p['H'], p['P'], p['L_min'], p['L_max'], 1 if i['energy']=='E1' else 2, i.get('margin_sel', 0))")
B=build/$(basename $VEC); mkdir -p $B
printf '`define VEC_DIR "%s"\n`define ROM_DIR "%s"\n`define MEM_PREFIX "%s/m3k3"\n`define N_SAMPLES %s\n`define T_FRAMES %s\n`define N_DETS %s\n`define T_ON %s\n`define T_OFF %s\n`define N_ON %s\n`define H %s\n`define P %s\n`define L_MIN %s\n`define L_MAX %s\n`define ENERGY_SEL %s\n`define MARGIN_SEL %s\n`define SAMPLE_CYC %s\n' \
  $VEC $ROM $ROM $NS $T $ND $TON $TOFF $NON $H $P $LMIN $LMAX $EN $MG $SC > $B/tb_cfg.vh
iverilog -g2005 ${GATE_OFF:+-DGATE_OFF=1} -I $B -o $B/sim.vvp tb_pipeline.v $RTL/kws_top.v $RTL/clk_gate_buf.v $RTL/vad.v $RTL/preroll_buf.v $RTL/decision_seg.v $RTL/snn_core.v $RTL/snn_ctrl.v \
  $RTL/row_acc.v $RTL/lif_update.v $RTL/out_integrator.v $RTL/decision.v $RTL/spike_hist.v $RTL/feat_hist.v $RTL/wrom.v $RTL/prom.v $RTL/fe/*.v
vvp -n $B/sim.vvp | grep -E "PIPELINE|MISMATCH"
