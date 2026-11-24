#!/bin/bash
# run_activity.sh <vector dir> [T0=6694] [VAD_ON=1] [SAMPLE_CYC=48] — activity counters on a scenario stream.
# VAD_ON=0 emulates "SNN always on": T_ON=0 (every frame triggers), L_MAX=255 (reset every 255 frames).
set -e; cd "$(dirname "$0")"
VEC=$(readlink -f "$1"); T0=${2:-6694}; VAD=${3:-1}; SC=${4:-48}; ROM=$(readlink -f ../../rom); RTL=../../rtl
read NS T TON TOFF NON H P LMIN LMAX < <(python3 -c "
import json; i=json.load(open('$VEC/info.json')); p=i['params']; print(i['n_samples'], i['T'], p['T_on'], p['T_off'], p['N_on'], p['H'], p['P'], p['L_min'], p['L_max'])")
if [ "$VAD" == 0 ]; then TON=0; TOFF=0; NON=1; LMAX=255; fi
B=build/$(basename $VEC)_t0${T0}_vad$VAD; mkdir -p $B
printf '`define VEC_DIR "%s"\n`define ROM_DIR "%s"\n`define MEM_PREFIX "%s/m3k3"\n`define N_SAMPLES %s\n`define T_FRAMES %s\n`define T_ON %s\n`define T_OFF %s\n`define N_ON %s\n`define H %s\n`define P %s\n`define L_MIN %s\n`define L_MAX %s\n`define T0 %s\n`define VAD_ON %s\n`define SAMPLE_CYC %s\n' \
  $VEC $ROM $ROM $NS $T $TON $TOFF $NON $H $P $LMIN $LMAX $T0 $VAD $SC > $B/tb_cfg.vh
iverilog -g2005 -DACTIVITY ${GATE_OFF:+-DGATE_OFF=1} -I $B -o $B/sim.vvp tb_activity.v $RTL/kws_top.v $RTL/clk_gate_buf.v $RTL/vad.v $RTL/preroll_buf.v $RTL/decision_seg.v $RTL/snn_core.v $RTL/snn_ctrl.v \
  $RTL/row_acc.v $RTL/lif_update.v $RTL/out_integrator.v $RTL/decision.v $RTL/spike_hist.v $RTL/feat_hist.v $RTL/wrom.v $RTL/prom.v $RTL/fe/*.v
vvp -n $B/sim.vvp | grep -E "^ACT"
