#!/bin/bash
# run_fe_top.sh <vector dir> [MULT_SERIAL=0] [SAMPLE_CYC=64] — fe_top sample-level test (3000 = real 16 kHz timing)
set -e; cd "$(dirname "$0")"
VEC=$(readlink -f "$1"); MS=${2:-0}; SC=${3:-64}; ROM=$(readlink -f ../../rom); RTL=../../rtl/fe
B=build/top_$(basename $VEC)_ms${MS}_sc$SC; mkdir -p $B
printf '`define VEC_DIR "%s"\n`define ROM_DIR "%s"\n`define MULT_SERIAL %s\n`define T_FRAMES 98\n`define SAMPLE_CYC %s\n`define T0 %s\n' $VEC $ROM $MS $SC ${T0:-6694} > $B/tb_cfg.vh
iverilog -g2005 ${GATE_OFF:+-DGATE_OFF=1} -I $B -o $B/sim.vvp tb_fe_top.v $RTL/fe_in_scale.v $RTL/fe_ring.v $RTL/fe_buf.v $RTL/fe_window.v $RTL/fe_fft512.v $RTL/fe_power.v $RTL/fe_mel_log.v $RTL/fe_top.v $RTL/../clk_gate_buf.v
vvp -n $B/sim.vvp | grep -v "^VCD"
