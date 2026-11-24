#!/bin/bash
# run_fe_blocks.sh <vector dir> [MULT_SERIAL=0] [stop|count] — block-level front-end golden test (Icarus)
set -e; cd "$(dirname "$0")"
VEC=$(readlink -f "$1"); MS=${2:-0}; MODE=${3:-count}; ROM=$(readlink -f ../../rom); RTL=../../rtl/fe
B=build/$(basename $VEC)_ms$MS; mkdir -p $B
printf '`define VEC_DIR "%s"\n`define ROM_DIR "%s"\n`define MULT_SERIAL %s\n`define T_FRAMES 98\n`define STOP_ON_MISMATCH %s\n' $VEC $ROM $MS $([ "$MODE" == stop ] && echo 1 || echo 0) > $B/tb_cfg.vh
iverilog -g2005 -I $B -o $B/sim.vvp tb_fe_blocks.v $RTL/fe_buf.v $RTL/fe_window.v $RTL/fe_fft512.v $RTL/fe_power.v $RTL/fe_mel_log.v
vvp -n $B/sim.vvp | grep -v "^VCD"
