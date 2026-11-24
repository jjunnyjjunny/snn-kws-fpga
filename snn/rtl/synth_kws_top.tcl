# synth_kws_top.tcl — Vivado 2023.2 batch OOC synthesis + implementation + reports for kws_top
# (fe_top + vad + preroll_buf + snn_core + decision_seg) at 48 MHz, xc7z020clg484-1.
# Usage: vivado -mode batch -source synth_kws_top.tcl -tclargs <abs rom dir> [MULT_SERIAL=0] [ZERO_SKIP=1]
#   e.g. vivado -mode batch -source synth_kws_top.tcl -tclargs C:/kws_snn/snn/rom 0 1
set rom_dir [lindex $argv 0]
if {$rom_dir eq ""} { set rom_dir [file normalize [file join [file dirname [info script]] ../rom]] }
set ms [lindex $argv 1]; if {$ms eq ""} { set ms 0 }
set zs [lindex $argv 2]; if {$zs eq ""} { set zs 1 }
set rtl_dir [file dirname [file normalize [info script]]]
set rep_dir [file join $rtl_dir reports_kws_top]; file mkdir $rep_dir
set part xc7z020clg484-1
create_project -in_memory -part $part
set_property target_language Verilog [current_project]
read_verilog [glob $rtl_dir/*.v]
read_verilog [glob $rtl_dir/fe/*.v]
read_xdc $rtl_dir/snn_core_ooc.xdc
synth_design -top kws_top -part $part -mode out_of_context \
    -generic ROM_DIR=\"$rom_dir\" -generic MEM_PREFIX=\"$rom_dir/m3k3\" -generic MULT_SERIAL=$ms -generic ZERO_SKIP=$zs \
    -flatten_hierarchy rebuilt -directive AreaOptimized_high
report_utilization -hierarchical -file $rep_dir/synth_util_hier.rpt
opt_design; place_design; route_design
report_utilization -file $rep_dir/impl_util.rpt
report_utilization -hierarchical -file $rep_dir/impl_util_hier.rpt
report_timing_summary -file $rep_dir/impl_timing.rpt
report_power -file $rep_dir/impl_power_vectorless.rpt
report_ram_utilization -file $rep_dir/impl_ram.rpt
write_checkpoint -force $rep_dir/kws_top_routed.dcp
puts "DONE: reports in $rep_dir"
