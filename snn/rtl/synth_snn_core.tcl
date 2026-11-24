# synth_snn_core.tcl — Vivado 2023.2 batch OOC synthesis + implementation + reports for snn_core.
# Usage (Windows, Vivado Tcl shell or cmd):
#   vivado -mode batch -source synth_snn_core.tcl -tclargs <abs path of snn/rom>/m3k3 [ZERO_SKIP]
# e.g.  vivado -mode batch -source synth_snn_core.tcl -tclargs C:/kws_snn/snn/rom/m3k3 1
# Reports are written to ./reports/ (utilization, timing summary, vectorless power, per-hierarchy utilization).
set mem_prefix [lindex $argv 0]
if {$mem_prefix eq ""} { set mem_prefix [file normalize [file join [file dirname [info script]] ../rom/m3k3]] }
set zero_skip [lindex $argv 1]
if {$zero_skip eq ""} { set zero_skip 1 }
set rtl_dir [file dirname [file normalize [info script]]]
set rep_dir [file join $rtl_dir reports]
file mkdir $rep_dir
set part xc7z020clg484-1
create_project -in_memory -part $part
set_property target_language Verilog [current_project]
read_verilog [list \
    $rtl_dir/wrom.v $rtl_dir/prom.v $rtl_dir/spike_hist.v $rtl_dir/feat_hist.v $rtl_dir/row_acc.v \
    $rtl_dir/lif_update.v $rtl_dir/out_integrator.v $rtl_dir/decision.v $rtl_dir/snn_ctrl.v $rtl_dir/snn_core.v]
read_xdc $rtl_dir/snn_core_ooc.xdc
# $readmemh paths are resolved relative to the working directory at synthesis -> pass an absolute prefix
synth_design -top snn_core -part $part -mode out_of_context \
    -generic MEM_PREFIX=\"$mem_prefix\" -generic ZERO_SKIP=$zero_skip \
    -flatten_hierarchy rebuilt -directive AreaOptimized_high
report_utilization -hierarchical -file $rep_dir/synth_util_hier.rpt
report_utilization -file $rep_dir/synth_util.rpt
opt_design
place_design
route_design
report_utilization -file $rep_dir/impl_util.rpt
report_utilization -hierarchical -file $rep_dir/impl_util_hier.rpt
report_timing_summary -file $rep_dir/impl_timing.rpt
report_power -file $rep_dir/impl_power_vectorless.rpt
report_ram_utilization -file $rep_dir/impl_ram.rpt
write_checkpoint -force $rep_dir/snn_core_routed.dcp
puts "DONE: reports in $rep_dir"
