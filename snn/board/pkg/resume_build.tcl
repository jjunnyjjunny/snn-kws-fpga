# resume_build.tcl — continue an interrupted build_bitstream.tcl run (e.g. the Tcl shell was closed) without recreating
# the project. Re-runs only the runs that did not finish, then writes the same reports / .bit / .ltx / .xsa.
#   (Vivado Tcl Shell)  cd C:/kws/snn/board/pkg ; source resume_build.tcl
# Before running: in Task Manager end any leftover vivado.exe from the interrupted build (keep only this shell's).
set pkg  [file dirname [file normalize [info script]]]
set proj $pkg/vivado_kws
if {![file exists $proj/ZSK_KWS.xpr]} { error "no project at $proj — run build_bitstream.tcl from scratch" }
if {[current_project -quiet] ne ""} { close_project }
open_project $proj/ZSK_KWS.xpr
update_compile_order -fileset sources_1
proc done {r} { expr {[get_property PROGRESS [get_runs $r]] eq "100%" && ![get_property NEEDS_REFRESH [get_runs $r]]} }
puts "synth_1: [get_property STATUS [get_runs synth_1]] ([get_property PROGRESS [get_runs synth_1]])"
puts "impl_1 : [get_property STATUS [get_runs impl_1]] ([get_property PROGRESS [get_runs impl_1]])"
if {![done synth_1]} {
    reset_run synth_1
    launch_runs synth_1 -jobs 4; wait_on_run synth_1
}
if {![done impl_1] || ![file exists $proj/ZSK_KWS.runs/impl_1/ZSK_Design_Top.bit]} {
    reset_run impl_1
    launch_runs impl_1 -to_step write_bitstream -jobs 4; wait_on_run impl_1
}
open_run impl_1
file mkdir $pkg/reports
report_utilization -file $pkg/reports/impl_util.rpt
report_utilization -hierarchical -file $pkg/reports/impl_util_hier.rpt
report_timing_summary -file $pkg/reports/impl_timing.rpt
report_power -file $pkg/reports/impl_power_vectorless.rpt
file copy -force $proj/ZSK_KWS.runs/impl_1/ZSK_Design_Top.bit $pkg/ZSK_Design_Top.bit
file copy -force $proj/ZSK_KWS.runs/impl_1/ZSK_Design_Top.ltx $pkg/ZSK_Design_Top.ltx
write_hw_platform -fixed -include_bit -force -file $pkg/ZSK_Design_Top.xsa
puts "DONE: $pkg/ZSK_Design_Top.bit .ltx .xsa, reports in $pkg/reports"
