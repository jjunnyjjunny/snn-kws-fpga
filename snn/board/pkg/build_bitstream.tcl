# build_bitstream.tcl — Windows Vivado 2023.2 batch build of the KWS board design (reference block design +
# reference I2S modules + KWS user_logic). Produces bitstream, XSA (with bitstream), reports.
#   (cmd)             cd /d C:\kws_snn\snn\board\pkg
#                     vivado -mode batch -source build_bitstream.tcl -tclargs C:/kws_snn/snn
#   (Vivado Tcl Shell) cd C:/kws_snn/snn/board/pkg ; set argv [list C:/kws_snn/snn <variant>] ; source build_bitstream.tcl
#   Tcl treats backslash as escape: use forward slashes inside the Tcl shell.
# (argument = absolute path of the snn directory, forward slashes; project is created in ./vivado_kws)
set snn [lindex $argv 0]
# variant (argv 1): "" = plain build ; names containing "popt" enable power_opt_design ; names containing "nogate" build
# without BUFGCE clock gating (NO_CLK_GATE). Each variant gets its own project dir / bitstream / report dir:
#   vivado_kws[_<variant>], ZSK_Design_Top[_<variant>].bit/.ltx/.xsa, reports[_<variant>]/
set variant [lindex $argv 1]
set sfx [expr {$variant eq "" ? "" : "_$variant"}]
set use_popt [string match *popt* $variant]
set no_gate  [string match *nogate* $variant]
if {$snn eq ""} { set snn [file normalize [file join [file dirname [info script]] ../..]] }
set pkg  [file dirname [file normalize [info script]]]
set ref  $snn/board/ref
set rtl  $snn/rtl
set rom  $snn/rom
set proj $pkg/vivado_kws$sfx
set part xc7z020clg484-1
file delete -force $proj
create_project ZSK_KWS $proj -part $part
set_property target_language Verilog [current_project]
# ---- block design (reference, unchanged)
source $ref/Block_Design.tcl
regenerate_bd_layout
validate_bd_design
save_bd_design
make_wrapper -files [get_files Block_Design.bd] -top
add_files -norecurse [glob $proj/ZSK_KWS.gen/sources_1/bd/Block_Design/hdl/Block_Design_wrapper.v]
# ---- RTL: reference I2S/codec modules (unchanged) + modified top + KWS
add_files -norecurse [list $ref/adau1761_interface.v $ref/i2s_clock_interface.v $ref/i2s_data_interface.v]
add_files -norecurse [list $pkg/ZSK_Design_Top.v $pkg/user_logic.v]
add_files -norecurse [glob $rtl/*.v]
add_files -norecurse [glob $rtl/fe/*.v]
# ---- ROM images: generate rom_paths.vh with the absolute path and add the .mem files as sources
set fh [open $pkg/rom_paths.vh w]
puts $fh "`define ROM_DIR_DEF \"$rom\""
close $fh
add_files -norecurse $pkg/rom_paths.vh
set_property file_type {Verilog Header} [get_files $pkg/rom_paths.vh]
add_files -norecurse [glob $rom/*.mem]
set_property is_global_include true [get_files $pkg/rom_paths.vh]
# ---- ILA IP (instantiated in user_logic.v under `ifdef SYNTHESIS; widths must match the dbg_* registers)
set pw [list 2 1 18 1 1 4 1 1 24 24 16 4 1 16 1 8 4 1 16 1 1 1 1]   ;# probe21 fe_clk_ce, probe22 snn_clk_ce
create_ip -name ila -vendor xilinx.com -library ip -version 6.2 -module_name ila_kws
set cfg [list CONFIG.C_NUM_OF_PROBES [llength $pw] CONFIG.C_DATA_DEPTH {16384} CONFIG.C_EN_STRG_QUAL {1} CONFIG.ALL_PROBE_SAME_MU_CNT {2} \
         CONFIG.C_ADV_TRIGGER {false} CONFIG.C_INPUT_PIPE_STAGES {0} CONFIG.C_TRIGIN_EN {false} CONFIG.C_TRIGOUT_EN {false}]
for {set i 0} {$i < [llength $pw]} {incr i} { lappend cfg CONFIG.C_PROBE${i}_WIDTH [lindex $pw $i] }
set_property -dict $cfg [get_ips ila_kws]
generate_target all [get_ips ila_kws]
# ---- constraints
add_files -fileset constrs_1 -norecurse [list $ref/constraints.xdc $pkg/ila_kws.xdc]
set_property top ZSK_Design_Top [current_fileset]
if {$no_gate} { set_property verilog_define {NO_CLK_GATE=1} [current_fileset] }
if {$use_popt} { set_property STEPS.POWER_OPT_DESIGN.IS_ENABLED true [get_runs impl_1] }
update_compile_order -fileset sources_1
# ---- run
launch_runs synth_1 -jobs 4; wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 4; wait_on_run impl_1
open_run impl_1
set rep $pkg/reports$sfx
file mkdir $rep
report_utilization -file $rep/impl_util.rpt
report_utilization -hierarchical -file $rep/impl_util_hier.rpt
report_timing_summary -file $rep/impl_timing.rpt
report_clock_interaction -file $rep/impl_clock_interaction.rpt      ;# gated clocks (BUFGCE) must appear as the same clock, no unsafe pairs
report_cdc -file $rep/impl_cdc.rpt
report_clock_utilization -file $rep/impl_clock_util.rpt             ;# BUFG count
report_power -file $rep/impl_power_vectorless.rpt
file copy -force $proj/ZSK_KWS.runs/impl_1/ZSK_Design_Top.bit $pkg/ZSK_Design_Top$sfx.bit
file copy -force $proj/ZSK_KWS.runs/impl_1/ZSK_Design_Top.ltx $pkg/ZSK_Design_Top$sfx.ltx
write_hw_platform -fixed -include_bit -force -file $pkg/ZSK_Design_Top$sfx.xsa
puts "DONE: $pkg/ZSK_Design_Top$sfx.bit .ltx .xsa, reports in $rep"
