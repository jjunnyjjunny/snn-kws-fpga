# hls4ml_power.tcl — vectorless power / utilization / timing of the hls4ml DS-CNN reference design (intermediate stage:
# hls4ml IP + AXI DMA + AXI GPIO x4 + AXI Interconnect + kws_out_latch on the ZSK audio reference), from its routed checkpoint,
# with EXACTLY the report settings used for our design in power_flow.tcl (vectorless part):
#   - report_power -hier all / report_power (summary), no set_operating_conditions, no set_switching_activity,
#     reset_switching_activity -all before reporting  => Vivado defaults: toggle rate 12.5 %, static probability 0.5,
#     ambient 25 C, airflow 250 LFM, no heat sink, process typical, grade commercial (the report's "Settings" section
#     prints these; compare them with reports_power/vectorless_summary.rpt of our design — they must be identical).
#   - same tool version (2023.2) and part (xc7z020clg484-1): checked and printed below.
# Usage (Vivado Tcl Shell):
#   set argv [list C:/kws/hls4ml_ref/ZSK_Design_Top_routed.dcp C:/kws/hls4ml_ref/reports] ; source C:/kws/snn/board/pkg/hls4ml_power.tcl
set dcp [lindex $argv 0]
set out [lindex $argv 1]; if {$out eq ""} { set out [file dirname $dcp]/reports_power_ref }
if {![file exists $dcp]} { error "checkpoint not found: $dcp" }
file mkdir $out
if {[current_project -quiet] ne ""} { close_project }
open_checkpoint $dcp
puts "part: [get_property PART [current_design]]  (ours: xc7z020clg484-1)"
puts "tool: [version -short]  (ours: 2023.2)"
# ---- clock constraints: our design = clk_out1_Block_Design_clk_wiz_0_0, 20.833 ns (48 MHz) from the clk_wiz MMCM fed by PS FCLK0
set fh [open $out/clocks.txt w]
puts "clock constraints in this checkpoint:"
foreach c [get_clocks] {
    set line [format "  %-50s period %8.3f ns  source %s" [get_property NAME $c] [get_property PERIOD $c] [get_property SOURCE_PINS $c]]
    puts $line; puts $fh $line
}
close $fh
set c48 [get_clocks -quiet -filter {PERIOD > 20.8 && PERIOD < 20.9}]
if {$c48 eq ""} { puts "WARNING: no 20.833 ns (48 MHz) clock in this checkpoint — the KWS logic runs on a different clock than our design; note it in the table" } else { puts "48 MHz clock present: $c48" }
# ---- identical vectorless power settings
reset_switching_activity -all
report_power -hier all -file $out/vectorless_hier.rpt
report_power -file $out/vectorless_summary.rpt
report_utilization -hierarchical -file $out/impl_util_hier.rpt
report_utilization -file $out/impl_util.rpt
report_timing_summary -file $out/impl_timing.rpt
report_clock_utilization -file $out/impl_clock_util.rpt
# ---- echo the settings section so that it can be compared with our report line by line
set fh [open $out/vectorless_summary.rpt r]; set txt [read $fh]; close $fh
foreach key {"| Grade" "| Process" "| Characterization" "| Ambient Temp" "| ThetaJA" "| Airflow" "| Heat Sink" "| Board Selection" "| Board Temperature" "| Confidence Level"} {
    foreach l [split $txt "\n"] { if {[string first $key $l] == 0} { puts $l; break } }
}
puts "DONE: $out  (compare 'Settings' with C:/kws/snn/board/pkg/reports_power/vectorless_summary.rpt; KWS-related rows: hls4ml IP, axi_dma, axi_gpio*, interconnect, kws_out_latch under Block_Design_i)"
