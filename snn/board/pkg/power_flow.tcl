# power_flow.tcl — Vivado 2023.2 (Windows) power analysis of the implemented KWS design. Run AFTER build_bitstream.tcl.
#   (Vivado Tcl Shell) set argv [list C:/kws_snn/snn <variant> <gate_off 0/1> [nosim] [scenarios]] ; source power_flow.tcl
#   variant = build variant name ("" plain, popt, nogate_popt, ...), reports -> reports_power[_variant][_gateoff]/
#   scenarios: silence keyword talk (24-bit sample files snn/board/pkg/power_vec/<scenario>.hex made by snn/eval/make_power_vectors.py)
# Outputs (snn/board/pkg/reports_power/):
#   vectorless_hier.rpt           report_power -hier (no activity data)
#   <scenario>.saif, <scenario>_saif_hier.rpt   BEHAVIORAL (RTL) xsim with SAIF -> read_saif onto the implemented design ->
#                                 report_power -hier. (A post-route functional sim of the whole chip took >10 h per
#                                 scenario in xsim; RTL activity annotated by name is the practical alternative: the
#                                 registers/ports keep their names, glue nets fall back to vectorless propagation.)
# Only PL power is meaningful here (PS7 rows in the reports are IP model values, not measurements). The ILA is not in
# the RTL sim (`ifdef SYNTHESIS), so u_ila rows stay vectorless — exclude them; use the u_kws rows.
set snn [lindex $argv 0]; if {$snn eq ""} { set snn [file normalize [file join [file dirname [info script]] ../..]] }
set variant [lindex $argv 1]; set sfx [expr {$variant eq "" ? "" : "_$variant"}]     ;# same variant name as build_bitstream.tcl
set gate_off [lindex $argv 2]; if {$gate_off eq ""} { set gate_off 0 }                 ;# 1 = sw[5] on: clock gating disabled in the sim
set scen [lrange $argv 3 end]
set nosim 0; if {[lindex $scen 0] eq "nosim"} { set nosim 1; set scen [lrange $scen 1 end] }   ;# nosim = analyse existing SAIF files only
if {$scen eq ""} { set scen {silence keyword talk} }
set pkg [file dirname [file normalize [info script]]]
set rep $pkg/reports_power$sfx[expr {$gate_off ? "_gateoff" : ""}]; file mkdir $rep
if {[current_project -quiet] ne ""} { close_project }
open_project $pkg/vivado_kws$sfx/ZSK_KWS.xpr
open_run impl_1
# ---- 1. vectorless
set_property SEVERITY {Warning} [get_drc_checks NSTD-1]
report_power -hier all -file $rep/vectorless_hier.rpt
report_power -file $rep/vectorless_summary.rpt
# ---- 2/3. SAIF from post-route functional simulation (xsim), one per scenario
#   tb: snn/board/pkg/tb_power.v drives pcm samples from power_vec/<scenario>.hex into ZSK_Design_Top's user_logic
#   (mode 2 live), 3 s of audio = 144M clk_48 cycles (RTL xsim: tens of minutes per scenario).
set_property top tb_power [get_filesets sim_1]
if {[get_files -quiet -of_objects [get_filesets sim_1] $pkg/tb_power.v] eq ""} { add_files -fileset sim_1 -norecurse $pkg/tb_power.v }
# ---- BUFGCE clock-enable activity: the gated clock / CE nets are renamed in the netlist (G_CG.u_scb/clk_snn ...) and do
#      not match the RTL SAIF, so report_power would assume a free-running clock. Read the CE statistics (T0/T1/TC) of
#      snn_clk_ce / fe_clk_ce from the SAIF and set them on the BUFGCE CE pins explicitly (static probability = clock-on
#      duty, toggle rate in % of clock cycles). With gate_off the duty is 1.0, i.e. the ungated case.
proc saif_stat {saif name} {
    # CE duty from the SAIF: find "(<name>" (optionally escaped) and read T0/T1/TX/TC in any order within the next 600 chars
    set fh [open $saif r]; set data [read $fh]; close $fh
    set pos -1
    foreach pat [list "($name\n" "($name " "($name\r" "($name\t" "(\\$name " "(\\$name\n"] { set pos [string first $pat $data]; if {$pos >= 0} { break } }
    if {$pos < 0} { return {} }
    set blk [string range $data $pos [expr {$pos + 600}]]
    set t0 0; set t1 0; set tx 0; set tc 0
    regexp {\(T0\s+(\d+)\)} $blk -> t0; regexp {\(T1\s+(\d+)\)} $blk -> t1
    regexp {\(TX\s+(\d+)\)} $blk -> tx; regexp {\(TC\s+(\d+)\)} $blk -> tc
    set dur [expr {$t0 + $t1 + $tx}]; if {$dur == 0} { return {} }
    set p  [expr {double($t1) / $dur}]
    set tr [expr {100.0 * $tc / ($dur / 20833.333)}]      ;# SAIF time unit 1 ps, clk_48 period 20833.333 ps
    return [list $p $tr $t0 $t1 $tc]
}
proc hier_w {f row} {
    # total power (W) of a hierarchy row in a report_power -hier file ("<0.001" -> 0.001)
    set fh [open $f r]; set w ""
    while {[gets $fh l] >= 0} { if {[regexp "^\\|\\s+$row\\s+\\|\\s+<?(\[0-9.\]+)" $l -> v]} { set w $v; break } }
    close $fh; return $w
}
proc gating_reports {s rep saif} {
    # Vivado ignores switching activity on BUFGCE CE (Power 33-410): only set_case_analysis 0/1 is accepted. So report the
    # four steady states (SNN clock on/off x FE clock on/off) and combine them with the clock-on duties measured in the
    # RTL simulation (SAIF T1/(T0+T1) of snn_clk_ce / fe_clk_ce):  P = P00 + d_snn*(P10-P00) + d_fe*(P01-P00)  (duty-weighted estimate).
    # the BUFGCE instances are BUFGCTRL primitives after implementation (CE pin = CE0): find them by instance name
    set cs [get_cells -quiet -hierarchical -filter {NAME =~ *u_scb/u_bufgce*}]
    set cf [get_cells -quiet -hierarchical -filter {NAME =~ *u_cb/u_bufgce*}]
    if {$cs eq "" || $cf eq ""} { puts "no BUFGCE in this variant: gating reports skipped"; return }
    set ps [get_pins -quiet -of_objects $cs -filter {REF_PIN_NAME == CE0 || REF_PIN_NAME == CE}]
    set pf [get_pins -quiet -of_objects $cf -filter {REF_PIN_NAME == CE0 || REF_PIN_NAME == CE}]
    if {$ps eq "" || $pf eq ""} { puts "WARNING: CE pin not found on [get_property REF_NAME $cs] $cs / [get_property REF_NAME $cf] $cf: [get_pins -of_objects $cs]"; return }
    puts "gating pins: $ps ([get_property REF_NAME $cs]) / $pf ([get_property REF_NAME $cf])"
    set ss [saif_stat $saif snn_clk_ce]; set sf [saif_stat $saif fe_clk_ce]
    if {$ss eq "" || $sf eq ""} { puts "WARNING: snn_clk_ce / fe_clk_ce not found in $saif"; return }
    set ds [lindex $ss 0]; set df [lindex $sf 0]
    foreach {tag vs vf} {ce11 1 1 ce10 1 0 ce01 0 1 ce00 0 0} {
        set_case_analysis $vs $ps; set_case_analysis $vf $pf
        report_power -hier all -file $rep/${s}_saif_${tag}_hier.rpt
        report_power -file $rep/${s}_saif_${tag}_summary.rpt
    }
    set fh [open $rep/${s}_gating_estimate.md w]
    puts $fh "# $s — duty-weighted clock-gating estimate (W). CE duty from SAIF: snn_clk_ce [format %.5f $ds], fe_clk_ce [format %.5f $df]"
    puts $fh "# P_on = both clocks on (ce11) = gating off ; P_off = both off (ce00) ; estimate = P00 + d_snn*(P10-P00) + d_fe*(P01-P00)"
    puts $fh "| hierarchy | ce11 (on/on) | ce10 (snn on) | ce01 (fe on) | ce00 (off/off) | duty-weighted estimate |"
    puts $fh "|---|---|---|---|---|---|"
    foreach row {user_logic_inst u_kws u_fe u_snn u_ila} {
        set p11 [hier_w $rep/${s}_saif_ce11_hier.rpt $row]; set p10 [hier_w $rep/${s}_saif_ce10_hier.rpt $row]
        set p01 [hier_w $rep/${s}_saif_ce01_hier.rpt $row]; set p00 [hier_w $rep/${s}_saif_ce00_hier.rpt $row]
        if {$p11 eq "" || $p10 eq "" || $p01 eq "" || $p00 eq ""} { puts $fh "| $row | n/a |"; continue }
        set est [expr {$p00 + $ds * ($p10 - $p00) + $df * ($p01 - $p00)}]
        puts $fh [format "| %s | %.3f | %.3f | %.3f | %.3f | %.4f |" $row $p11 $p10 $p01 $p00 $est]
    }
    close $fh
    set fh [open $rep/${s}_gating_estimate.md r]; puts [read $fh]; close $fh
}
# ---- one xsim run for all scenarios (silence 0-3 s, keyword 3-6 s, talk 6-9 s): the build PC blocks the re-built xsim
#      executable of a second launch_simulation in the same shell, so the scenario windows are cut inside one run by a
#      custom simulation Tcl (open_saif / log_saif / run 3000 ms / close_saif per window).
set all {silence keyword talk}
if {$nosim} { puts "nosim: using existing SAIF files in $rep" } else {
set ctcl $rep/saif_multi.tcl
set fh [open $ctcl w]
puts $fh "current_scope /tb_power/dut/user_logic_inst"
foreach s $all {
    puts $fh "open_saif $rep/$s.saif"
    puts $fh "log_saif \[get_objects -r *\]"
    puts $fh "run 3000 ms"
    puts $fh "close_saif"
    puts $fh "puts \"SAIF window $s written\""
}
close $fh                                    ;# no "quit" here: in the Tcl shell it would exit Vivado itself (close_sim below)
set defs [list "POWER_DIR=\"$pkg/power_vec\""]
if {$gate_off} { lappend defs "POWER_SW=8'h22" }                              ;# mode 2 + sw[5] (gating off)
foreach d [get_property verilog_define [current_fileset]] { lappend defs $d } ;# e.g. NO_CLK_GATE of a nogate variant
set_property verilog_define $defs [get_filesets sim_1]
set_property -name {xsim.simulate.saif_scope} -value {tb_power/dut/user_logic_inst} -objects [get_filesets sim_1]
set_property -name {xsim.simulate.custom_tcl} -value $ctcl -objects [get_filesets sim_1]
set_property -name {xsim.simulate.runtime} -value {all} -objects [get_filesets sim_1]
set ok 0
for {set try 1} {$try <= 3 && !$ok} {incr try} {
    if {[catch {launch_simulation -mode behavioral} msg]} { puts "launch_simulation failed (try $try/3): $msg"; catch {close_sim}; after 5000 } else { set ok 1 }
}
if {!$ok} { error "simulation could not be started" }
close_sim
}
# ---- per-scenario power from the SAIF windows
foreach s $scen {
    if {![file exists $rep/$s.saif]} { puts "SKIP $s: $rep/$s.saif missing"; continue }
    open_run impl_1
    reset_switching_activity -all
    read_saif -file $rep/$s.saif -strip_path tb_power/dut -out_file $rep/${s}_saif_match.rpt
    report_power -hier all -file $rep/${s}_saif_hier.rpt
    report_power -file $rep/${s}_saif_summary.rpt
    gating_reports $s $rep $rep/$s.saif
}
puts "DONE: $rep (PL dynamic power: see the user_logic_inst rows of the -hier reports; PS7 rows are model values, exclude them)"
