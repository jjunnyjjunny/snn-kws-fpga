# vitis_xsct.tcl — xsct (Vitis 2023.2 command line) flow: platform from the XSA, standalone app with the codec
# init source, build, (optionally) program FPGA + run. Avoids the Vitis Classic GUI.
#   xsct vitis_xsct.tcl C:/kws/snn/board/pkg C:/kws/vitis_ws [48k|16k|16k_v0|16k_vA|16k_vB|16k_vC|16k_vD] [run]
#   16k_vX -> reg_set_init_16k_vX.c (B-2 codec input-gain experiments, gen_codec_variants.py)
set pkg [lindex $argv 0]; set ws [lindex $argv 1]; set mode [lindex $argv 2]; set dorun [lindex $argv 3]
if {$mode eq ""} { set mode 48k }
set src [expr {$mode eq "16k" ? "$pkg/reg_set_init_16k.c" : ($mode eq "48k" ? "$pkg/reg_set_init.c" : "$pkg/reg_set_init_$mode.c")}]
if {![file exists $src]} { puts "ERROR: $src not found"; exit 1 }
setws $ws
platform create -name kws_plat -hw $pkg/ZSK_Design_Top.xsa -proc ps7_cortexa9_0 -os standalone -out $ws
platform write
platform generate
app create -name kws_app -platform kws_plat -domain standalone_domain -template {Hello World} -lang c
file delete -force $ws/kws_app/src/helloworld.c
importsources -name kws_app -path $src
app build -name kws_app
if {$dorun eq "run"} {
    connect
    targets -set -filter {name =~ "APU*"}
    fpga -file $pkg/ZSK_Design_Top.bit
    targets -set -filter {name =~ "ARM*#0"}
    source $ws/kws_plat/hw/ps7_init.tcl
    ps7_init; ps7_post_config
    dow $ws/kws_app/Debug/kws_app.elf
    con
    puts "running kws_app ($mode) — open the UART at 115200 for the register read-back"
}
