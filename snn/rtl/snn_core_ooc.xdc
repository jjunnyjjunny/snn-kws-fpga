# snn_core_ooc.xdc — out-of-context constraints for snn_core (clk_48 = 48 MHz)
create_clock -period 20.833 -name clk [get_ports clk]
set_input_delay  -clock clk 2.0 [all_inputs]
set_output_delay -clock clk 2.0 [all_outputs]
