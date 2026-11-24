// tb_power.v — xsim test bench for SAIF generation (behavioral RTL sim). Samples from `POWER_VEC (24-bit hex, one per
// line; or the three scenario files of `POWER_DIR back to back) are shifted into the I2S interface at 16 kHz (BCLK 1.024 MHz, 64 x fs, codec-master format as in
// i2s_data_interface.v), sw = mode 2 (live), margin_sel 0, 3 s of audio.
// The DUT is the PL part of ZSK_Design_Top only (adau1761_interface + user_logic, same instance names): the Zynq block
// design (PS7, clk_wiz, proc_sys_reset) is left out because it has no simulation model and its wrappers are VHDL
// (xvhdl is blocked on the build PC). clk_48 / rstn_48 are driven by the bench. SAIF scope: tb_power/dut/user_logic_inst,
// read_saif -strip_path tb_power/dut maps it onto user_logic_inst of the implemented design.
`ifndef POWER_SW
`define POWER_SW 8'h02           // sw as seen by user_logic: mode 2 ; 8'h22 = mode 2 + clock gating off (sw[5])
`endif
`timescale 1ns/1ps
module tb_power_pl (
    input  wire clk_48, input wire rstn_48, input wire [7:0] sw,
    input  wire AC_GPIO1, input wire AC_GPIO2, input wire AC_GPIO3,
    output wire AC_MCLK, AC_GPIO0, AC_ADR0, AC_ADR1
);
    wire [23:0] audio_l_in, audio_r_in, audio_l_out, audio_r_out; wire new_sample;
    adau1761_interface adau1761_interface_inst (
        .clk_48(clk_48), .rst_48(!rstn_48), .AC_ADR0(AC_ADR0), .AC_ADR1(AC_ADR1), .AC_MCLK(AC_MCLK),
        .AC_GPIO0(AC_GPIO0), .AC_GPIO1(AC_GPIO1), .AC_GPIO2(AC_GPIO2), .AC_GPIO3(AC_GPIO3),
        .audio_l_in(audio_l_in), .audio_r_in(audio_r_in), .audio_l_out(audio_l_out), .audio_r_out(audio_r_out), .new_sample(new_sample));
    user_logic user_logic_inst (
        .clk(clk_48), .rst(!rstn_48), .sw(~sw), .audio_l_in(audio_l_in), .audio_r_in(audio_r_in),
        .audio_l_out(audio_l_out), .audio_r_out(audio_r_out), .new_sample(new_sample), .i2s_lr(AC_GPIO3), .i2s_bclk(AC_GPIO2));
endmodule

module tb_power;
    reg clk48 = 0; always #10.4167 clk48 = ~clk48;          // 48 MHz
    reg rstn = 0; initial #3000 rstn = 1;
    // I2S from the "codec": bclk = 1.024 MHz, lrclk = 16 kHz, 32-bit slots (24-bit data MSB first after 1 bclk delay)
    reg bclk = 0; always #488.28125 bclk = ~bclk;
    reg lrclk = 1; reg sdata = 0;
`ifdef POWER_DIR
    // one simulation for all scenarios (the build PC blocks a re-built xsim executable): silence 0-3 s, keyword 3-6 s, talk 6-9 s
    reg [23:0] vec [0:16000*9-1]; integer idx = 0, bit = 0; reg [63:0] fr = 0;
    initial begin
        $readmemh({`POWER_DIR, "/silence.hex"}, vec, 0, 47999);
        $readmemh({`POWER_DIR, "/keyword.hex"}, vec, 48000, 95999);
        $readmemh({`POWER_DIR, "/talk.hex"}, vec, 96000, 143999);
    end
    localparam real SIM_NS = 9.5e9;
`else
    reg [23:0] vec [0:16000*3-1]; integer idx = 0, bit = 0; reg [63:0] fr = 0;
    initial $readmemh(`POWER_VEC, vec);
    localparam real SIM_NS = 3.0e9;
`endif
    always @(negedge bclk) begin
        if (bit == 0) begin fr <= {vec[idx], 8'h00, vec[idx], 8'h00}; idx <= idx + 1; end
        lrclk <= (bit < 32) ? 1'b0 : 1'b1;       // falling edge of lrclk = frame start (left first)
        sdata <= fr[63 - bit];
        bit <= (bit == 63) ? 0 : bit + 1;
    end
    wire AC_MCLK, AC_GPIO0, AC_ADR0, AC_ADR1;
    tb_power_pl dut (.clk_48(clk48), .rstn_48(rstn), .sw(~(`POWER_SW)),   // physical sw pins are inverted in the top
        .AC_GPIO1(sdata), .AC_GPIO2(bclk), .AC_GPIO3(lrclk), .AC_MCLK(AC_MCLK), .AC_GPIO0(AC_GPIO0), .AC_ADR0(AC_ADR0), .AC_ADR1(AC_ADR1));
    initial begin
        #(SIM_NS) $finish;                        // 3 s of audio (POWER_VEC) or 9.5 s (POWER_DIR, SAIF windows set by saif_multi.tcl)
    end
endmodule
