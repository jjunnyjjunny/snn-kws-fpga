// fe_buf.v — 512 x (re 18 + im 18) true-dual-port buffer for window load / in-place FFT / mel read.
// Written as two 512x18 memories (re, im), each in the Vivado TDP "no change" template with one always block per
// port and ram_style = "block": a single 512x36 array coded in one always block was implemented as 18.4k flip-flops
// (+ ~47k LUTs of muxes) by Vivado 2023.2. Behaviour is unchanged: rd_x holds its value during a write on port x.
`timescale 1ns/1ps
module fe_buf #(parameter DB = 18, AW = 9) (
    input  wire clk,
    input  wire          en_a, we_a, input wire [AW-1:0] addr_a, input wire [2*DB-1:0] wd_a, output wire [2*DB-1:0] rd_a,
    input  wire          en_b, we_b, input wire [AW-1:0] addr_b, input wire [2*DB-1:0] wd_b, output wire [2*DB-1:0] rd_b
);
    (* ram_style = "block" *) reg [DB-1:0] mem_re [0:(1<<AW)-1];
    (* ram_style = "block" *) reg [DB-1:0] mem_im [0:(1<<AW)-1];
    reg [DB-1:0] rd_a_re, rd_a_im, rd_b_re, rd_b_im;
    assign rd_a = {rd_a_re, rd_a_im};
    assign rd_b = {rd_b_re, rd_b_im};
`ifdef ACTIVITY
    integer act_rd = 0, act_wr = 0;
    always @(posedge clk) begin if (en_a) begin if (we_a) act_wr = act_wr + 1; else act_rd = act_rd + 1; end
                                if (en_b) begin if (we_b) act_wr = act_wr + 1; else act_rd = act_rd + 1; end end
`endif
    // port A
    always @(posedge clk) if (en_a) begin if (we_a) mem_re[addr_a] <= wd_a[2*DB-1:DB]; else rd_a_re <= mem_re[addr_a]; end
    always @(posedge clk) if (en_a) begin if (we_a) mem_im[addr_a] <= wd_a[DB-1:0];    else rd_a_im <= mem_im[addr_a]; end
    // port B
    always @(posedge clk) if (en_b) begin if (we_b) mem_re[addr_b] <= wd_b[2*DB-1:DB]; else rd_b_re <= mem_re[addr_b]; end
    always @(posedge clk) if (en_b) begin if (we_b) mem_im[addr_b] <= wd_b[DB-1:0];    else rd_b_im <= mem_im[addr_b]; end
endmodule
