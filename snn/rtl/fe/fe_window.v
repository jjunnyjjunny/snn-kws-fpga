// fe_window.v — block 3: 480 ring samples x Hann(Q1.15) -> sat18((x*w + 2^14) >>> 15), written to the FFT
// buffer at the bit-reversed address (imag = 0); samples 480..511 are zero (zero padding).
// Pipeline: issue ring address (stage 0) -> ring data valid two edges later (stage 2) -> multiply/round/sat -> write.
`timescale 1ns/1ps
module fe_window #(parameter DB = 18, WIN = 480, N = 512, AW = 9, HANN_MEM = "../rom/hann_w480.mem") (
    input  wire clk, input wire rst,
    input  wire start,
    output reg  done,
    output reg  [AW-1:0] ring_addr,          // relative to the frame base (added by fe_top)
    input  wire signed [DB-1:0] ring_data,   // registered ring output (valid 2 cycles after ring_addr)
    output reg           buf_we,
    output reg  [AW-1:0] buf_addr,
    output reg  [2*DB-1:0] buf_wd,
    output wire          busy
);
    reg [15:0] hann [0:WIN-1];
    initial $readmemh(HANN_MEM, hann);
    reg run; reg [AW:0] n;
    reg        p1_v, p2_v; reg [AW-1:0] p1_n, p2_n; reg [15:0] p1_w, p2_w;
    wire signed [DB+16:0] prod = ring_data * $signed({1'b0, p2_w});           // 18 x 17 signed (DSP48E1)
    wire signed [DB+16:0] rnd  = (prod + $signed({{(DB+2){1'b0}}, 15'h4000})) >>> 15;
    wire signed [DB-1:0] xw = (rnd > $signed({{17{1'b0}}, {1'b0, {(DB-1){1'b1}}}})) ? {1'b0, {(DB-1){1'b1}}} :
                              (rnd < -$signed({{17{1'b0}}, {1'b1, {(DB-1){1'b0}}}})) ? {1'b1, {(DB-1){1'b0}}} : rnd[DB-1:0];
    function [AW-1:0] bitrev; input [AW-1:0] a; integer i; begin for (i = 0; i < AW; i = i + 1) bitrev[i] = a[AW-1-i]; end endfunction
    assign busy = run | p1_v | p2_v;
`ifdef ACTIVITY
    integer act_mult = 0, act_cyc = 0, act_ring_rd = 0;
    always @(posedge clk) begin if (p2_v) act_mult = act_mult + 1; if (busy) act_cyc = act_cyc + 1; if (run) act_ring_rd = act_ring_rd + 1; end
`endif
    always @(posedge clk) begin
        done <= 1'b0; buf_we <= 1'b0;
        if (rst) begin run <= 0; n <= 0; p1_v <= 0; p2_v <= 0; end
        else begin
            if (start) begin run <= 1'b1; n <= 0; end
            p1_v <= run; p1_n <= n[AW-1:0]; p1_w <= (n < WIN) ? hann[n[AW-1:0]] : 16'd0;
            if (run) begin ring_addr <= n[AW-1:0]; if (n == N - 1) run <= 1'b0; n <= n + 1'b1; end
            p2_v <= p1_v; p2_n <= p1_n; p2_w <= p1_w;
            if (p2_v) begin
                buf_we <= 1'b1; buf_addr <= bitrev(p2_n);
                buf_wd <= (p2_n < WIN) ? {xw, {DB{1'b0}}} : {(2*DB){1'b0}};
            end
            if (p2_v && !p1_v) done <= 1'b1;
        end
    end
endmodule
