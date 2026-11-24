// clk_gate_buf.v — clock gating primitives (4-A follow-up 4).
//   clk_gate_buf : glitch-free gated clock. Vivado/Yosys (SYNTHESIS defined): BUFGCE. Simulation: behavioural model with the
//                  same semantics as BUFGCE (CE sampled while the input clock is low, output held low when gated).
//   clk_gate_ctrl: CE handshake kept in the always-on domain. ce rises the cycle after req; `ready` rises one cycle after
//                  ce (= the gated domain has already received a clock edge), so a `start` registered on `ready` is seen
//                  by the gated block on a running clock. ce falls OFF_DELAY cycles after req falls (lets the last
//                  outputs of the gated block settle). force_on = 1 or rst = 1 keeps the clock running (reset must reach
//                  the gated registers; the sw "gating off" switch uses force_on). Timing of ready is the same with
//                  force_on, so the behaviour (golden match) is identical with gating on or off.
`timescale 1ns/1ps
module clk_gate_buf (input wire clk_in, input wire ce, output wire clk_out);
`ifdef SYNTHESIS
    BUFGCE u_bufgce (.I(clk_in), .CE(ce), .O(clk_out));
`else
    reg en_l = 1'b0;
    always @(negedge clk_in) en_l <= ce;          // CE takes effect at the next low phase (BUFGCE behaviour)
    assign clk_out = clk_in & en_l;
`endif
endmodule

module clk_gate_ctrl #(parameter OFF_DELAY = 2) (
    input  wire clk, input wire rst,
    input  wire req,            // level: the gated block is needed (wake-up request or busy)
    input  wire force_on,       // 1 = never gate (sw / ext mode)
    output wire ce,             // to clk_gate_buf
    output reg  ready,          // the gated clock has been running for >= 1 cycle: safe to issue `start`
    output reg  [31:0] cnt_ce   // cycles with ce = 1 (duty counter)
);
    reg ce_r; reg [2:0] off_cnt;
    always @(posedge clk) begin
        if (rst) begin ce_r <= 0; ready <= 0; off_cnt <= 0; cnt_ce <= 0; end
        else begin
            if (req) begin ce_r <= 1; off_cnt <= OFF_DELAY[2:0]; end
            else if (off_cnt != 0) off_cnt <= off_cnt - 1'b1;
            else ce_r <= 0;
            ready <= ce_r & req;
            if (ce) cnt_ce <= cnt_ce + 1;
        end
    end
    assign ce = ce_r | rst | force_on;
endmodule
